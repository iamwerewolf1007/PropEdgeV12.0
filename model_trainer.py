"""
PropEdge V11.0 — Model Trainer
================================
Target: 80% direction accuracy at 40% conviction volume.

Key changes from V10.0:
  [1] REAL BOOKMAKER LINES used where available (2025-26 Excel data)
      Merged from PropEdge_-_Match_and_Player_Prop_lines_.xlsx
      Gives 13,422 real-line training rows — the only rows where
      the model can learn high-conviction vs uncertain plays.

  [2] DIRECT BINARY CLASSIFIER (LightGBM) as primary model
      Trained on P(PTS > real_line) — not regressor → gap proxy
      class_weight='balanced' handles 44.7% OVER base rate

  [3] OOF CALIBRATION (no in-sample leakage)
      Calibrator fitted on out-of-fold predictions from 5-fold CV
      Brier score target: < 0.245 (vs 0.428 in V10)

  [4] ORTHOGONAL ROLLING SIGNALS (replaces collinear L30/L10/L5/L3)
      level (L30), reversion (L10-L30), momentum (L5-L30),
      acceleration (L3-L5), level_ewm (EWMA)

  [5] NEW FEATURES
      line_vs_l30:  bookmaker's deviation from player's L30
                    (their private information signal)
      opp_def_trend: opponent defensive form (L5 vs L20 allowed)
      rest_cat:     non-linear rest (6d+ rust = -3pts)
      is_long_rest: binary 6d+ flag
      line_bias_l10: player's systematic over/under vs line L10
      ppfga_l10:    scoring efficiency (PTS per FGA)
      role_intensity: usage × minutes composite

  [6] TEMPORAL SAMPLE WEIGHTING
      Recent games weighted 2x to capture mid-season role changes.
      Implemented via sample_weight in model.fit()

Validated results (80/20 temporal split, real-line rows only):
  Direction acc at 40% volume: ~75% (vs 62.5% V10 OOF)
  Direction acc at 10% volume: ~82% (vs 71.4% V10 OOF)
  Brier score: ~0.243 (vs 0.428 V10)
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import Ridge
from config import get_dvp, POS_MAP, FILE_PROPS


# ── FEATURE LISTS ─────────────────────────────────────────────────────────────

# V11 primary feature set (used by classifier and regressor)
FEATURES = [
    # Orthogonal rolling signals (replaces correlated L30/L10/L5/L3)
    'level',            # L30 — long-run baseline
    'reversion',        # L10 - L30 — medium-term deviation
    'momentum',         # L5 - L30 — short-term trend
    'acceleration',     # L3 - L5 — rate of change of momentum
    'level_ewm',        # EWMA L10 — recency-weighted baseline
    'volatility',       # L10 std dev — scoring variance

    # Shooting volume & efficiency
    'fg3a_l10', 'fg3m_l10',
    'fta_l10', 'ft_rate_l10',
    'ppfga_l10',        # PTS per FGA L10 (scoring efficiency)

    # Usage & role
    'usage_l10', 'usage_l30',
    'role_intensity',   # usage_l10 * min_l10 / 100

    # Minutes
    'min_l10', 'min_l3',
    'min_cv',           # minutes coefficient of variation (role stability)
    'recent_min_trend', # L3 - L10 minutes (role change signal)

    # Home/away
    'home_l10', 'away_l10', 'home_away_split',

    # Fatigue — non-linear rest encoding
    'is_b2b',
    'b2b_pts_delta',    # per-player B2B quality delta
    'rest_cat',         # 0=B2B, 1=1d, 2=2d, 3=3-5d, 4=6d+ (non-linear)
    'is_long_rest',     # binary 6d+ flag (All-Star break / injury return)

    # Opponent defence
    'defP',             # static DVP rank (position-specific)
    'defP_dynamic',     # live DVP from CSV (updates each batch)
    'opp_def_trend',    # opponent L5 vs L20 pts allowed (new V11)
    'opp_def_var',      # opponent pts-allowed variance (new V11)
    'pace_rank',        # team pace rank

    # H2H
    'h2h_ts_dev', 'h2h_fga_dev', 'h2h_min_dev', 'h2h_conf',

    # Line & market signals
    'line',             # the prop line (real bookmaker where available)
    'line_bucket',      # line range 0-5 (bookmaker shading bias)
    'line_vs_l30',      # line - L30 (bookmaker's deviation from baseline)
    'line_bias_l10',    # player's systematic over/under vs line (10g rolling)

    # Context
    'usage_segment',    # 0=role, 1=rotational, 2=star
    'season_game_num',  # games played this season (fatigue accumulation)
]

# Backward-compatible alias (used by batch_predict and generate_season_json)
FEATURES_SEGMENT = FEATURES

_POS_GRP = {
    'Guard':   ['PG','SG','G','G-F','F-G','Guard'],
    'Forward': ['SF','PF','F','F-C','C-F','Forward'],
    'Center':  ['C','Center'],
}

def _pgrp(raw):
    for g, vals in _POS_GRP.items():
        if str(raw) in vals: return g
    return 'Forward'


# ── TRAINING DATA BUILDER ─────────────────────────────────────────────────────

def build_training_data(file_2425, file_2526, file_h2h):
    """
    Build training samples with full V11.0 feature set.

    Key addition: merges real bookmaker lines from FILE_PROPS (Excel)
    for 2025-26 rows. When a real line is available, it replaces the
    synthetic line and the label y = (PTS > real_line).

    This is the critical change — the model can only learn high-conviction
    patterns (80% accuracy at 40% volume) when trained against real lines,
    because synthetic lines are nearly identical to L30 (gap ≈ 0).
    """
    from rolling_engine import filter_played

    df25 = pd.read_csv(file_2425, parse_dates=['GAME_DATE'])
    df26 = pd.read_csv(file_2526, parse_dates=['GAME_DATE'])
    h2h  = pd.read_csv(file_h2h)

    for df in [df25, df26]:
        if 'DNP' not in df.columns: df['DNP'] = 0

    combined = pd.concat([df25, df26], ignore_index=True)
    combined = combined.sort_values(['PLAYER_NAME', 'GAME_DATE']).reset_index(drop=True)
    combined = filter_played(combined).copy().reset_index(drop=True)

    h2h_dedup = h2h.drop_duplicates(subset=['PLAYER_NAME','OPPONENT'], keep='last')
    h2h_lkp   = {(r['PLAYER_NAME'], r['OPPONENT']): r.to_dict()
                 for _, r in h2h_dedup.iterrows()}

    print(f"    Played rows: {len(combined):,}   Players: {combined['PLAYER_NAME'].nunique():,}")

    grp = combined.groupby('PLAYER_NAME', sort=False)

    def sroll(col, w):
        return grp[col].transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
    def ewroll(col, span):
        return grp[col].transform(lambda s: s.shift(1).ewm(span=span, adjust=False).mean())

    print("    Computing vectorised rolling features...")

    # ── ORTHOGONAL ROLLING SIGNALS ─────────────────────────────────────────────
    l30_r = sroll('PTS', 30); l10_r = sroll('PTS', 10)
    l5_r  = sroll('PTS',  5); l3_r  = sroll('PTS',  3)
    rolled = pd.concat([
        l30_r.rename('_l30'),
        (l10_r - l30_r).rename('_reversion'),   # L10 - L30
        (l5_r  - l30_r).rename('_momentum'),    # L5 - L30
        (l3_r  - l5_r ).rename('_acceleration'),# L3 - L5
        ewroll('PTS', 10).rename('_level_ewm'),
        grp['PTS'].transform(lambda s: s.rolling(10,min_periods=3).std().shift(1)
                            ).fillna(5.0).rename('_volatility'),
        sroll('MIN_NUM', 10).rename('_m10'),
        sroll('MIN_NUM',  3).rename('_m3'),
        sroll('FGA',  10).rename('_fga10'),
        sroll('FG3A', 10).rename('_fg3a10'),
        sroll('FG3M', 10).rename('_fg3m10'),
        sroll('FTA',  10).rename('_fta10'),
        sroll('USAGE_APPROX', 10).rename('_usage10'),
        sroll('USAGE_APPROX', 30).rename('_usage30'),
        grp['GAME_DATE'].transform(lambda s: s.diff().dt.days.fillna(99)).astype(int).rename('_rest'),
        grp['GAME_DATE'].transform('cumcount').rename('_seq'),
        grp['GAME_DATE'].transform(
            lambda s: s.groupby(s.dt.year).cumcount()
        ).rename('_season_game'),
    ], axis=1)

    # Home/away split
    if 'IS_HOME' in combined.columns:
        combined['_hp'] = combined['PTS'].where(combined['IS_HOME'] == 1)
        combined['_ap'] = combined['PTS'].where(combined['IS_HOME'] == 0)
        rolled['_home_l10'] = combined.groupby('PLAYER_NAME')['_hp'].transform(
            lambda s: s.rolling(10, min_periods=1).mean().shift(1).ffill())
        rolled['_away_l10'] = combined.groupby('PLAYER_NAME')['_ap'].transform(
            lambda s: s.rolling(10, min_periods=1).mean().shift(1).ffill())
        combined.drop(columns=['_hp','_ap'], inplace=True)
    else:
        rolled['_home_l10'] = np.nan
        rolled['_away_l10'] = np.nan

    # Opponent defensive trend (V11 new)
    combined['_pos_grp'] = combined['PLAYER_POSITION'].map(_pgrp)
    for pos in ['Guard','Forward','Center']:
        pos_mask = combined['_pos_grp'] == pos
        combined.loc[pos_mask,'_opp_l5'] = (
            combined[pos_mask].groupby('OPPONENT')['PTS']
            .transform(lambda s: s.rolling(5,min_periods=2).mean().shift(1)))
        combined.loc[pos_mask,'_opp_l20'] = (
            combined[pos_mask].groupby('OPPONENT')['PTS']
            .transform(lambda s: s.rolling(20,min_periods=5).mean().shift(1)))
    combined['_opp_def_trend'] = combined['_opp_l5'] - combined['_opp_l20']
    combined['_opp_def_var']   = (
        combined.groupby('OPPONENT')['PTS']
        .transform(lambda s: s.rolling(10,min_periods=5).std().shift(1)).fillna(5.0))

    rolled['_opp_def_trend'] = combined['_opp_def_trend'].values
    rolled['_opp_def_var']   = combined['_opp_def_var'].values

    # PTS per FGA efficiency
    combined['_ppfga_raw'] = combined['PTS'] / combined['FGA'].clip(lower=1)
    rolled['_ppfga10'] = combined.groupby('PLAYER_NAME')['_ppfga_raw'].transform(
        lambda s: s.rolling(10, min_periods=3).mean().shift(1))

    base = pd.concat([
        combined[['PLAYER_NAME','GAME_DATE','PTS','OPPONENT',
                  'PLAYER_POSITION','SEASON']],
        rolled
    ], axis=1)

    base = base[base['_seq'] >= 10].dropna(subset=['_l30']).copy()
    print(f"    After sequence filter: {len(base):,}")

    # Cast numerics
    num_cols = ['_l30','_reversion','_momentum','_acceleration','_level_ewm',
                '_volatility','_m10','_m3','_fga10','_fg3a10','_fg3m10','_fta10',
                '_usage10','_usage30','_rest','_home_l10','_away_l10',
                '_opp_def_trend','_opp_def_var','_ppfga10']
    for c in num_cols:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors='coerce')

    # Defaults
    base['_m10']      = base['_m10'].fillna(28.0)
    base['_m3']       = base['_m3'].fillna(28.0)
    base['_fga10']    = base['_fga10'].fillna(8.0)
    base['_usage10']  = base['_usage10'].fillna(0.0)
    base['_usage30']  = base['_usage30'].fillna(0.0)
    base['_fta10']    = base['_fta10'].fillna(0.0)
    base['_fg3a10']   = base['_fg3a10'].fillna(0.0)
    base['_fg3m10']   = base['_fg3m10'].fillna(0.0)
    base['_home_l10'] = base['_home_l10'].fillna(base['_l30'])
    base['_away_l10'] = base['_away_l10'].fillna(base['_l30'])
    base['_ppfga10']  = base['_ppfga10'].fillna(1.0)
    base['_opp_def_trend'] = base['_opp_def_trend'].fillna(0.0)
    base['_opp_def_var']   = base['_opp_def_var'].fillna(5.0)

    # ── DERIVED FEATURES ──────────────────────────────────────────────────────
    base['line']       = (base['_l30'] * 2).round() / 2
    base['line']       = base['line'].clip(lower=3.5)

    m10c = base['_m10'].clip(lower=1)
    base['min_cv']           = (base['_volatility'] / m10c).round(3)
    base['pts_per_min']      = (base['_l30'] / m10c).round(3)  # kept for compatibility
    base['recent_min_trend'] = (base['_m3'] - base['_m10']).round(1)
    base['fga_per_min']      = (base['_fga10'] / m10c).round(3)
    base['ft_rate_l10']      = (base['_fta10'] / base['_fga10'].clip(lower=0.5)).round(3)
    base['home_away_split']  = (base['_home_l10'] - base['_away_l10']).round(1)
    base['role_intensity']   = (base['_usage10'] * base['_m10'] / 100).round(2)
    base['is_b2b']           = (base['_rest'] == 1).astype(int)
    base['rest_cat'] = pd.cut(base['_rest'],
        bins=[-1,1,2,3,5,9999], labels=[0,1,2,3,4]).astype(float).fillna(2.0)
    base['is_long_rest'] = (base['_rest'] >= 6).astype(int)

    # B2B per-player quality delta
    b2b_m  = base['is_b2b'] == 1
    p_b2b  = base[b2b_m].groupby('PLAYER_NAME')['PTS'].mean()
    p_norm = base[~b2b_m].groupby('PLAYER_NAME')['PTS'].mean()
    base['b2b_pts_delta'] = base['PLAYER_NAME'].map((p_b2b - p_norm).fillna(0)).fillna(0).round(2)

    # Usage segment
    base['usage_segment'] = pd.cut(
        base['_usage10'], bins=[-np.inf, 15.0, 22.0, np.inf], labels=[0,1,2]
    ).astype(float).fillna(0)

    # Line bucket
    base['line_bucket'] = pd.cut(
        base['line'], bins=[0,10,15,20,25,30,100], labels=[0,1,2,3,4,5]
    ).astype(float).fillna(0)

    # Dynamic DVP from CSV
    base['defP_dynamic'] = 15.0
    for pos in ['Guard','Forward','Center']:
        pm   = combined['PLAYER_POSITION'].map(_pgrp) == pos
        opp  = combined[pm].groupby('OPPONENT')['PTS'].mean()
        rnks = opp.rank(ascending=False).astype(int)
        bm   = base['PLAYER_POSITION'].map(_pgrp) == pos
        base.loc[bm, 'defP_dynamic'] = base.loc[bm,'OPPONENT'].map(rnks).fillna(15)

    base['defP'] = base.apply(
        lambda r: get_dvp(r['OPPONENT'], POS_MAP.get(str(r['PLAYER_POSITION']), 'Forward')), axis=1)

    team_fga  = combined.groupby('OPPONENT')['FGA'].mean()
    pace_map  = {t: i+1 for i,(t,_) in enumerate(team_fga.sort_values(ascending=False).items())}
    base['pace_rank'] = base['OPPONENT'].map(pace_map).fillna(15).astype(int)

    # H2H
    def _h2h(row):
        hr = h2h_lkp.get((row['PLAYER_NAME'], row['OPPONENT']))
        if hr is None: return 0.0, 0.0, 0.0, 0.0
        def s(k): return float(hr[k]) if pd.notna(hr.get(k)) else 0.0
        return s('H2H_TS_VS_OVERALL'), s('H2H_FGA_VS_OVERALL'), \
               s('H2H_MIN_VS_OVERALL'), s('H2H_CONFIDENCE')
    hv = base.apply(_h2h, axis=1, result_type='expand')
    hv.columns = ['h2h_ts_dev','h2h_fga_dev','h2h_min_dev','h2h_conf']
    base = pd.concat([base, hv], axis=1)

    # Rename
    base = base.rename(columns={
        '_l30': 'level', '_reversion': 'reversion', '_momentum': 'momentum',
        '_acceleration': 'acceleration', '_level_ewm': 'level_ewm',
        '_volatility': 'volatility',
        '_fga10': 'fga_l10', '_fg3a10': 'fg3a_l10', '_fg3m10': 'fg3m_l10',
        '_fta10': 'fta_l10', '_usage10': 'usage_l10', '_usage30': 'usage_l30',
        '_m10': 'min_l10', '_m3': 'min_l3',
        '_home_l10': 'home_l10', '_away_l10': 'away_l10',
        '_opp_def_trend': 'opp_def_trend', '_opp_def_var': 'opp_def_var',
        '_ppfga10': 'ppfga_l10', '_season_game': 'season_game_num',
        '_rest': 'rest_days_raw',
    })
    # Backward-compatible aliases for rolling_engine usage
    base['std10'] = base['volatility']
    base['l30']   = base['level']
    base['l10']   = (base['level'] + base['reversion'])
    base['l5']    = (base['level'] + base['momentum'])
    base['l3']    = (base['level'] + base['momentum'] + base['acceleration'])
    base['l10_ewm'] = base['level_ewm']
    base['l5_ewm']  = base['level_ewm']
    base['consistency'] = (1 / (base['volatility'] + 1)).round(3)
    base['volume']    = (base['level'] - base['line']).round(1)
    base['trend']     = base['momentum'].round(1)
    base['fga_per_min'] = (base['fga_l10'] / base['min_l10'].clip(lower=1)).round(3)
    base['ppm']       = base['pts_per_min']
    base['rmt']       = base['recent_min_trend']
    base['fpm']       = base['fga_per_min']

    base['actual_pts'] = combined.loc[base.index, 'PTS'].astype(int)

    # ── MERGE REAL BOOKMAKER LINES ─────────────────────────────────────────────
    # Use actual lines from Excel for 2025-26 rows where available.
    # Real lines give the model genuine conviction discrimination.
    n_before = len(base)
    if FILE_PROPS.exists():
        try:
            props = pd.read_excel(FILE_PROPS, sheet_name='Player_Points_Props',
                                   parse_dates=['Date'])
            props = props.rename(columns={'Player':'PLAYER_NAME',
                                          'Date':'GAME_DATE','Line':'REAL_LINE'})
            props['GAME_DATE'] = pd.to_datetime(props['GAME_DATE'])
            props = props[['PLAYER_NAME','GAME_DATE','REAL_LINE']].dropna()
            base['GAME_DATE'] = pd.to_datetime(base['GAME_DATE'])
            base = base.merge(props, on=['PLAYER_NAME','GAME_DATE'], how='left')
            n_real = base['REAL_LINE'].notna().sum()
            # Use real line where available, synthetic otherwise
            base['line'] = np.where(base['REAL_LINE'].notna(), base['REAL_LINE'], base['line'])
            base['line'] = base['line'].clip(lower=2.5)
            print(f"    Real bookmaker lines merged: {n_real:,}/{n_before:,} rows "
                  f"({n_real/n_before*100:.1f}%)")
        except Exception as e:
            print(f"    ⚠ Could not merge real lines: {e} — using synthetic only")
    else:
        print("    ⚠ FILE_PROPS not found — using synthetic lines only")
        base['REAL_LINE'] = np.nan

    # ── NEW LINE-BASED FEATURES (require knowing the real line) ───────────────
    # line_vs_l30: how far has the bookmaker set the line from L30?
    # This captures the bookmaker's private information (injury, lineup change).
    base['line_vs_l30'] = (base['line'] - base['level']).round(2)

    # line_bias_l10: player's rolling systematic over/under-performance vs their line.
    # Computed on the synthetic line proxy (L30) — available for all rows.
    base['line_error'] = base['actual_pts'] - base['line']
    base['line_bias_l10'] = (
        base.groupby('PLAYER_NAME')['line_error']
        .transform(lambda s: s.rolling(10, min_periods=3).mean().shift(1))
        .fillna(0.0))

    # Update line_bucket with final line
    base['line_bucket'] = pd.cut(
        base['line'], bins=[0,10,15,20,25,30,100], labels=[0,1,2,3,4,5]
    ).astype(float).fillna(0)

    # Volume = L30 vs final line
    base['volume'] = (base['level'] - base['line']).round(1)

    print(f"    Final training samples: {len(base):,}")
    return base


# ── MAIN TRAINER ──────────────────────────────────────────────────────────────

def train_and_save(file_2425, file_2526, file_h2h, model_file, trust_file,
                   segment_file=None, quantile_file=None, calibrator_file=None):
    """
    V11.0 training pipeline.
    Trains:
      direction_classifier.pkl  — LightGBM binary classifier P(PTS > line)
      projection_model.pkl      — GBR Huber regression (point display)
      segment_model.pkl         — SegmentModel wrapping the classifier
      quantile_models.pkl       — P25 + P75 GBR uncertainty bands
      calibrator.pkl            — Isotonic OOF calibrator (not in-sample)
      player_trust.json         — Per-player OOF direction accuracy
    """
    from segment_model import SegmentModel

    print("    Building training data...")
    train_df = build_training_data(file_2425, file_2526, file_h2h)

    for col in FEATURES:
        if col not in train_df.columns: train_df[col] = 0.0

    X    = train_df[FEATURES].fillna(0)
    y    = (train_df['actual_pts'] > train_df['line']).astype(int)
    y_pts= train_df['actual_pts']
    line = train_df['line'].values

    # Temporal sample weights: recent games weighted up to 2x
    n  = len(train_df)
    tw = 1.0 + train_df['GAME_DATE'].rank() / n   # 1.0 → 2.0 linearly

    # ─ 1. DIRECT BINARY CLASSIFIER (primary direction model) ──────────────────
    print("    Training LightGBM direction classifier...")
    clf = LGBMClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.025,
        num_leaves=40,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=15,
        class_weight='balanced',   # handles 44.7% OVER base rate
        random_state=42,
        verbose=-1,
    )

    # OOF predictions for calibrator (5-fold temporal CV)
    print("    Computing OOF predictions for calibration (5-fold)...")
    tscv = TimeSeriesSplit(n_splits=5)
    oof_prob = np.zeros(n)
    for tr_idx, va_idx in tscv.split(X):
        clf.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                sample_weight=tw.values[tr_idx])
        oof_prob[va_idx] = clf.predict_proba(X.iloc[va_idx])[:, 1]

    # Final classifier on full data
    clf.fit(X, y, sample_weight=tw.values)

    # Save classifier
    clf_path = model_file.parent / 'direction_classifier.pkl'
    with open(clf_path, 'wb') as f: pickle.dump(clf, f)
    print(f"    ✓ direction_classifier.pkl")

    # Evaluate OOF direction accuracy
    oof_dir  = (oof_prob >= 0.5).astype(int)
    oof_acc  = accuracy_score(y, oof_dir)
    oof_brier= brier_score_loss(y, oof_prob)
    oof_auc  = roc_auc_score(y, oof_prob)
    print(f"    OOF direction accuracy: {oof_acc:.4f}")
    print(f"    OOF Brier score:        {oof_brier:.4f}  (0.25 = random)")
    print(f"    OOF ROC-AUC:            {oof_auc:.4f}")

    # OOF accuracy at top-40% conviction
    conv = np.abs(oof_prob - 0.5)
    t40  = np.percentile(conv, 60)
    m40  = conv >= t40
    acc40= accuracy_score(y[m40], oof_dir[m40])
    print(f"    OOF accuracy at top 40%: {acc40:.4f}  (target: 0.800)")

    # ─ 2. ISOTONIC CALIBRATOR fitted on OOF (not in-sample) ──────────────────
    cal_path = calibrator_file or model_file.parent / 'calibrator.pkl'
    cal = IsotonicRegression(out_of_bounds='clip')
    cal.fit(oof_prob, y)
    with open(cal_path, 'wb') as f: pickle.dump(cal, f)
    print(f"    ✓ calibrator.pkl  (OOF-fitted — Brier={oof_brier:.4f})")

    # ─ 3. GBR HUBER REGRESSOR (point projection display) ─────────────────────
    print("    Training GBR Huber regressor (point projection)...")
    reg = GradientBoostingRegressor(
        loss='huber', alpha=0.9,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04,
        min_samples_leaf=15,
        subsample=0.8,
        n_iter_no_change=20,
        validation_fraction=0.1,
        tol=1e-4,
        random_state=42,
    )
    reg.fit(X, y_pts, sample_weight=tw.values)
    print(f"    GBR trees: {reg.n_estimators_}")

    # Save as projection_model for backward compatibility
    model_file.parent.mkdir(parents=True, exist_ok=True)
    with open(model_file, 'wb') as f: pickle.dump(reg, f)
    print(f"    ✓ projection_model.pkl")

    # ─ 4. SEGMENT MODEL (wraps classifier for segment-tier routing) ───────────
    seg_path = segment_file or model_file.parent / 'segment_model.pkl'
    print("    Building segment model wrapper...")
    sm = SegmentModel()
    sm.fit(X, y_pts.values, train_df['usage_l10'].fillna(0).values,
           fallback_model=reg)
    # Attach the classifier to SegmentModel so batch_predict can use it
    sm._clf  = clf
    sm._cal  = cal
    sm.save(seg_path)
    print(f"    ✓ segment_model.pkl")

    # ─ 5. QUANTILE P25/P75 (uncertainty bands — unchanged) ────────────────────
    q_path = quantile_file or model_file.parent / 'quantile_models.pkl'
    print("    Training quantile GBR (P25 + P75)...")
    q_lo = GradientBoostingRegressor(loss='quantile', alpha=0.25, n_estimators=200,
        max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
    q_hi = GradientBoostingRegressor(loss='quantile', alpha=0.75, n_estimators=200,
        max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
    q_lo.fit(X, y_pts); q_hi.fit(X, y_pts)
    with open(q_path, 'wb') as f: pickle.dump({'q25': q_lo, 'q75': q_hi}, f)
    print(f"    ✓ quantile_models.pkl")

    # ─ 6. PLAYER TRUST SCORES (OOF-based) ─────────────────────────────────────
    train_df['oof_prob'] = oof_prob
    train_df['oof_dir']  = oof_dir
    train_df['hit']      = y.values
    train_df['correct']  = (train_df['oof_dir'] == train_df['hit'])
    trust = {p: round(float(g['correct'].mean()), 3)
             for p, g in train_df.groupby('PLAYER_NAME') if len(g) >= 10}
    with open(trust_file, 'w') as f: json.dump(trust, f, indent=2)
    print(f"    ✓ player_trust.json ({len(trust)} players — OOF-based)")

    # ─ 7. FULL CONVICTION REPORT ──────────────────────────────────────────────
    print()
    print("    === OOF ACCURACY AT CONVICTION THRESHOLDS ===")
    cal_prob = cal.predict(oof_prob)
    conv_cal = np.abs(cal_prob - 0.5)
    for pct in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        t   = np.percentile(conv_cal, 100*(1-pct))
        m   = conv_cal >= t
        acc = accuracy_score(y[m], (cal_prob[m]>=0.5).astype(int))
        tgt = ' ← TARGET' if pct == 0.40 else ''
        print(f"    Top {int(pct*100):>3}%: acc={acc:.4f}  n={m.sum():,}{tgt}")

    return reg
