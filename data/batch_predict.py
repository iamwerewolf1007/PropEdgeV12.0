#!/usr/bin/env python3
"""
PropEdge V12.0 — BATCH PREDICT
================================
Usage: python3 batch_predict.py [1|2|3] [YYYY-MM-DD]

V11 changes:
  - Direction: LightGBM classifier P(PTS > line) with OOF calibration
  - Conviction: |P(OVER) - 0.5|  (replaces raw regression gap)
  - New features: opp_def_trend, rest_cat, is_long_rest, line_vs_l30,
                  line_bias_l10, ppfga_l10, role_intensity
  - Excel append: PropEdge_-_Match_and_Player_Prop_lines_.xlsx updated every batch
"""
import pandas as pd
import numpy as np
import json, sys, time, pickle, requests, unicodedata, re
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from config import *
from audit import log_event, log_file_state, verify_no_deletion, log_batch_summary
from rolling_engine import (load_combined, build_player_index,
    get_prior_games_played, extract_prediction_features,
    filter_played, build_b2b_delta, build_dynamic_dvp)
from reasoning_engine import generate_pre_match_reason

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] in ('1','2','3') else 2

_NICKNAMES = {
    'nic':'nicolas','nick':'nicolas','herb':'herbert','moe':'mohamed',
    'cam':'cameron','drew':'andrew','alex':'alexander','will':'william',
    'kenny':'kenyon','mo':'mohamed','greg':'gregory','matt':'matthew',
    'mike':'michael','chris':'christopher','jon':'jonathan','joe':'joseph',
    'ben':'benjamin','dan':'daniel','dave':'david','rob':'robert',
    'bob':'robert','ed':'edward','jeff':'jeffrey','jake':'jacob',
    'tony':'anthony','tj':'tj','cj':'cj','pj':'pj','aj':'aj',
}

def _norm(n):
    n = unicodedata.normalize('NFKD',str(n)).encode('ascii','ignore').decode()
    n = n.replace('.','').replace("'",'').strip()
    n = re.sub(r'\s+',' ',n)
    n = re.sub(r'\s+(Jr|Sr|II|III|IV|V)\s*$','',n,flags=re.IGNORECASE)
    return n.lower().strip()

def build_name_map(pidx): return {_norm(k):k for k in pidx}

def resolve_name(odds_name,pidx,name_map):
    if odds_name in pidx: return odds_name
    n=_norm(odds_name)
    if n in name_map: return name_map[n]
    for sfx in ['jr','sr','ii','iii','iv']:
        if n+' '+sfx in name_map: return name_map[n+' '+sfx]
    parts=odds_name.strip().split()
    if len(parts)>=2:
        first=parts[0].lower()
        if first in _NICKNAMES:
            en=_norm(_NICKNAMES[first]+' '+' '.join(parts[1:]))
            if en in name_map: return name_map[en]
    return None

def append_to_excel(games,date_str):
    prop_rows=[]; spread_rows=[]
    for eid,g in games.items():
        ht=g['home']; at=g['away']; ms=f"{at} @ {ht}"
        spread_rows.append({'Date':pd.Timestamp(date_str),'Game_Time_ET':str(g.get('gt','')).strip(),
            'Game':ms,'Home':ht,'Away':at,'Spread (Home)':g.get('spread'),
            'Spread Home Odds':g.get('spread_home_odds'),'Spread Away Odds':g.get('spread_away_odds'),
            'Total':g.get('total'),'Over Odds':g.get('total_over_odds'),
            'Under Odds':g.get('total_under_odds'),
            'Commence':str(g.get('ts','')).strip(),'Book':'consensus','Event ID':str(eid)})
        for pname,pd_ in g['props'].items():
            prop_rows.append({'Date':pd.Timestamp(date_str),'Game_Time_ET':str(g.get('gt','')).strip(),
                'Player':str(pname).strip(),'Position':'','Game':ms,'Home':ht,'Away':at,
                'Line':pd_.get('line'),'Over Odds':pd_.get('over'),'Under Odds':pd_.get('under'),
                'Books':pd_.get('books',1),'Min Line':pd_.get('min_line',pd_.get('line')),
                'Max Line':pd_.get('max_line',pd_.get('line')),
                'Commence':str(g.get('ts','')).strip(),'Event ID':str(eid)})
    if not prop_rows and not spread_rows: print("  no data to append"); return
    new_p=pd.DataFrame(prop_rows); new_s=pd.DataFrame(spread_rows)
    SOURCE_DIR.mkdir(parents=True,exist_ok=True)
    if FILE_PROPS.exists():
        try:
            ep=pd.read_excel(FILE_PROPS,sheet_name='Player_Points_Props')
            es=pd.read_excel(FILE_PROPS,sheet_name='Team_Spreads_Totals')
        except:
            ep=pd.DataFrame(columns=new_p.columns); es=pd.DataFrame(columns=new_s.columns)
    else:
        ep=pd.DataFrame(columns=new_p.columns); es=pd.DataFrame(columns=new_s.columns)
    rb_p=len(ep); rb_s=len(es)
    def cl(df,sc,nc):
        df=df.copy()
        if 'Date' in df.columns: df['Date']=pd.to_datetime(df['Date'],errors='coerce')
        for c in sc:
            if c in df.columns: df[c]=df[c].astype(str).str.strip().replace({'nan':'','None':'','NaT':''})
        for c in nc:
            if c in df.columns: df[c]=pd.to_numeric(df[c],errors='coerce')
        return df
    p_sc=['Player','Position','Game','Home','Away','Game_Time_ET','Commence','Event ID']
    p_nc=['Line','Over Odds','Under Odds','Books','Min Line','Max Line']
    s_sc=['Game','Home','Away','Game_Time_ET','Commence','Book','Event ID']
    s_nc=['Spread (Home)','Spread Home Odds','Spread Away Odds','Total','Over Odds','Under Odds']
    ep=cl(ep,p_sc,p_nc); es=cl(es,s_sc,s_nc)
    new_p=cl(new_p,p_sc,p_nc); new_s=cl(new_s,s_sc,s_nc)
    new_p=new_p.dropna(subset=['Line']); new_p=new_p[new_p['Game'].str.strip()!='']
    new_s=new_s[new_s['Game'].str.strip()!='']
    def dm(ex,nd,kc):
        if ex.empty: return nd.reset_index(drop=True)
        mk=lambda df:df[kc].astype(str).apply(lambda r:'|'.join(r.values),axis=1)
        kept=ex[~mk(ex).isin(mk(nd))]
        return pd.concat([kept,nd],ignore_index=True).sort_values(['Date']+kc[1:]).reset_index(drop=True)
    mp=dm(ep,new_p,['Date','Player','Game']); ms_=dm(es,new_s,['Date','Game'])
    try:
        with pd.ExcelWriter(FILE_PROPS,engine='openpyxl') as w:
            mp.to_excel(w,sheet_name='Player_Points_Props',index=False)
            ms_.to_excel(w,sheet_name='Team_Spreads_Totals',index=False)
        print(f"  ✓ Excel: Player_Points_Props {rb_p:,}→{len(mp):,} (+{len(mp)-rb_p}) | Spreads {rb_s:,}→{len(ms_):,} (+{len(ms_)-rb_s})")
        log_event(f'B{BATCH}','EXCEL_UPDATED',detail=f'props={len(mp)} spreads={len(ms_)}')
    except Exception as e:
        print(f"  ⚠ Excel write failed: {e}")

def _check_credits(h,l=''):
    r=h.get('x-requests-remaining','?')
    print(f"    Credits: {r} {l}")
    if r!='?' and int(r)<=CREDIT_ALERT: print("    ⚠ LOW CREDITS")

def fetch_props(date_str):
    print(f"\n  Fetching props: {date_str} (B{BATCH})")
    d=datetime.strptime(date_str,'%Y-%m-%d')
    fr=(d-timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%SZ')
    to=(d+timedelta(hours=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
    r1=requests.get(f"{ODDS_API_BASE}/sports/{SPORT}/events",
        params={'apiKey':ODDS_API_KEY,'dateFormat':'iso','commenceTimeFrom':fr,'commenceTimeTo':to},timeout=30)
    r1.raise_for_status(); _check_credits(r1.headers,'events')
    et=get_et()
    events=[e for e in r1.json()
        if datetime.fromisoformat(e['commence_time'].replace('Z','+00:00')).astimezone(et).strftime('%Y-%m-%d')==date_str]
    print(f"    {len(events)} games"); 
    if not events: return {},[]
    games={}
    for e in events:
        eid=e['id']; hr=e['home_team']; ar=e['away_team']; ts=e['commence_time']
        try: gt=datetime.fromisoformat(ts.replace('Z','+00:00')).astimezone(et).strftime('%-I:%M %p ET')
        except: gt=''
        ht=resolve_abr(hr); at=resolve_abr(ar)
        games[eid]={'home':ht,'away':at,'home_raw':hr,'away_raw':ar,'gt':gt,'ts':ts,
            'spread':None,'spread_home_odds':None,'spread_away_odds':None,
            'total':None,'total_over_odds':None,'total_under_odds':None,
            'props':{},'_prop_lines':{}}
    for eid,g in games.items():
        time.sleep(0.3)
        try:
            r2=requests.get(f"{ODDS_API_BASE}/sports/{SPORT}/events/{eid}/odds",
                params={'apiKey':ODDS_API_KEY,'regions':'us',
                        'markets':'player_points,spreads,totals','oddsFormat':'american','dateFormat':'iso'},timeout=30)
            r2.raise_for_status(); _check_credits(r2.headers)
            hr=g['home_raw']
            for bm in r2.json().get('bookmakers',[]):
                for m in bm.get('markets',[]):
                    mk=m.get('key','')
                    if mk=='spreads':
                        for o in m.get('outcomes',[]):
                            if o.get('name')==hr and g['spread'] is None:
                                g['spread']=o.get('point'); g['spread_home_odds']=o.get('price')
                            elif o.get('name')!=hr and g['spread_away_odds'] is None:
                                g['spread_away_odds']=o.get('price')
                    elif mk=='totals':
                        for o in m.get('outcomes',[]):
                            nm=o.get('name','').upper()
                            if nm=='OVER' and g['total'] is None:
                                g['total']=o.get('point'); g['total_over_odds']=o.get('price')
                            elif nm=='UNDER' and g['total_under_odds'] is None:
                                g['total_under_odds']=o.get('price')
                    elif mk=='player_points':
                        for o in m.get('outcomes',[]):
                            pl=(o.get('description') or '').strip() or o.get('name','').strip()
                            pt=o.get('point'); sd=o.get('name','').upper(); pr=o.get('price')
                            if not pl or pt is None: continue
                            if pl not in g['_prop_lines']: g['_prop_lines'][pl]=[]
                            if sd=='OVER': g['_prop_lines'][pl].append(pt)
                            if pl not in g['props']:
                                g['props'][pl]={'line':pt,'over':None,'under':None,'books':0,'min_line':pt,'max_line':pt}
                            if sd=='OVER': g['props'][pl]['over']=pr; g['props'][pl]['books']+=1
                            elif sd=='UNDER': g['props'][pl]['under']=pr
            for pl,lines in g['_prop_lines'].items():
                if lines and pl in g['props']:
                    g['props'][pl]['min_line']=min(lines); g['props'][pl]['max_line']=max(lines)
            print(f"    ✓ {g['away']} @ {g['home']}: {len(g['props'])} props")
        except Exception as ex:
            print(f"    ✗ {g['away_raw']} @ {g['home_raw']}: {ex}"); time.sleep(1)
    tp=sum(len(g['props']) for g in games.values())
    print(f"  Total: {tp} props, {len(games)} games")
    log_event(f'B{BATCH}','PROPS_FETCHED',detail=f'{tp} props')
    append_to_excel(games,date_str)
    spreads=[{'Date':date_str,'Game':f"{g['away']} @ {g['home']}",
        'Home':g['home'],'Away':g['away'],'Spread':g['spread'],'Total':g['total'],'Commence':g['ts']}
        for g in games.values() if g['spread'] is not None]
    return games,spreads

def run_predictions(games,date_str):
    print(f"\n  Running V12.0 predictions...")
    combined=load_combined(FILE_GL_2425,FILE_GL_2526)
    h2h=pd.read_csv(FILE_H2H)
    h2h_dedup=h2h.drop_duplicates(subset=['PLAYER_NAME','OPPONENT'],keep='last')
    h2h_lkp={(r['PLAYER_NAME'],r['OPPONENT']):r.to_dict() for _,r in h2h_dedup.iterrows()}
    pidx=build_player_index(combined); name_map=build_name_map(pidx)
    played=filter_played(combined)
    team_fga=played.groupby('OPPONENT')['FGA'].mean()
    pace_rank={t:i+1 for i,(t,_) in enumerate(team_fga.sort_values(ascending=False).items())}
    print("    Building caches...")
    b2b_delta_cache=build_b2b_delta(played)
    dyn_dvp_cache=build_dynamic_dvp(played)
    # Opp defensive trend per position
    _PG={'Guard':['PG','SG','G','G-F','F-G','Guard'],'Forward':['SF','PF','F','F-C','C-F','Forward'],'Center':['C','Center']}
    def _pg(r):
        for g,vs in _PG.items():
            if str(r) in vs: return g
        return 'Forward'
    played['_pg']=played['PLAYER_POSITION'].map(_pg)
    opp_def_trend_cache={}; opp_def_var_cache={}
    for pos in ['Guard','Forward','Center']:
        pm=played[played['_pg']==pos]
        for team in pm['OPPONENT'].unique():
            pts=pm[pm['OPPONENT']==team]['PTS']
            if len(pts)>=5:
                l5=pts.tail(5).mean(); l20=pts.tail(20).mean() if len(pts)>=20 else pts.mean()
                opp_def_trend_cache[(team,pos)]=round(float(l5-l20),2)
                opp_def_var_cache[(team,pos)]=round(float(pts.tail(10).std() if len(pts)>=10 else pts.std()),2)
    played.drop(columns=['_pg'],inplace=True,errors='ignore')
    # Line bias per player
    ps=played.sort_values(['PLAYER_NAME','GAME_DATE'])
    l30r=ps.groupby('PLAYER_NAME')['PTS'].transform(lambda s:s.rolling(30,min_periods=1).mean().shift(1))
    errr=ps['PTS']-(l30r*2).round()/2
    bias=ps.groupby('PLAYER_NAME').apply(lambda g:errr.reindex(g.index).rolling(10,min_periods=3).mean().shift(1)).explode().astype(float).reset_index(level=0,drop=True).reindex(ps.index)
    line_bias_cache={pn:round(float(bias.reindex(grp.index).iloc[-1]) if pd.notna(bias.reindex(grp.index).iloc[-1]) else 0.0,2) for pn,grp in ps.groupby('PLAYER_NAME')}
    b2b_map={}
    for pn,g in played.sort_values('GAME_DATE').groupby('PLAYER_NAME'):
        ds=g['GAME_DATE'].values
        for i in range(len(ds)):
            k=(pn,pd.Timestamp(ds[i]).strftime('%Y-%m-%d'))
            b2b_map[k]=int((ds[i]-ds[i-1]).astype('timedelta64[D]').astype(int)) if i>0 else 99
    # Load models
    global_model=None
    if FILE_MODEL.exists():
        with open(FILE_MODEL,'rb') as f: global_model=pickle.load(f)
    trust={}
    if FILE_TRUST.exists():
        with open(FILE_TRUST) as f: trust=json.load(f)
    seg_model=None
    if FILE_SEG_MODELS.exists():
        from segment_model import SegmentModel
        seg_model=SegmentModel.load(FILE_SEG_MODELS)
    direction_clf=None; direction_cal=None
    if FILE_DIR_CLF.exists():
        with open(FILE_DIR_CLF,'rb') as f: direction_clf=pickle.load(f)
        print("    ✓ V11 direction classifier")
    if FILE_CALIBRATOR.exists():
        with open(FILE_CALIBRATOR,'rb') as f: direction_cal=pickle.load(f)
    q_models={}
    if FILE_Q_MODELS.exists():
        with open(FILE_Q_MODELS,'rb') as f: q_models=pickle.load(f)
    existing=[]
    if TODAY_JSON.exists():
        with open(TODAY_JSON) as f: existing=json.load(f)
    exist_map={(p['player'],p.get('match','')):p for p in existing if p.get('date')==date_str}
    from model_trainer import FEATURES
    batch_ts=now_uk().strftime('%H:%M')
    plays=[]; skipped={'low_line':0,'no_player':0,'few_games':0,'no_features':0}
    for eid,g in games.items():
        ht=g['home']; at=g['away']; ms=f"{at} @ {ht}"
        fms=f"{TEAM_FULL.get(at,at)} @ {TEAM_FULL.get(ht,ht)}"
        sv=g['spread']; tv=g['total']; blow=abs(sv)>=10 if sv else False
        for pname_raw,pd_ in g['props'].items():
            line=pd_.get('line')
            if not line or line<3: skipped['low_line']+=1; continue
            pname=resolve_name(pname_raw,pidx,name_map)
            if pname is None: skipped['no_player']+=1; continue
            prior=get_prior_games_played(pidx,pname,date_str)
            if len(prior)<5: skipped['few_games']+=1; continue
            sn=prior.iloc[-1]; ta=str(sn.get('GAME_TEAM_ABBREVIATION',''))
            ih=ta==ht; opp=at if ih else ht
            pos=POS_MAP.get(str(sn.get('PLAYER_POSITION','')),'Forward')
            feats=extract_prediction_features(prior,line,b2b_delta=b2b_delta_cache,dyn_dvp=dyn_dvp_cache,
                opp=opp,pos=pos,player_name=pname,game_date=date_str)
            if feats is None: skipped['no_features']+=1; continue
            L30=feats['L30']; L20=feats['L20']; L10=feats['L10']
            L5=feats['L5']; L3=feats['L3']
            vol=feats['vol']; trend=feats['trend']; std10=feats['std10']
            hr10=feats['hr10']; hr30=feats['hr30']
            r20=feats['recent20']; r20h=feats['r20_homes']
            fg30=feats['fg30']; fg10=feats['fg10']; fgTrend=feats['fgTrend']
            m30=feats['m30']; m10=feats['m10']; minTrend=feats['minTrend']
            fga30=feats['fga30']; fga10=feats['fga10']
            min_cv=feats['min_cv']; ppm=feats['ppm']; rmt=feats['rmt']; fpm=feats['fpm']
            l10_ewm=feats.get('l10_ewm',L10); l5_ewm=feats.get('l5_ewm',L5)
            usage_l10=feats.get('usage_l10',0.0); usage_l30=feats.get('usage_l30',0.0)
            fg3a_l10=feats.get('fg3a_l10',0.0); fg3m_l10=feats.get('fg3m_l10',0.0)
            fta_l10=feats.get('fta_l10',0.0); ft_rate_l10=feats.get('ft_rate_l10',0.0)
            home_l10=feats.get('home_l10',L10); away_l10=feats.get('away_l10',L10)
            home_away_split=feats.get('home_away_split',0.0)
            b2b_pts_delta=feats.get('b2b_pts_delta',0.0)
            usage_segment=feats.get('usage_segment',0); line_bucket=feats.get('line_bucket',0)
            pg_key=(opp,pos)
            opp_def_trend=opp_def_trend_cache.get(pg_key,0.0)
            opp_def_var=opp_def_var_cache.get(pg_key,5.0)
            rest_days_v=b2b_map.get((pname,date_str),99)
            if rest_days_v<=1: rest_cat=0
            elif rest_days_v==2: rest_cat=1
            elif rest_days_v==3: rest_cat=2
            elif rest_days_v<=5: rest_cat=3
            else: rest_cat=4
            is_long_rest=1 if rest_days_v>=6 else 0
            ib2b=1 if rest_days_v==1 else 0
            line_vs_l30=round(line-L30,2)
            line_bias_l10=line_bias_cache.get(pname,0.0)
            fga10_safe=fga10 or 8.0
            ppfga_l10=round(L10/max(fga10_safe,1.0),3)
            role_intensity=round(usage_l10*(m10 or 28)/100,2)
            season_game_n=len(prior)
            hr_=h2h_lkp.get((pname,opp))
            hG=int(hr_['H2H_GAMES']) if hr_ else 0
            hA=float(hr_['H2H_AVG_PTS']) if hr_ else None
            hTS=float(hr_['H2H_TS_VS_OVERALL']) if hr_ and pd.notna(hr_.get('H2H_TS_VS_OVERALL')) else 0
            hFA=float(hr_['H2H_FGA_VS_OVERALL']) if hr_ and pd.notna(hr_.get('H2H_FGA_VS_OVERALL')) else 0
            hMN=float(hr_['H2H_MIN_VS_OVERALL']) if hr_ and pd.notna(hr_.get('H2H_MIN_VS_OVERALL')) else 0
            hCF=float(hr_['H2H_CONFIDENCE']) if hr_ and pd.notna(hr_.get('H2H_CONFIDENCE')) else 0
            hStr=f"{hA:.1f} ({hG}g)" if hG>=3 and hA else ''
            uh=hG>=3 and hA is not None
            dP=get_dvp(opp,pos); dO=get_def_overall(opp); op=pace_rank.get(opp,15)
            fd={'level':feats.get('level',L30),'reversion':feats.get('reversion',round(L10-L30,2)),
                'momentum':feats.get('momentum',round(L5-L30,2)),'acceleration':feats.get('acceleration',round(L3-L5,2)),
                'level_ewm':l10_ewm,'volatility':std10,
                'fg3a_l10':fg3a_l10,'fg3m_l10':fg3m_l10,'fta_l10':fta_l10,
                'ft_rate_l10':ft_rate_l10,'ppfga_l10':ppfga_l10,
                'usage_l10':usage_l10,'usage_l30':usage_l30,'role_intensity':role_intensity,
                'min_l10':m10 or 28.0,'min_l3':feats.get('min_l3',m10 or 28.0),
                'min_cv':min_cv,'recent_min_trend':rmt,
                'home_l10':home_l10,'away_l10':away_l10,'home_away_split':home_away_split,
                'is_b2b':ib2b,'b2b_pts_delta':b2b_pts_delta,'rest_cat':rest_cat,'is_long_rest':is_long_rest,
                'opp_def_trend':opp_def_trend,'opp_def_var':opp_def_var,
                'defP':dP,'defP_dynamic':feats.get('defP_dynamic',dP),'pace_rank':op,
                'h2h_ts_dev':hTS,'h2h_fga_dev':hFA,'h2h_min_dev':hMN,'h2h_conf':hCF,
                'line':line,'line_bucket':line_bucket,'line_vs_l30':line_vs_l30,'line_bias_l10':line_bias_l10,
                'usage_segment':usage_segment,'season_game_num':season_game_n,
                'l30':L30,'l10':L10,'l5':L5,'l3':L3,'l10_ewm':l10_ewm,'l5_ewm':l5_ewm,
                'volume':vol,'trend':trend,'std10':std10,'consistency':1/(std10+1),
                'pts_per_min':ppm,'fga_per_min':fpm,'fga_l10':fga10_safe,
                'is_long_rest':is_long_rest,
            }
            Xp=pd.DataFrame([fd])[FEATURES].fillna(0); ua=np.array([usage_l10])
            pp=None; pg=0; pred_q25=None; pred_q75=None; prob_over=None
            if global_model:
                pp=float(seg_model.predict(Xp,ua)[0]) if seg_model else float(global_model.predict(Xp)[0])
                pg=abs(pp-line)
                if q_models:
                    pred_q25=float(q_models['q25'].predict(Xp)[0])
                    pred_q75=float(q_models['q75'].predict(Xp)[0])
            if direction_clf is not None:
                raw_p=float(direction_clf.predict_proba(Xp)[0,1])
                # Calibrated prob → confidence display (accurate probability)
                prob_over=float(direction_cal.predict([raw_p])[0]) if direction_cal else raw_p
                # Raw prob → conviction ranking (has genuine spread — calibration creates plateaus)
                raw_conviction = abs(raw_p - 0.5)
            W=POS_WEIGHTS.get(pos,POS_WEIGHTS['Forward'])
            S={1:np.clip((L30-line)/5,-1,1),2:(hr30/100-0.5)*2,3:(hr10/100-0.5)*2,
               4:np.clip((L5-L30)/5,-1,1),5:np.clip(vol/5,-1,1),6:np.clip((dP-15)/15,-1,1),
               7:np.clip((hA-line)/5,-1,1) if uh else 0.0,8:np.clip((15-op)/15,-1,1),
               9:np.clip((fgTrend or 0)/10,-1,1),10:np.clip((minTrend or 0)/5,-1,1)}
            tw_w=sum(W.values()) if uh else sum(v for k,v in W.items() if k!=7)
            ws_w=sum(W[k]*S[k] for k in S) if uh else sum(W[k]*S[k] for k in S if k!=7)
            comp=ws_w/tw_w if tw_w else 0
            if prob_over is not None:
                if prob_over>=0.53:   dr='OVER';       is_lean=False
                elif prob_over<=0.47: dr='UNDER';      is_lean=False
                elif prob_over>=0.50: dr='LEAN OVER';  is_lean=True
                else:                 dr='LEAN UNDER'; is_lean=True
                # conf = calibrated P(correct direction) — for display
                conf=prob_over if 'OVER' in dr else (1-prob_over)
                conf=float(np.clip(conf,0.45,0.90))
                # conviction = raw classifier distance from 0.5 — for tier ranking
                # Raw prob has genuine spread; isotonic calibrator creates plateaus
                conviction = raw_conviction if 'raw_conviction' in dir() else abs(prob_over-0.5)
            else:
                if (pp and pp>line+0.3) or (not pp and comp>0.05): dr='OVER'; is_lean=False
                elif (pp and pp<line-0.3) or (not pp and comp<-0.05): dr='UNDER'; is_lean=False
                else: dr=f"LEAN {'OVER' if comp>=0 else 'UNDER'}"; is_lean=True
                conf=float(np.clip(0.5+abs(comp)*0.3,0.50,0.85)); conviction=pg/20
            if 'OVER' in dr and line>=25: conf=float(np.clip(conf-0.03,0.45,0.90))
            io=('UNDER' not in dr); fl=0; fds=[]
            for nm,ag,dt in [
                ('Volume',(io and vol>0) or(not io and vol<0),f"{vol:+.1f}"),
                ('HR L30',(io and hr30>50) or(not io and hr30<50),f"{hr30}%"),
                ('HR L10',(io and hr10>50) or(not io and hr10<50),f"{hr10}%"),
                ('Trend',(io and trend>0) or(not io and trend<0),f"{trend:+.1f}"),
                ('Context',(io and vol>-1) or(not io and vol<1),f"vol={vol:+.1f}"),
                ('Defense',(io and dP>15) or(not io and dP<15),f"#{dP}"),
                ('H2H',uh and((io and hA>line) or(not io and hA<line)),f"{hA:.1f}" if uh else "N/A"),
                ('Pace',(io and op<15) or(not io and op>15),f"#{op}"),
                ('FG Trend',fgTrend is not None and((io and fgTrend>0) or(not io and fgTrend<0)),f"{fgTrend:+.1f}%" if fgTrend else "N/A"),
                ('Min Trend',minTrend is not None and((io and minTrend>0) or(not io and minTrend<0)),f"{minTrend:+.1f}" if minTrend else "N/A"),
            ]:
                fl+=1 if ag else 0; fds.append({'name':nm,'agrees':bool(ag),'detail':dt})
            ha=True
            if hTS!=0:
                if 'OVER' in dr and hTS<-3: ha=False
                elif 'UNDER' in dr and hTS>3: ha=False
            q25_gate=True
            if q_models and pred_q25 is not None:
                if 'OVER' in dr: q25_gate=pred_q25>line
                elif 'UNDER' in dr: q25_gate=pred_q75 is not None and pred_q75<line
            # V12 tier thresholds — calibrated to actual OOF results:
            #   conviction >= 0.23  → top ~5%  of plays → 86.3% OOF accuracy  (T1_ULTRA)
            #   conviction >= 0.17  → top ~10% of plays → 79.7% OOF accuracy  (T1_PREMIUM)
            #   conviction >= 0.11  → top ~15% of plays → 72.9% OOF accuracy  (T1)
            #   conviction >= 0.06  → top ~30% of plays → 72.9% OOF accuracy  (T2)
            # conviction = |raw_prob - 0.5|, NOT calibrated (calibrated creates plateaus)
            if is_lean:              tier=3; tl='T3_LEAN'
            elif conviction>=0.23 and fl>=7 and std10<=7 and ha and q25_gate: tier=1; tl='T1_ULTRA'
            elif conviction>=0.17 and fl>=6 and std10<=7 and ha and q25_gate: tier=1; tl='T1_PREMIUM'
            elif conviction>=0.11 and fl>=6 and std10<=8 and ha: tier=1; tl='T1'
            elif conviction>=0.06 and fl>=5 and std10<=9 and ha: tier=2; tl='T2'
            else:                    tier=3; tl='T3'
            tr=trust.get(pname)
            if tr is not None and tr<0.42 and tier==1: tier=2; tl='T2'
            units=3.0 if tl=='T1_ULTRA' else 2.0 if tier==1 else 1.0 if tier==2 else 0.0
            oo=american_to_decimal(pd_.get('over')); uo=american_to_decimal(pd_.get('under'))
            ro=sum(1 for r in r20 if r>line); ru=sum(1 for r in r20 if r<=line)
            lh=[{'line':line,'batch':BATCH,'ts':batch_ts}]
            ekey=(pname,ms)
            if ekey in exist_map:
                ep=exist_map[ekey]; old_lh=ep.get('lineHistory',[])
                if isinstance(old_lh,list) and old_lh:
                    lh=old_lh
                    if not any(isinstance(h,dict) and h.get('batch')==BATCH for h in lh):
                        lh.append({'line':line,'batch':BATCH,'ts':batch_ts})
                    else:
                        for h in lh:
                            if isinstance(h,dict) and h.get('batch')==BATCH:
                                h['line']=line; h['ts']=batch_ts
            play_data={'player':pname,'dir':dr,'line':line,'l30':L30,'l10':L10,'l5':L5,'l3':L3,
                'volume':vol,'trend':trend,'std10':std10,'flags':fl,'flagDetails':fds,
                'h2h':hStr,'h2hG':hG,'h2hTsDev':hTS,'h2hFgaDev':hFA,
                'h2hProfile':hr_.get('H2H_SCORING_PROFILE','') if hr_ else '',
                'defP':dP,'defO':dO,'pace':op,'fgTrend':fgTrend,'minTrend':minTrend,
                'minL30':m30,'minL10':m10,'conf':conf,
                'predPts':round(pp,1) if pp else None,'predGap':round(pg,1) if pp else None,
                'predQ25':round(pred_q25,1) if pred_q25 is not None else None,
                'predQ75':round(pred_q75,1) if pred_q75 is not None else None,
                'tierLabel':tl,'position':pos,'match':ms,'isHome':ih,
                'recent':r20[:5],'hr30':hr30,'hr10':hr10,
                'usage_l10':usage_l10,'fg3a_l10':fg3a_l10,
                'home_l10':home_l10,'away_l10':away_l10,'home_away_split':home_away_split,
                'b2b_pts_delta':b2b_pts_delta,'usage_segment':usage_segment,'is_b2b':ib2b,
                'prob_over':round(prob_over,4) if prob_over is not None else None,
            }
            reason=generate_pre_match_reason(play_data)
            plays.append({
                'date':date_str,'player':pname,'match':ms,'fullMatch':fms,
                'isHome':ih,'team':ta,'gameTime':g['gt'],'position':pos,'posSimple':pos[:1],
                'line':line,'overOdds':oo,'underOdds':uo,'books':pd_.get('books',1),
                'spread':sv,'total':tv,'blowout':blow,
                'l30':round(L30,1),'l20':round(L20,1),'l10':round(L10,1),
                'l5':round(L5,1),'l3':round(L3,1),'hr30':hr30,'hr10':hr10,
                'recent':r20[:5],'recent10':r20[:10],'recent20':r20,
                'recent20homes':[bool(x) for x in r20h],
                'defO':dO,'defP':dP,'pace':op,'h2h':hStr,'h2hG':hG,
                'h2hTsDev':hTS,'h2hFgaDev':hFA,'h2hConfidence':hCF,
                'h2hProfile':hr_.get('H2H_SCORING_PROFILE','') if hr_ else '',
                'fgL30':fg30,'fgL10':fg10,'fga30':fga30,'fga10':fga10,
                'minL30':m30,'minL10':m10,'std10':round(std10,1),
                'dir':dr,'rawDir':dr,'conf':round(conf,3),'tier':tier,'tierLabel':tl,
                'units':units,'avail':'OK','volume':vol,'trend':trend,
                'fgTrend':fgTrend,'minTrend':minTrend,'flags':fl,'flagsStr':f"{fl}/10",
                'flagDetails':fds,'recentOver':ro,'recentUnder':ru,'lineHistory':lh,
                'predPts':round(pp,1) if pp else None,'predGap':round(pg,1) if pp else None,
                'predQ25':round(pred_q25,1) if pred_q25 is not None else None,
                'predQ75':round(pred_q75,1) if pred_q75 is not None else None,
                'probOver':round(prob_over,4) if prob_over is not None else None,
                'conviction':round(conviction,4),
                'l10_ewm':l10_ewm,'l5_ewm':l5_ewm,'usage_l10':usage_l10,'usage_l30':usage_l30,
                'fg3a_l10':fg3a_l10,'fg3m_l10':fg3m_l10,'fta_l10':fta_l10,'ft_rate_l10':ft_rate_l10,
                'home_l10':home_l10,'away_l10':away_l10,'home_away_split':home_away_split,
                'b2b_pts_delta':b2b_pts_delta,'usage_segment':usage_segment,
                'opp_def_trend':opp_def_trend,'line_vs_l30':line_vs_l30,
                'preMatchReason':reason,'actualPts':None,'result':None,'delta':None,
                'postMatchReason':'','lossType':None,'reason':'','season':'2025-26',
            })
    ts=sum(skipped.values()); leans=sum(1 for p in plays if 'LEAN' in p.get('dir',''))
    print(f"  {len(plays)} predictions ({len(plays)-leans} conviction + {leans} leans, {ts} skipped)")
    if ts: print(f"    Skips: {', '.join(f'{v} {k}' for k,v in skipped.items() if v)}")
    log_event(f'B{BATCH}','PREDICTIONS',detail=f'{len(plays)} plays')
    return plays

def save_today(plays,date_str):
    batch_ts=now_uk().strftime('%H:%M')
    existing=[]
    if TODAY_JSON.exists():
        with open(TODAY_JSON) as f: existing=json.load(f)
    before=len(existing)
    today_ex=[p for p in existing if p.get('date')==date_str]
    historical=[p for p in existing if p.get('date')!=date_str]
    ex_map={(p['player'],p.get('match','')):p for p in today_ex}
    new_map={(p['player'],p['match']):p for p in plays}
    merged=[]; added=updated=preserved=0
    for key in set(ex_map)|set(new_map):
        old=ex_map.get(key); new=new_map.get(key)
        if old and old.get('result') in ('WIN','LOSS','DNP'): merged.append(old); continue
        if old and new:
            old_lh=old.get('lineHistory',[])
            if isinstance(old_lh,list) and old_lh:
                new['lineHistory']=old_lh
                if not any(isinstance(h,dict) and h.get('batch')==BATCH for h in old_lh):
                    new['lineHistory'].append({'line':new['line'],'batch':BATCH,'ts':batch_ts})
                else:
                    for h in new['lineHistory']:
                        if isinstance(h,dict) and h.get('batch')==BATCH: h['line']=new['line']; h['ts']=batch_ts
            merged.append(new); updated+=1
        elif old and not new: merged.append(old); preserved+=1
        elif new and not old: merged.append(new); added+=1
    merged.sort(key=lambda p:(p.get('tier',9),-p.get('conviction',0)))
    all_p=merged+sorted(historical,key=lambda p:p.get('date',''),reverse=True)
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    with open(TODAY_JSON,'w') as f: json.dump(clean_json(all_p),f)
    t1=sum(1 for p in merged if p.get('tier')==1); t2=sum(1 for p in merged if p.get('tier')==2)
    print(f"\n  ✓ today.json: {len(merged)} plays ({t1} T1, {t2} T2) | Added:{added} Updated:{updated} Preserved:{preserved}")
    log_batch_summary(f'B{BATCH}',props_fetched=len(plays),plays_added=added)
    verify_no_deletion(f'B{BATCH}',TODAY_JSON,before,len(all_p),'SAVE_TODAY')

def main():
    date_str=today_et()
    if len(sys.argv)>2 and '-' in sys.argv[2]: date_str=sys.argv[2]
    print("="*60); print(f"PropEdge V12.0 — BATCH {BATCH}: PREDICT")
    print(f"  Date: {date_str} | {now_uk().strftime('%Y-%m-%d %H:%M %Z')}"); print("="*60)
    log_event(f'B{BATCH}','BATCH_START',detail=date_str)
    games,_=fetch_props(date_str)
    if not games: print("  No games today."); return
    plays=run_predictions(games,date_str)
    save_today(plays,date_str)
    repo=REPO_DIR if REPO_DIR.exists() else ROOT
    from batch0_grade import git_push
    git_push(repo,f"B{BATCH}: {date_str} — {len(plays)} plays")
    log_event(f'B{BATCH}','BATCH_COMPLETE')
    try:
        import subprocess
        subprocess.run(['osascript','-e',f'display notification "B{BATCH}: {len(plays)} plays" with title "PropEdge V12.0"'],capture_output=True,timeout=5)
    except: pass

if __name__=='__main__': main()
