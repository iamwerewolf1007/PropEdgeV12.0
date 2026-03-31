"""
PropEdge V12.0 — Configuration & Shared Constants
===================================================
Completely independent from V9.2.
Repo:       git@github.com:iamwerewolf1007/PropEdgeV10.0.git
Working dir: ~/Documents/GitHub/PropEdgeV10.0

Timezone design:
  All game dates are stored and compared in US Eastern Time (ET).
  NBA schedules are published in ET. All batch date logic uses ET.
  Logging uses UK time (BST/GMT) for the user's convenience.

  US DST  (EDT UTC-4): 2nd Sunday of March → 1st Sunday of November
  UK DST  (BST UTC+1): Last Sunday of March → Last Sunday of October

  These two zones diverge:
    Mar 8–28: US on EDT (-4), UK still on GMT (UTC+0)  ← the gap
    Oct 25 – Nov 1: UK back to GMT, US still on EDT (-4)
  The old single _is_dst() function returned wrong UK offsets during these gaps.
  This version uses separate algorithmic functions for each zone.
"""
import calendar
from pathlib import Path
from datetime import timezone, timedelta, datetime

VERSION = 'V12.0'

# ─── PATHS ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.resolve()
SOURCE_DIR = ROOT / 'source-files'
DATA_DIR   = ROOT / 'data'
MODEL_DIR  = ROOT / 'models'
LOG_DIR    = ROOT / 'logs'
DAILY_DIR  = ROOT / 'daily'
MASTER_DIR = ROOT / 'master'

# Source data files (place in source-files/ before first run)
FILE_GL_2425 = SOURCE_DIR / 'nba_gamelogs_2024_25.csv'
FILE_GL_2526 = SOURCE_DIR / 'nba_gamelogs_2025_26.csv'
FILE_H2H     = SOURCE_DIR / 'h2h_database.csv'
FILE_PROPS   = SOURCE_DIR / 'PropEdge_-_Match_and_Player_Prop_lines_.xlsx'

# Model files
FILE_MODEL      = MODEL_DIR / 'projection_model.pkl'
FILE_TRUST      = MODEL_DIR / 'player_trust.json'
FILE_SEG_MODELS = MODEL_DIR / 'segment_model.pkl'
FILE_Q_MODELS   = MODEL_DIR / 'quantile_models.pkl'
FILE_CALIBRATOR = MODEL_DIR / 'calibrator.pkl'
FILE_DIR_CLF    = MODEL_DIR / 'direction_classifier.pkl'  # V11: LightGBM binary classifier

# Live data files
TODAY_JSON  = DATA_DIR / 'today.json'
SEASON_2425 = DATA_DIR / 'season_2024_25.json'
SEASON_2526 = DATA_DIR / 'season_2025_26.json'
AUDIT_LOG   = DATA_DIR / 'audit_log.csv'

# ─── REPO / GIT ───────────────────────────────────────────────────────────────
REPO_DIR   = Path.home() / 'Documents' / 'GitHub' / 'PropEdgeV12.0'
GIT_REMOTE = 'git@github.com:iamwerewolf1007/PropEdgeV12.0.git'

# ─── API ──────────────────────────────────────────────────────────────────────
ODDS_API_KEY  = 'c0bab20a574208a41a6e0d930cdaf313'
ODDS_API_BASE = 'https://api.the-odds-api.com/v4'
SPORT         = 'basketball_nba'
CREDIT_ALERT  = 170

# ─── DST HELPERS ──────────────────────────────────────────────────────────────

def _nth_weekday(year, month, weekday, n):
    """
    Return the day-of-month of the nth occurrence of weekday in month/year.
    weekday: 0=Monday … 6=Sunday.  n=1 → first occurrence.
    """
    first_wd = datetime(year, month, 1).weekday()
    offset   = (weekday - first_wd) % 7
    return 1 + offset + (n - 1) * 7

def _last_weekday(year, month, weekday):
    """Return the day-of-month of the last occurrence of weekday in month/year."""
    last_day    = calendar.monthrange(year, month)[1]
    last_day_wd = datetime(year, month, last_day).weekday()
    offset      = (last_day_wd - weekday) % 7
    return last_day - offset

def _us_is_dst(now_utc=None):
    """
    True when US Eastern is on EDT (UTC-4).
    EDT: 2nd Sunday of March 02:00 EST (07:00 UTC) →
         1st Sunday of November 02:00 EDT (06:00 UTC)
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    y         = now_utc.year
    dst_start = datetime(y, 3,  _nth_weekday(y, 3,  6, 2), 7,  0, tzinfo=timezone.utc)
    dst_end   = datetime(y, 11, _nth_weekday(y, 11, 6, 1), 6,  0, tzinfo=timezone.utc)
    return dst_start <= now_utc < dst_end

def _uk_is_dst(now_utc=None):
    """
    True when UK is on BST (UTC+1).
    BST: Last Sunday of March 01:00 UTC → Last Sunday of October 01:00 UTC
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    y         = now_utc.year
    dst_start = datetime(y, 3,  _last_weekday(y, 3,  6), 1, 0, tzinfo=timezone.utc)
    dst_end   = datetime(y, 10, _last_weekday(y, 10, 6), 1, 0, tzinfo=timezone.utc)
    return dst_start <= now_utc < dst_end

# ─── TIMEZONE ACCESSORS ───────────────────────────────────────────────────────

def _offset(hours):
    return timezone(timedelta(hours=hours))

def get_et(now_utc=None):
    """Return the current US Eastern timezone offset (EDT UTC-4 or EST UTC-5)."""
    return _offset(-4 if _us_is_dst(now_utc) else -5)

def get_uk(now_utc=None):
    """Return the current UK timezone offset (BST UTC+1 or GMT UTC+0)."""
    return _offset(1 if _uk_is_dst(now_utc) else 0)

def today_et():
    """Today's date string in US Eastern Time — used for all game date logic."""
    return datetime.now(get_et()).strftime('%Y-%m-%d')

def now_uk():
    """Current datetime in UK time — used for logging only."""
    return datetime.now(get_uk())

def now_utc():
    return datetime.now(timezone.utc)

def et_tz_for_date(date_str):
    """
    Return the correct ET timezone offset for a specific ET date string.
    Safe to use even when called in a different UTC hour than the date.
    Uses noon UTC on that date to determine the offset — safe for any NBA game day.
    """
    y, m, d = map(int, date_str.split('-'))
    noon_utc = datetime(y, m, d, 12, 0, tzinfo=timezone.utc)
    return get_et(noon_utc)

def et_window(date_str):
    """
    Return (fr_utc_str, to_utc_str) covering the full ET day for date_str.

    Uses midnight–midnight ET, ensuring ALL NBA games on that ET date
    (which can start from ~5pm ET and run to ~1am ET next day) are included.

    fr = midnight ET → UTC
    to = midnight ET next day + 1 hour buffer → UTC
         (buffer catches any game start times right at midnight ET)
    """
    et = et_tz_for_date(date_str)
    y, m, d = map(int, date_str.split('-'))
    d_start = datetime(y, m, d,  0,  0,  0, tzinfo=et)
    d_end   = datetime(y, m, d, 23, 59, 59, tzinfo=et)
    # Add 90-minute buffer past midnight ET to catch any late game starts
    fr = d_start.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    to = (d_end + timedelta(minutes=90)).astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    return fr, to

def tz_info_str(date_str):
    """Return a human-readable string describing timezone state for date_str."""
    et  = et_tz_for_date(date_str)
    now = datetime.now(timezone.utc)
    uk  = get_uk(now)
    et_name = 'EDT (UTC-4)' if _us_is_dst(now) else 'EST (UTC-5)'
    uk_name = 'BST (UTC+1)' if _uk_is_dst(now) else 'GMT (UTC+0)'
    fr, to  = et_window(date_str)
    return (f"ET={et_name}  UK={uk_name} | "
            f"Fetch window UTC: {fr} → {to}")

# ─── TEAM MAPS ────────────────────────────────────────────────────────────────
TEAM_ABR = {
    'Atlanta Hawks':'ATL','Boston Celtics':'BOS','Brooklyn Nets':'BKN',
    'Charlotte Hornets':'CHA','Chicago Bulls':'CHI','Cleveland Cavaliers':'CLE',
    'Dallas Mavericks':'DAL','Denver Nuggets':'DEN','Detroit Pistons':'DET',
    'Golden State Warriors':'GSW','Houston Rockets':'HOU','Indiana Pacers':'IND',
    'LA Clippers':'LAC','Los Angeles Clippers':'LAC','Los Angeles Lakers':'LAL',
    'Memphis Grizzlies':'MEM','Miami Heat':'MIA','Milwaukee Bucks':'MIL',
    'Minnesota Timberwolves':'MIN','New Orleans Pelicans':'NOP',
    'New York Knicks':'NYK','Oklahoma City Thunder':'OKC','Orlando Magic':'ORL',
    'Philadelphia 76ers':'PHI','Phoenix Suns':'PHX','Portland Trail Blazers':'POR',
    'Sacramento Kings':'SAC','San Antonio Spurs':'SAS','Toronto Raptors':'TOR',
    'Utah Jazz':'UTA','Washington Wizards':'WAS',
}
TEAM_FULL = {v: k for k, v in TEAM_ABR.items()}

def resolve_abr(full_name):
    return TEAM_ABR.get(full_name, full_name[:3].upper())

# ─── DVP RANKINGS ─────────────────────────────────────────────────────────────
DVP_RAW = {
    'BOS':{'PG':4,'SG':1,'SF':1,'PF':3,'C':1},
    'DET':{'PG':1,'SG':7,'SF':10,'PF':17,'C':5},
    'GSW':{'PG':16,'SG':16,'SF':16,'PF':11,'C':21},
    'ATL':{'PG':12,'SG':27,'SF':27,'PF':22,'C':17},
    'HOU':{'PG':7,'SG':3,'SF':8,'PF':1,'C':7},
    'BKN':{'PG':9,'SG':26,'SF':17,'PF':23,'C':19},
    'MEM':{'PG':24,'SG':20,'SF':21,'PF':25,'C':23},
    'LAC':{'PG':19,'SG':21,'SF':6,'PF':2,'C':12},
    'DAL':{'PG':13,'SG':29,'SF':12,'PF':26,'C':27},
    'CLE':{'PG':15,'SG':10,'SF':15,'PF':16,'C':20},
    'CHA':{'PG':5,'SG':13,'SF':2,'PF':14,'C':4},
    'DEN':{'PG':14,'SG':8,'SF':20,'PF':19,'C':10},
    'IND':{'PG':29,'SG':14,'SF':28,'PF':13,'C':24},
    'LAL':{'PG':11,'SG':15,'SF':22,'PF':8,'C':6},
    'MIA':{'PG':27,'SG':18,'SF':19,'PF':27,'C':14},
    'CHI':{'PG':20,'SG':17,'SF':29,'PF':28,'C':26},
    'NOP':{'PG':21,'SG':25,'SF':23,'PF':20,'C':25},
    'UTA':{'PG':30,'SG':30,'SF':24,'PF':30,'C':22},
    'SAC':{'PG':22,'SG':28,'SF':14,'PF':18,'C':29},
    'POR':{'PG':18,'SG':22,'SF':26,'PF':15,'C':28},
    'WAS':{'PG':26,'SG':23,'SF':30,'PF':29,'C':30},
    'OKC':{'PG':2,'SG':11,'SF':13,'PF':4,'C':8},
    'NYK':{'PG':3,'SG':6,'SF':9,'PF':7,'C':2},
    'PHI':{'PG':8,'SG':24,'SF':18,'PF':24,'C':15},
    'PHX':{'PG':6,'SG':2,'SF':7,'PF':9,'C':16},
    'MIN':{'PG':25,'SG':4,'SF':4,'PF':10,'C':13},
    'ORL':{'PG':23,'SG':12,'SF':3,'PF':12,'C':11},
    'TOR':{'PG':17,'SG':9,'SF':5,'PF':6,'C':9},
    'SAS':{'PG':10,'SG':5,'SF':11,'PF':5,'C':18},
    'MIL':{'PG':28,'SG':19,'SF':25,'PF':21,'C':3},
}

POS_MAP = {
    'G':'Guard','F':'Forward','C':'Center','G-F':'Guard','F-G':'Guard',
    'PG':'Guard','SG':'Guard','F-C':'Forward','C-F':'Center','SF':'Forward',
    'PF':'Forward','Guard':'Guard','Forward':'Forward','Center':'Center',
}

def get_dvp(team, pos, fallback=15):
    if team not in DVP_RAW: return fallback
    d   = DVP_RAW[team]
    pos = POS_MAP.get(str(pos), str(pos))
    if pos == 'Guard':   return round((d['PG'] + d['SG']) / 2)
    elif pos == 'Center': return d['C']
    else:                 return round((d['SF'] + d['PF']) / 2)

def get_def_overall(team, fallback=15):
    if team not in DVP_RAW: return fallback
    return round(sum(DVP_RAW[team].values()) / 5)

# ─── POSITION WEIGHTS (10-signal Engine A) ────────────────────────────────────
POS_WEIGHTS = {
    'Guard':   {1:3.0,2:2.5,3:2.0,4:2.0,5:1.0,6:1.5,7:1.2,8:0.5,9:1.5,10:1.0},
    'Forward': {1:3.0,2:2.5,3:2.0,4:1.5,5:1.5,6:1.5,7:1.0,8:0.5,9:1.0,10:0.75},
    'Center':  {1:2.5,2:2.0,3:2.0,4:1.0,5:1.5,6:2.5,7:1.0,8:1.0,9:0.5,10:1.5},
}

# ─── ROLLING WINDOWS & COLUMNS ────────────────────────────────────────────────
WINDOWS   = [3,5,10,20,30,50,100,200]
ROLL_COLS = [
    'MIN_NUM','FGM','FGA','FG_PCT','FG3M','FG3A','FG3_PCT',
    'FTM','FTA','FT_PCT','OREB','DREB','REB','AST','STL',
    'BLK','TOV','PF','PTS','PLUS_MINUS',
    'WL_WIN','WL_LOSS','IS_HOME',
    'EFF_FG_PCT','TRUE_SHOOTING_PCT','USAGE_APPROX','FANTASY_PTS',
    'PTS_REB_AST','PTS_REB','PTS_AST','REB_AST',
    'DOUBLE_DOUBLE','TRIPLE_DOUBLE',
]

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def american_to_decimal(odds):
    if odds is None: return None
    try: odds = float(odds)
    except: return None
    if odds == 0: return None
    return round(odds/100+1,3) if odds > 0 else round(100/abs(odds)+1,3)

def clean_json(obj):
    """Recursively convert numpy types to native Python for JSON serialisation."""
    import numpy as np
    if isinstance(obj, dict):        return {k: clean_json(v) for k,v in obj.items()}
    if isinstance(obj, list):        return [clean_json(v) for v in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return None if np.isnan(obj) else round(float(obj),4)
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.ndarray):  return [clean_json(v) for v in obj.tolist()]
    if isinstance(obj, float) and obj != obj: return None
    return obj
