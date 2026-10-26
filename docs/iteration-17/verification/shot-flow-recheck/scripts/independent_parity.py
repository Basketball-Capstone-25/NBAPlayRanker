"""Independent count/rate/rank audit; deliberately does not import backend math."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math
import numpy as np
import pandas as pd

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
root = next(p for p in Path(__file__).resolve().parents if (p / 'backend/application').is_dir())
out=args.output;out.mkdir(parents=True,exist_ok=True)
files={'shots':'backend/data/pbp/shots_clean.parquet','shots_canonical':'backend/data/pbp/cache/shots_canonical.parquet','shot_aggregates':'backend/data/pbp/shots_agg.parquet','shot_league':'backend/data/pbp/shots_agg_league.parquet'}
def digest(path):
 with path.open('rb') as f:return 'sha256:'+hashlib.file_digest(f,'sha256').hexdigest()
versions={k:digest(root/v) for k,v in files.items()}
clean=pd.read_parquet(root/files['shots']);canonical=pd.read_parquet(root/files['shots_canonical']);actual=pd.read_parquet(root/files['shot_aggregates']);actual_league=pd.read_parquet(root/files['shot_league'])
report={'started_at':datetime.now(timezone.utc).isoformat(),'calculation':'Independent pandas grouping and arithmetic; no application aggregation or ranking function imported.','source_versions':versions,'checks':[],'limitations':['EPA is points per recorded shot attempt, not points per possession.','Reliability weights are log-scaled sample weights, not calibrated confidence probabilities or intervals.','FG% is computed here as makes/attempts; the shot rank API does not return it.','The current shot rank/heatmap routes have no minimum-attempt threshold parameter.','Matching derived data does not by itself validate event classification; the separate raw-source zone audit is required.']}
def check(name,passed,**details):report['checks'].append({'name':name,'passed':bool(passed),**details})
def same_numeric(left,right):return len(left)==len(right) and np.allclose(np.asarray(left,dtype=float),np.asarray(right,dtype=float),rtol=1e-11,atol=1e-12,equal_nan=True)
mapcols={'SEASON_STR':'season','TEAM_ABBR':'team','OPP_ABBR':'opp','SHOT_TYPE':'shot_type','ZONE':'zone','MADE':'is_make','POINTS':'points','X':'x','Y':'y','HOME_FLAG':'home','MARGIN':'margin','LOCATION_VALID':'location_valid','ZONE_METHOD':'zone_method'}
for a,b in mapcols.items():
 left=clean[a].reset_index(drop=True);right=canonical[b].reset_index(drop=True)
 passed=same_numeric(left,right) if pd.api.types.is_numeric_dtype(left) else len(left)==len(right) and left.fillna('<null>').astype(str).equals(right.fillna('<null>').astype(str))
 check('canonical mirrors cleaned '+a,passed,rows=len(left))
check('no self-opponents',(clean.TEAM_ABBR!=clean.OPP_ABBR).all(),rows=len(clean))
check('made values are binary',clean.MADE.isin([0,1]).all())
check('misses have zero points',clean.loc[clean.MADE==0,'POINTS'].eq(0).all())
check('made points are two or three',clean.loc[clean.MADE==1,'POINTS'].isin([2,3]).all())
computed={};league_tables={}
for level,labels in [('shot_type',['SHOT_TYPE']),('zone',['ZONE']),('shot_type_zone',['SHOT_TYPE','ZONE'])]:
 frames=[]
 for role,teamcol in [('offense','TEAM_ABBR'),('defense_allowed','OPP_ABBR')]:
  g=clean.groupby(['SEASON_STR',teamcol]+labels,dropna=True).agg(attempts=('MADE','size'),makes=('MADE','sum'),points=('POINTS','sum')).reset_index().rename(columns={teamcol:'TEAM_ABBR'})
  g['ROLE']=role;frames.append(g)
 tab=pd.concat(frames,ignore_index=True);tab['EPA']=tab['points']/tab['attempts'];tab['RELIABILITY_WEIGHT']=np.log1p(tab['attempts'])/math.log1p(int(tab['attempts'].max()));tab['FG_PCT']=tab['makes']/tab['attempts']
 keys=['ROLE','SEASON_STR','TEAM_ABBR']+labels
 expected=tab.set_index(keys).sort_index();observed=actual.loc[actual.LEVEL==level].set_index(keys).sort_index()
 indexok=expected.index.equals(observed.index);check(level+' aggregate categories',indexok,expected_rows=len(expected),actual_rows=len(observed))
 for field in ['attempts','makes','points','EPA','RELIABILITY_WEIGHT']:
  ok=indexok and same_numeric(expected[field],observed[field]);check(level+' aggregate '+field,ok,max_absolute_error=float(np.nanmax(np.abs(expected[field].to_numpy()-observed[field].to_numpy()))) if indexok else None)
 check(level+' reliability is bounded',tab.RELIABILITY_WEIGHT.between(0,1).all())
 lg=clean.groupby(['SEASON_STR']+labels,dropna=True).agg(LEAGUE_ATTEMPTS=('MADE','size'),LEAGUE_POINTS=('POINTS','sum')).reset_index();lg['EPA_LEAGUE']=lg.LEAGUE_POINTS/lg.LEAGUE_ATTEMPTS
 for role in ['offense','defense_allowed']:
  seen=actual_league.loc[(actual_league.LEVEL==level)&(actual_league.ROLE==role)].set_index(['SEASON_STR']+labels).sort_index();wanted=lg.set_index(['SEASON_STR']+labels).sort_index();idx=wanted.index.equals(seen.index)
  check(level+' '+role+' league rates',idx and all(same_numeric(wanted[field],seen[field]) for field in ['LEAGUE_ATTEMPTS','LEAGUE_POINTS','EPA_LEAGUE']),rows=len(wanted))
 computed[level]=tab;league_tables[level]=lg
contexts=[('2021-22','TOR','BOS'),('2021-22','BOS','TOR'),('2025-26','TOR','BOS'),('2025-26','BOS','TOR'),('2024-25','LAL','GSW'),('2025-26','OKC','SAS')]
expected_cases=[]
for index,(season,our,opp) in enumerate(contexts):
 for weight in ([0.0,0.7,1.0] if index==0 else [0.7]):
  result={'season':season,'our_team':our,'opp_team':opp,'k':10,'w_off':weight,'w_def':1-weight}
  for level,label,key in [('shot_type','SHOT_TYPE','top_shot_types'),('zone','ZONE','top_zones')]:
   table=computed[level];off=table.loc[(table.ROLE=='offense')&(table.SEASON_STR==season)&(table.TEAM_ABBR==our)].set_index(label);deff=table.loc[(table.ROLE=='defense_allowed')&(table.SEASON_STR==season)&(table.TEAM_ABBR==opp)].set_index(label);league=league_tables[level].loc[league_tables[level].SEASON_STR==season].set_index(label)
   rows=[]
   for category in sorted(set(off.index)&set(deff.index)):
    if level=='zone' and category=='unknown':continue
    o=off.loc[category];d=deff.loc[category];l=league.loc[category];shrunk_o=float(o.RELIABILITY_WEIGHT*o.EPA+(1-o.RELIABILITY_WEIGHT)*l.EPA_LEAGUE);shrunk_d=float(d.RELIABILITY_WEIGHT*d.EPA+(1-d.RELIABILITY_WEIGHT)*l.EPA_LEAGUE)
    rows.append({label:category,'EPA_PRED':weight*shrunk_o+(1-weight)*shrunk_d,'EPA_OFF_SHRUNK':shrunk_o,'EPA_DEF_SHRUNK':shrunk_d,'EPA_OFF':float(o.EPA),'EPA_DEF':float(d.EPA),'EPA_LEAGUE_OFF':float(l.EPA_LEAGUE),'EPA_LEAGUE_DEF':float(l.EPA_LEAGUE),'attempts_OFF':int(o.attempts),'attempts_DEF':int(d.attempts),'RELIABILITY_WEIGHT_OFF':float(o.RELIABILITY_WEIGHT),'RELIABILITY_WEIGHT_DEF':float(d.RELIABILITY_WEIGHT),'FG_PCT_OFF':float(o.FG_PCT),'FG_PCT_DEF':float(d.FG_PCT)})
   rows.sort(key=lambda r:(-r['EPA_PRED'],-r['attempts_OFF']));result[key]=rows[:10]
  expected_cases.append(result)
(out/'expected-rankings.json').write_text(json.dumps(expected_cases,indent=2,allow_nan=False)+'\n')
counts=[]
for season,team,opp in contexts:
 df=clean.loc[(clean.SEASON_STR==season)&(clean.TEAM_ABBR==team)&(clean.OPP_ABBR==opp)]
 oriented_x=df.X.to_numpy()*np.where(df.HOME_FLAG.to_numpy(),1.0,-1.0)
 valid=np.isfinite(df.X.to_numpy())&np.isfinite(df.Y.to_numpy())&df.LOCATION_VALID.fillna(False).to_numpy(dtype=bool)
 on=(oriented_x>=0)&(oriented_x<=47)&(df.Y.abs().to_numpy()<=25)
 eligible=valid&on
 small=df.loc[eligible].groupby(['SHOT_TYPE','ZONE']).size().sort_values().iloc[:1]
 counts_for_plot={'n_shots_total':len(df),'n_shots_excluded_location':int((~valid).sum()),'n_shots_outside_halfcourt':int((valid&~on).sum()),'n_shots_rendered':min(1000,int(eligible.sum())),'n_shots_omitted_by_cap':max(0,int(eligible.sum())-1000)}
 counts.append({'season':season,'team':team,'opp':opp,'attempts':len(df),'makes':int(df.MADE.sum()),'points':int(df.POINTS.sum()),'fg_pct':float(df.MADE.mean()),'points_per_attempt':float(df.POINTS.mean()),'small_sample':{'shot_type':small.index[0][0],'zone':small.index[0][1],'attempts':int(small.iloc[0])},'heatmap_counts':counts_for_plot,'zone_counts':df.groupby('ZONE').size().to_dict(),'type_counts':df.groupby('SHOT_TYPE').size().to_dict()})
(out/'expected-matchups.json').write_text(json.dumps(counts,indent=2,allow_nan=False)+'\n')
check('files remained stable through audit',versions=={k:digest(root/v) for k,v in files.items()})
report['finished_at']=datetime.now(timezone.utc).isoformat();report['passed']=all(c['passed'] for c in report['checks']);report['expected_rank_contexts']=len(expected_cases);report['matchups']=len(counts)
(out/'local-parity.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'passed':report['passed'],'checks':len(report['checks']),'failed':[c['name'] for c in report['checks'] if not c['passed']],'source_versions':versions}))

raise SystemExit(0 if report['passed'] else 1)
