from pathlib import Path
from datetime import datetime,timezone
import argparse,sys,json,math,base64,hashlib
import numpy as np
import pandas as pd
root = next(p for p in Path(__file__).resolve().parents if (p / 'backend/application').is_dir())
sys.path.insert(0,str(root/'backend'))
from domain.baseline_recommendation.shot_baseline_recommender import ShotBaselineRecommender
from infrastructure.data_access.pbp_viz import _heatmap_data,render_pbp_heatmap_json
from infrastructure.visualization_and_export.viz_shot_heatmap import heatmap_plot_data
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
out=args.output
expect=json.loads((out/'expected-rankings.json').read_text());matchups=json.loads((out/'expected-matchups.json').read_text());local=json.loads((out/'local-parity.json').read_text());assert local['passed']
checks=[]
def check(name,passed,**details):checks.append({'name':name,'passed':bool(passed),**details})
rec=ShotBaselineRecommender();check('rank input versions match checked files',rec.agg_df.attrs['dataset_version']==local['source_versions']['shot_aggregates'] and rec.league_df.attrs['dataset_version']==local['source_versions']['shot_league'])
responses=[]
for i,wanted in enumerate(expect):
 actual=rec.rank(wanted['season'],wanted['our_team'],wanted['opp_team'],wanted['k'],wanted['w_off']);responses.append({'context':{k:v for k,v in wanted.items() if not isinstance(v,list)},'result':actual})
 for group,label in [('top_shot_types','SHOT_TYPE'),('top_zones','ZONE')]:
  observed=actual[group];ref=wanted[group];check(f'context{i+1} {group} rank order',[r[label] for r in observed]==[r[label] for r in ref])
  errors=[]
  for row,expected in zip(observed,ref):
   for field,value in expected.items():
    if field==label or field.startswith('FG_'):continue
    if not math.isclose(float(row[field]),value,rel_tol=1e-10,abs_tol=1e-11):errors.append({'category':expected[label],'field':field,'expected':value,'actual':row[field]})
  check(f'context{i+1} {group} values',len(observed)==len(ref) and not errors,errors=errors)
  if group=='top_zones':check(f'context{i+1} no unknown recommended zone',all(r[label]!='unknown' for r in observed))
counts=[]
for i,ref in enumerate(matchups):
 request,frame,observed=_heatmap_data(season=ref['season'],team=ref['team'],opp=ref['opp'],max_shots=1000)
 expected=ref['heatmap_counts'];check(f'heatmap{i+1} every count matches source',all(observed.get(k)==v for k,v in expected.items()),expected=expected,actual=observed)
 check(f'heatmap{i+1} counts conserved',observed['n_shots_total']==sum(observed[k] for k in ['n_shots_rendered','n_shots_excluded_location','n_shots_outside_halfcourt','n_shots_omitted_by_cap']))
 check(f'heatmap{i+1} exact loaded canonical version',frame.attrs.get('dataset_version')==observed.get('dataset_version')==local['source_versions']['shots_canonical'])
 eligible,x,y,cropped=heatmap_plot_data(frame);check(f'heatmap{i+1} renderer drops no sampled rows',len(eligible)==len(frame) and all(v==0 for v in cropped.values()))
 counts.append({'season':ref['season'],'team':ref['team'],'opp':ref['opp'],'counts':observed})
# Verify the actual image for the capped context without an API request.
ref=matchups[-1];image=render_pbp_heatmap_json(season=ref['season'],team=ref['team'],opp=ref['opp'],max_shots=1000)
png=base64.b64decode(image.pop('image_base64'));(out/'local-capped-heatmap.png').write_bytes(png);image.update(image_file='local-capped-heatmap.png',image_sha256=hashlib.sha256(png).hexdigest());(out/'local-capped-heatmap.json').write_text(json.dumps(image,indent=2)+'\n')
check('capped image returns exact counts',all(image[k]==v for k,v in ref['heatmap_counts'].items()))
(out/'local-rankings.json').write_text(json.dumps(responses,indent=2)+'\n');(out/'local-plot-counts.json').write_text(json.dumps(counts,indent=2)+'\n')
report={'checked_at':datetime.now(timezone.utc).isoformat(),'scope':'Actual local rank and heatmap components compared with independently computed event-level expectations; no HTTP requests or model fitting.','source_versions':local['source_versions'],'checks':checks,'passed':all(c['passed'] for c in checks)};(out/'local-component-parity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':report['passed'],'checks':len(checks),'failures':[c for c in checks if not c['passed']],'plot_counts':counts}))

raise SystemExit(0 if report['passed'] else 1)
