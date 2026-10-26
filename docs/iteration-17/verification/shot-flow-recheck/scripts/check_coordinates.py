from pathlib import Path
from datetime import datetime,timezone
import argparse,sys,json,hashlib
import numpy as np
import pandas as pd
root = next(p for p in Path(__file__).resolve().parents if (p / 'backend/application').is_dir())
sys.path.insert(0,str(root/'backend'))
from infrastructure.visualization_and_export.viz_shot_heatmap import _coords_to_sportypy_halfcourt,_patch_sportypy_circle,heatmap_plot_data
import matplotlib.pyplot as plt
from sportypy.surfaces.basketball import NBACourt
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
out=args.output
out.mkdir(parents=True,exist_ok=True)
rows=[
 {'name':'home rim','X':41.75,'Y':0.,'HOME_FLAG':True,'expected_x':41.75,'expected_y':0.},
 {'name':'away rim','X':-41.75,'Y':0.,'HOME_FLAG':False,'expected_x':41.75,'expected_y':0.},
 {'name':'home corner','X':41.75,'Y':22.,'HOME_FLAG':True,'expected_x':41.75,'expected_y':22.},
 {'name':'away corner','X':-41.75,'Y':-22.,'HOME_FLAG':False,'expected_x':41.75,'expected_y':22.},
 {'name':'home midrange','X':20.,'Y':5.,'HOME_FLAG':True,'expected_x':20.,'expected_y':5.},
 {'name':'away midrange','X':-20.,'Y':-5.,'HOME_FLAG':False,'expected_x':20.,'expected_y':5.},
 {'name':'home backcourt heave','X':-20.,'Y':3.,'HOME_FLAG':True,'expected_x':-20.,'expected_y':3.},
 {'name':'away backcourt heave','X':20.,'Y':-3.,'HOME_FLAG':False,'expected_x':-20.,'expected_y':3.},
 {'name':'home halfcourt','X':0.,'Y':0.,'HOME_FLAG':True,'expected_x':0.,'expected_y':0.},
 {'name':'away halfcourt','X':0.,'Y':0.,'HOME_FLAG':False,'expected_x':0.,'expected_y':0.},
]
_patch_sportypy_circle();court=NBACourt();fig,ax=plt.subplots();court.draw(ax=ax,display_range='offense')
df=pd.DataFrame(rows);x,y=_coords_to_sportypy_halfcourt(df=df,ax=ax,court=court)
checks=[]
for i,row in enumerate(rows):checks.append({'name':row['name'],'input':{'x':row['X'],'y':row['Y'],'home':row['HOME_FLAG']},'expected':{'x':row['expected_x'],'y':row['expected_y']},'actual':{'x':float(x[i]),'y':float(y[i])},'passed':bool(np.isclose(x[i],row['expected_x']) and np.isclose(y[i],row['expected_y']))})
checks.append({'name':'rim distance to true basket unchanged','passed':bool(np.isclose(np.hypot(x[0]-41.75,y[0]),0) and np.isclose(np.hypot(x[1]-41.75,y[1]),0))})
checks.append({'name':'home and away heaves remain backcourt','passed':bool(x[6]<0 and x[7]<0)})
fixture=pd.concat([df.assign(LOCATION_VALID=True),pd.DataFrame([{'X':np.nan,'Y':1.,'HOME_FLAG':True,'LOCATION_VALID':True},{'X':41.75,'Y':0.,'HOME_FLAG':True,'LOCATION_VALID':False},{'X':49.,'Y':0.,'HOME_FLAG':True,'LOCATION_VALID':True},{'X':41.,'Y':26.,'HOME_FLAG':True,'LOCATION_VALID':True}])],ignore_index=True)
eligible,px,py,counts=heatmap_plot_data(fixture)
checks.append({'name':'invalid locations and physical outside-court points excluded before plotting','passed':len(eligible)==8 and counts=={'n_shots_excluded_location':2,'n_shots_outside_halfcourt':4},'input_rows':len(fixture),'eligible_rows':len(eligible),'counts':counts})
checks.append({'name':'plotted coordinates stay inside physical half court','passed':bool(((px>=0)&(px<=47)&(np.abs(py)<=25)).all())})
limits={'x':list(ax.get_xlim()),'y':list(ax.get_ylim())};physical={k:court.court_params.get(k) for k in ['court_length','court_width','basket_center_to_baseline']}
version=hashlib.sha256((root/'backend/infrastructure/visualization_and_export/viz_shot_heatmap.py').read_bytes()).hexdigest()
report={'checked_at':datetime.now(timezone.utc).isoformat(),'scope':'Synthetic coordinate fixtures against the actual renderer conversion; no live request or dataset read.','renderer_sha256':version,'coordinate_contract':'Cleaned hoopR coordinates are feet from midcourt. Home attacks positive X; away attacks negative X. Rotate both coordinates for away attempts.','court_parameters':physical,'axes_limits':limits,'checks':checks,'passed':all(c['passed'] for c in checks),'observations':['Case A uses the existing feet coordinates directly. Plot-axis padding does not move the basket or alter distance.','Backcourt heaves retain negative X and are not folded into the frontcourt.','The fallback inches conversion is not used for the current cleaned hoopR files.','The current renderer excludes invalid locations and points outside the physical half court before sampling: X from0to47feet and |Y|<=25feet.','Matched, excluded-location, outside-halfcourt, cap-omitted and shown counts are separate; the fixture verifies their physical filtering boundary.']}
(out/'coordinate-transform.json').write_text(json.dumps(report,indent=2)+'\n');plt.close(fig)
print(json.dumps({'passed':report['passed'],'checks':len(checks),'axes_limits':limits,'court_parameters':physical}))

raise SystemExit(0 if report['passed'] else 1)
