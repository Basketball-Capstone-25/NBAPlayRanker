"""Check the repaired API with ordinary test-account sessions; never save tokens."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time

from dotenv import dotenv_values
from pypdf import PdfReader
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    config = dotenv_values(repo / '.env.local')
    accounts = json.loads((Path.home() / '.config/nbaplayranker/test-accounts-20261003.json').read_text())
    accounts = accounts.get('accounts', accounts)
    auth_url = config['NEXT_PUBLIC_SUPABASE_URL'].rstrip('/')
    public_key = config['NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY']
    api = 'https://nba-playranker-api-937082897804.northamerica-northeast1.run.app'
    sessions, checks = {}, []
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'api': api,
              'revision_under_test': args.revision, 'checks': checks, 'sessions_closed': []}
    args.output.mkdir(parents=True, exist_ok=True)

    def check(name, condition, **details):
        checks.append({'name': name, 'passed': bool(condition), **details})
        if not condition:
            raise AssertionError(name)

    def call(name, route, role=None, params=None, expected=200, method='GET', payload=None):
        started = time.perf_counter()
        response = requests.request(method, api + route, params=params, json=payload,
                                    headers=sessions.get(role, {}), timeout=60, allow_redirects=False)
        check(name, response.status_code == expected, route=route, status=response.status_code,
              expected=expected, elapsed_ms=round((time.perf_counter()-started)*1000, 2))
        return response

    def save(name, response):
        content = response.content
        (args.output / name).write_bytes(content)
        report.setdefault('files', []).append({'file': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})

    try:
        call('public liveness', '/health')
        ready = call('configured readiness', '/ready').json()
        check('readiness is ready', ready.get('ready') is True)
        for role in ('analyst', 'coach'):
            response = requests.post(auth_url + '/auth/v1/token?grant_type=password',
                                     headers={'apikey': public_key}, json={k: accounts[role][k] for k in ('email','password')}, timeout=30)
            check(role + ' sign-in', response.status_code == 200)
            sessions[role] = {'Authorization': 'Bearer ' + response.json()['access_token']}
        for path in ('/meta/datasets', '/meta/data-dictionary', '/metrics/shot-models', '/analysis/shot-ml', '/pbp/shotplan/rank'):
            call('anonymous denied ' + path, path, expected=401,
                 params={'season':'2024-25','our':'TOR','opp':'BOS'})
        sessions['invalid'] = {'Authorization': 'Bearer expired-or-invalid-session'}
        call('invalid session denied', '/meta/datasets', 'invalid', expected=401)
        catalog_response = call('dataset catalog', '/meta/datasets', 'analyst')
        catalog = {d['id']: d for d in catalog_response.json()['datasets']}
        check('six available versioned datasets', len(catalog) == 6 and all(d['status']=='available' and d['version'].startswith('sha256:') for d in catalog.values()))
        check('shot source matches local release', catalog['shots']['version'] == 'sha256:' + hashlib.sha256((repo/'backend/data/pbp/shots_clean.parquet').read_bytes()).hexdigest())
        check('shot retrieval date recorded', bool(catalog['shots']['pulled_at']))
        check('historical Synergy pull date remains unknown', catalog['synergy']['pulled_at'] is None)
        save('dataset-catalog.json', catalog_response)
        dictionary_response = call('data dictionary', '/meta/data-dictionary', 'analyst')
        dictionary = dictionary_response.json()
        columns = dictionary['columns']
        check('dictionary covers each loaded column', {(c['dataset'],c['column']) for c in columns} == {(d['id'],c) for d in catalog.values() for c in d['columns']})
        check('current columns have definitions', all(c['documented'] and c['meaning'] and c['unit'] for c in columns))
        save('data-dictionary.json', dictionary_response)
        filters = {'season':'2024-25', 'our':'TOR', 'opp':'BOS', 'k':5}
        preview = call('play-type preview', '/data/team-playtypes', 'analyst', {'season':'2024-25','team':'TOR','side':'offense','limit':200})
        check('play-type version matches catalog', preview.headers.get('X-Dataset-Version') == catalog['synergy']['version'])
        baseline = call('coach baseline', '/rank-plays/baseline', 'coach', filters)
        check('baseline explains unused game context', len(baseline.json()['rankings']) == 5 and all('score margin, period and clock do not affect' in r['RATIONALE'] for r in baseline.json()['rankings']))
        baseline_csv = call('coach baseline CSV', '/rank-plays/baseline.csv', 'coach', filters)
        baseline_rows = list(csv.DictReader(io.StringIO(baseline_csv.text)))
        check('baseline CSV includes actual source version', len(baseline_rows)==5 and all(r.get('DATASET_VERSION') == catalog['synergy']['version'] for r in baseline_rows))
        call('analyst cannot call coaching route', '/rank-plays/baseline', 'analyst', filters, expected=403)
        parsed = call('parse overtime situation', '/nlp/parse', 'coach', method='POST', payload={'text':'OT down 3 with 0:28 left, need a three'}).json()
        ctx = parsed['context']
        check('overtime parse keeps score and clock', ctx['period']==5 and ctx['margin']==-3 and ctx['time_remaining']==28)
        situation = {**filters, 'margin':ctx['margin'], 'period':ctx['period'], 'time_remaining':ctx['time_remaining']}
        plain = call('overtime ranking', '/rank-plays/context-ml', 'coach', situation)
        flagged = call('need-three ranking', '/rank-plays/context-ml', 'coach', {**situation, 'coaching':json.dumps({'need3':True})})
        p = {r['PLAY_TYPE']:r for r in plain.json()['rankings']}
        q = {r['PLAY_TYPE']:r for r in flagged.json()['rankings']}
        check('coaching flag changes ranking scores', any(p[k]['PPP_CONTEXT'] != q[k]['PPP_CONTEXT'] for k in p.keys() & q.keys()))
        check('ranker uses parsed overtime context', flagged.json()['period']==5 and flagged.json()['margin']==-3 and flagged.json()['time_remaining_period_sec']==28)
        save('context-overtime.json', flagged)
        call('invalid overtime clock rejected', '/rank-plays/context-ml', 'coach', {**situation,'time_remaining':301}, expected=422)
        call('unsupported coaching flag rejected', '/rank-plays/context-ml', 'coach', {**situation,'coaching':'{"arbitrary":true}'}, expected=422)
        for role in ('analyst','coach'):
            shotplan = call(role+' shot plan', '/pbp/shotplan/rank', role, filters)
            check(role+' has shot rankings', bool(shotplan.json().get('top_shot_types')))
            heatmap = call(role+' shot heatmap', '/pbp/viz/shot-heatmap', role, {**filters,'max_shots':1000})
            check(role+' heatmap has image', bool(heatmap.json().get('image_base64')))
            shot_pdf = call(role+' shot PDF', '/export/shotplan.pdf', role, filters)
            pdf = PdfReader(io.BytesIO(shot_pdf.content))
            check(role+' shot PDF is readable and versioned', len(pdf.pages)>0 and catalog['shots']['version'].removeprefix('sha256:') in '\n'.join(p.extract_text() for p in pdf.pages))
            save(role+'-shot-plan.pdf', shot_pdf)
        shots = call('repaired shot preview', '/pbp/shots/preview', 'analyst', {'season':'2025-26','team':'TOR','limit':5000})
        shot_body = shots.json()
        rows = shot_body.get('rows', shot_body.get('data', []))
        check('preview contains valid opposing teams', len(rows)>0 and all(r.get('team',r.get('TEAM_ABBR')) != r.get('opp',r.get('OPP_ABBR')) for r in rows))
        save('shot-preview.json', shots)
        metrics = call('saved shot metrics', '/metrics/shot-models', 'analyst').json()
        check('shot metrics use saved results', metrics.get('cached') is True and bool(metrics.get('metrics')))
        analysis_response = call('saved shot analysis', '/analysis/shot-ml', 'analyst')
        check('shot analysis response', analysis_response.json().get('cached') is True)
        check('shot analysis matches deployed input', analysis_response.json().get('source_sha256') == catalog['shots']['version'].removeprefix('sha256:'))
        save('shot-analysis.json', analysis_response)
        call('coach cannot access analyst metrics', '/metrics/shot-models', 'coach', expected=403)
        call('HTTP cannot retrain shot model', '/pbp/metrics/shot-models', 'analyst', {'refresh':'true'}, expected=400)
        call('unbuilt fold choice is clear', '/metrics/shot-models', 'analyst', {'n_splits':4}, expected=503)
        report['complete'] = True
    except Exception as exc:
        report['error_type'] = type(exc).__name__
        if isinstance(exc, AssertionError): report['failed_check'] = str(exc)
    finally:
        for role in ('analyst','coach'):
            if role in sessions:
                try:
                    response = requests.post(auth_url + '/auth/v1/logout?scope=local', headers={'apikey':public_key, **sessions[role]}, timeout=30)
                    report['sessions_closed'].append({'role':role,'status':response.status_code})
                except Exception:
                    report['sessions_closed'].append({'role':role,'status':'unconfirmed'})
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['passed'] = report.get('complete',False) and all(c['passed'] for c in checks)
        (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'passed':report['passed'],'checks':len(checks),'failure':report.get('failed_check'),'output':str(args.output)}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
