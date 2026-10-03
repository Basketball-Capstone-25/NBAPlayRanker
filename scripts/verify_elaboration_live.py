"""Verify Iteration-14 against real Supabase sessions without saving tokens.

Run with backend/.venv/bin/python. Credentials stay in a private JSON file with
analyst/coach objects containing email/password; never commit that file.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import secrets
import string

from dotenv import dotenv_values
import requests


class FixtureTracker:
    """Track possible writes before sending them, including negative tests.

    Keys must be fresh and checked empty before the first request. Cleanup uses
    only the fixture owner's ordinary session and never broadens privileges.
    """

    def __init__(self, data, check, expect):
        self.data, self.check, self.expect = data, check, expect
        self.fixtures = []
        self.cleanup_results = []

    def insert(self, name, role, scope, payload, expected, *, request_role=None):
        self.check(name + ' fixture key initially empty',
                   self.expect(name + ' fixture preflight', self.data('GET', role, scope), 200).json() == [])
        fixture = {'name': name, 'role': role, 'scope': scope, 'state': 'possible'}
        self.fixtures.append(fixture)
        response = self.data('POST', request_role or role, payload=payload)
        # These database/API rejections do not commit an INSERT. A timeout,
        # 5xx response, or unexpected success remains a possible write.
        if response.status_code in (400, 401, 403, 404, 409, 422):
            fixture['state'] = 'rejected'
        elif response.status_code == 201:
            fixture['state'] = 'created'
        return self.expect(name, response, expected)

    def cleanup(self):
        all_clean = True
        for fixture in reversed(self.fixtures):
            if fixture['state'] == 'rejected':
                continue
            result = {'fixture': fixture['name'], 'test_owner_role': fixture['role'],
                      'synthetic_matchup': {key: value for key, value in fixture['scope'].items() if key != 'user_id'},
                      'state_before_cleanup': fixture['state'], 'passed': False}
            try:
                response = self.expect('remove ' + fixture['name'],
                                       self.data('DELETE', fixture['role'], fixture['scope']), 200)
                deleted = response.json()
                remaining = self.expect('verify removal of ' + fixture['name'],
                                        self.data('GET', fixture['role'], fixture['scope']), 200).json()
                # A role-isolation regression can leave a row invisible to an
                # analyst. An empty analyst SELECT alone cannot prove removal.
                removed = remaining == [] and (
                    bool(deleted) or (fixture['state'] == 'possible' and fixture['role'] == 'coach'))
                self.check(fixture['name'] + ' fixture removed', removed)
                result['passed'] = True
            except Exception as exc:
                # Do not replace the original test failure or print a response
                # body/session. Preserve a separate, explicit cleanup failure.
                result['error_type'] = type(exc).__name__
                all_clean = False
            self.cleanup_results.append(result)
        return all_clean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--api', required=True)
    parser.add_argument('--credentials', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    config = dotenv_values(Path(__file__).resolve().parents[1] / '.env.local')
    supabase = config['NEXT_PUBLIC_SUPABASE_URL'].rstrip('/')
    key = config['NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY']
    accounts = json.loads(args.credentials.read_text())
    results, sessions, ids = [], {}, {}
    args.output.mkdir(parents=True, exist_ok=True)

    def expect(name, response, status):
        result = {'test': name, 'expected_status': status, 'actual_status': response.status_code,
                  'passed': response.status_code == status}
        results.append(result)
        if not result['passed']:
            raise AssertionError(f'{name}: expected HTTP {status}, got {response.status_code}')
        return response

    def check(name, condition):
        results.append({'test': name, 'passed': bool(condition)})
        if not condition:
            raise AssertionError(name)

    def api(path, role=None, params=None):
        return requests.get(args.api.rstrip('/') + path, headers=sessions.get(role, {}),
                            params=params, timeout=150)

    def data(method, role, params=None, payload=None):
        return requests.request(method, supabase + '/rest/v1/gameplan_notes',
                                headers={'apikey': key, 'Prefer': 'return=representation', **sessions[role]},
                                params=params, json=payload, timeout=30)

    started = datetime.now(timezone.utc).isoformat()
    completed = False
    tracker = FixtureTracker(data, check, expect)
    run_id = secrets.token_hex(16)
    our_team = ''.join(secrets.choice(string.ascii_uppercase) for _ in range(4))
    opp_team = our_team
    while opp_team == our_team:
        opp_team = ''.join(secrets.choice(string.ascii_uppercase) for _ in range(4))
    identity = {'season': '2099-00', 'our_team': our_team, 'opp_team': opp_team}
    marker = {'verification': 'Disposable Iteration-14 test ' + run_id}
    scope = {key: 'eq.' + value for key, value in identity.items()}
    try:
        expect('public health', api('/health'), 200)
        for role in ('analyst', 'coach'):
            response = expect(role + ' password sign-in', requests.post(
                supabase + '/auth/v1/token?grant_type=password', headers={'apikey': key},
                json={k: accounts[role][k] for k in ('email', 'password')}, timeout=30), 200)
            payload = response.json()
            ids[role] = payload['user']['id']
            sessions[role] = {'Authorization': 'Bearer ' + payload['access_token']}
            profile = expect(role + ' protected profile', requests.get(
                supabase + '/rest/v1/profiles', params={'select': 'role'},
                headers={'apikey': key, **sessions[role]}, timeout=30), 200).json()
            check(role + ' trusted profile role', profile == [{'role': role}])
            expect(role + ' metadata', api('/meta/options', role), 200)

        filters = {'season': '2024-25', 'our': 'TOR', 'opp': 'BOS', 'k': 3}
        for path in ('/metrics/topk-uplift', '/metrics/topk-uplift.json', '/metrics/topk-uplift.csv', '/metrics/calibration'):
            parameters = filters if 'topk' in path else None
            expect('anonymous denied ' + path, api(path, params=parameters), 401)
            expect('coach denied ' + path, api(path, 'coach', parameters), 403)
        report = expect('analyst Top-K JSON', api('/metrics/topk-uplift', 'analyst', filters), 200).json()
        downloaded = expect('analyst Top-K JSON attachment', api('/metrics/topk-uplift.json', 'analyst', filters), 200)
        exported = expect('analyst Top-K CSV attachment', api('/metrics/topk-uplift.csv', 'analyst', filters), 200)
        check('JSON download parity', downloaded.json() == report)
        rows = list(csv.DictReader(io.StringIO(exported.text)))
        check('CSV parity', len(rows) == report['k_returned'] and all(
            float(row['uplift_ppp']) == report['uplift_ppp'] and
            row['data_provenance_source_sha256'] == report['data_provenance']['source_sha256'] for row in rows))
        check('attachment headers', all('attachment;' in r.headers.get('content-disposition', '') for r in (downloaded, exported)))
        expect('Top-K invalid K', api('/metrics/topk-uplift', 'analyst', {**filters, 'k': 0}), 422)
        expect('Top-K identical teams', api('/metrics/topk-uplift', 'analyst', {**filters, 'opp': 'TOR'}), 400)
        calibration = expect('analyst calibration', api('/metrics/calibration', 'analyst'), 200).json()
        check('calibration strictly later seasons', all(
            all(season < fold['test_season'] for season in fold['train_seasons']) for fold in calibration['folds']))
        check('calibration observations', calibration['summary']['count'] > 0)
        expect('calibration invalid bins', api('/metrics/calibration', 'analyst', {'n_bins': 1}), 422)
        expect('coach baseline', api('/rank-plays/baseline', 'coach', filters), 200)
        expect('coach baseline CSV', api('/rank-plays/baseline.csv', 'coach', filters), 200)
        expect('analyst coach-API denied', api('/rank-plays/baseline', 'analyst', filters), 403)
        expect('coach context recommendations', api('/rank-plays/context-ml', 'coach', {
            **filters, 'margin': -2, 'period': 4, 'time_remaining': 72}), 200)
        expect('analyst data explorer', api('/data/team-playtypes', 'analyst', {'season': '2024-25', 'team': 'TOR', 'side': 'offense', 'limit': 5}), 200)

        # New random matchup keys avoid user state; owner predicates bound each
        # operation even if a tested RLS policy unexpectedly becomes permissive.
        scope['user_id'] = 'eq.' + ids['coach']
        row = tracker.insert('coach Gameplan insert', 'coach', scope,
                             {**identity, 'notes': marker}, 201).json()[0]
        check('initial server revision', row['revision'] == 1)
        row = expect('coach Gameplan conditional update', data('PATCH', 'coach', {**scope, 'revision': 'eq.1'},
            {'notes': {**marker, 'update': 'Updated disposable test'}}), 200).json()[0]
        check('revision increments', row['revision'] == 2)
        stale = expect('stale conditional update', data('PATCH', 'coach', {**scope, 'revision': 'eq.1'}, {'notes': {}}), 200)
        check('stale update affects zero rows', stale.json() == [])
        check('analyst sees no Gameplan state', expect('analyst Gameplan select', data('GET', 'analyst', scope), 200).json() == [])
        analyst_scope = {**scope, 'user_id': 'eq.' + ids['analyst']}
        tracker.insert('analyst Gameplan insert denied', 'analyst', analyst_scope,
                       {**identity, 'notes': marker}, 403)
        forged_identity = {**identity, 'our_team': opp_team, 'opp_team': our_team}
        forged_scope = {key: 'eq.' + value for key, value in forged_identity.items()}
        forged_scope['user_id'] = 'eq.' + ids['analyst']
        # Preflight and cleanup use the target owner's session. The tested
        # insertion deliberately uses the coach session without changing roles.
        tracker.insert('coach owner forgery denied', 'analyst', forged_scope,
                       {'user_id': ids['analyst'], **forged_identity, 'notes': marker},
                       403, request_role='coach')
        expect('malformed Gameplan JSON denied', data('PATCH', 'coach', scope, {'notes': []}), 400)
        for filename, payload in [('topk-uplift-live.json', report), ('calibration-live.json', calibration)]:
            (args.output / filename).write_text(json.dumps(payload, indent=2) + '\n')
        (args.output / 'topk-uplift-live.csv').write_text(exported.text)
        completed = True
    finally:
        cleanup_completed = tracker.cleanup()
        evidence = {'started_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                    'api': args.api, 'authentication': 'Real password sessions; current protected profile roles',
                    'verification_run_id': run_id,
                    'tests': results, 'cleanup': tracker.cleanup_results,
                    'cleanup_passed': cleanup_completed,
                    'passed': completed and cleanup_completed and all(r['passed'] for r in results)}
        (args.output / 'live-verification.json').write_text(json.dumps(evidence, indent=2) + '\n')
    if not cleanup_completed:
        raise AssertionError('Fixture cleanup could not be confirmed; inspect live-verification.json before retrying.')
    print(f'{len(results)} live checks passed; tokens/passwords were not saved to evidence.')


if __name__ == '__main__':
    main()
