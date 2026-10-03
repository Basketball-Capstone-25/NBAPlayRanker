"""Offline safety checks: no production sessions or database requests."""
import unittest
from unittest.mock import Mock

from verify_elaboration_live import FixtureTracker


def response(status=200, payload=None):
    value = Mock(status_code=status)
    value.json.return_value = [] if payload is None else payload
    return value


def expect(name, actual, status):
    if actual.status_code != status:
        raise AssertionError(name)
    return actual


def check(name, condition):
    if not condition:
        raise AssertionError(name)


class FixtureSafetyTests(unittest.TestCase):
    def setUp(self):
        self.scope = {'user_id': 'eq.fixture-user', 'season': 'eq.2099-00',
                      'our_team': 'eq.XQAZ', 'opp_team': 'eq.PBWN'}
        self.payload = {'season': '2099-00', 'our_team': 'XQAZ', 'opp_team': 'PBWN'}

    def tracker(self, responses):
        self.data = Mock(side_effect=responses)
        return FixtureTracker(self.data, check, expect)

    def test_existing_state_is_never_inserted_or_deleted(self):
        tracker = self.tracker([response(payload=[{'notes': {'real': 'state'}}])])
        with self.assertRaises(AssertionError):
            tracker.insert('insert', 'coach', self.scope, self.payload, 201)
        self.assertTrue(tracker.cleanup())
        self.assertEqual(self.data.call_count, 1)
        self.assertEqual(tracker.fixtures, [])

    def test_insert_json_failure_is_still_cleaned(self):
        created = response(201)
        created.json.side_effect = ValueError('invalid response JSON')
        tracker = self.tracker([response(), created, response(payload=[{'revision': 1}]), response()])
        with self.assertRaises(ValueError):
            tracker.insert('insert', 'coach', self.scope, self.payload, 201).json()
        self.assertTrue(tracker.cleanup())
        self.assertEqual(self.data.call_args_list[2].args, ('DELETE', 'coach', self.scope))

    def test_uncertain_transport_write_attempts_bounded_cleanup(self):
        tracker = self.tracker([response(), TimeoutError('request uncertainty'), response(), response()])
        with self.assertRaises(TimeoutError):
            tracker.insert('insert', 'coach', self.scope, self.payload, 201)
        self.assertTrue(tracker.cleanup())
        self.assertEqual(self.data.call_args_list[2].args, ('DELETE', 'coach', self.scope))

    def test_expected_rejection_needs_no_delete(self):
        tracker = self.tracker([response(), response(403)])
        tracker.insert('denied', 'analyst', self.scope, self.payload, 403)
        self.assertTrue(tracker.cleanup())
        self.assertEqual(self.data.call_count, 2)

    def test_unexpected_negative_test_success_uses_owner_for_cleanup(self):
        tracker = self.tracker([response(), response(201), response(payload=[{'revision': 1}]), response()])
        with self.assertRaises(AssertionError):
            tracker.insert('forgery', 'analyst', self.scope, self.payload, 403, request_role='coach')
        self.assertTrue(tracker.cleanup())
        self.assertEqual(self.data.call_args_list[1].args, ('POST', 'coach'))
        self.assertEqual(self.data.call_args_list[2].args, ('DELETE', 'analyst', self.scope))

    def test_invisible_analyst_row_is_not_claimed_removed(self):
        tracker = self.tracker([response(), response(201), response(), response()])
        with self.assertRaises(AssertionError):
            tracker.insert('denied', 'analyst', self.scope, self.payload, 403)
        self.assertFalse(tracker.cleanup())
        self.assertFalse(tracker.cleanup_results[0]['passed'])

    def test_cleanup_error_does_not_hide_original_test_failure(self):
        tracker = self.tracker([response(), response(201), response(403)])
        with self.assertRaisesRegex(AssertionError, 'denied'):
            try:
                tracker.insert('denied', 'analyst', self.scope, self.payload, 403)
            finally:
                self.assertFalse(tracker.cleanup())
        self.assertEqual(tracker.cleanup_results[0]['error_type'], 'AssertionError')


if __name__ == '__main__':
    unittest.main()
