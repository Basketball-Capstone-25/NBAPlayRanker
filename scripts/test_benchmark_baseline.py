import unittest

from benchmark_baseline import p95, summarize, valid_baseline


class BaselineBenchmarkTests(unittest.TestCase):
    def test_p95_uses_nearest_rank_and_does_not_drop_slow_requests(self):
        self.assertEqual(p95(list(range(1, 21))), 19)
        self.assertEqual(p95([5, 10, 12000]), 12000)
        self.assertIsNone(p95([]))

    def test_a_fast_error_cannot_pass_acceptance(self):
        self.assertFalse(summarize([{"elapsed_ms": 10, "valid_response": False}], 5000)["pass"])
        self.assertFalse(summarize([], 5000)["pass"])
        self.assertFalse(summarize([{"elapsed_ms": 5001, "valid_response": True}], 5000)["pass"])

    def test_response_must_match_the_requested_matchup_and_rank_order(self):
        matchup = {"season": "2024-25", "our": "TOR", "opp": "BOS", "k": 2}
        payload = {"season": "2024-25", "our_team": "TOR", "opp_team": "BOS", "rankings": [
            {"PLAY_TYPE": "Spotup", "PPP_PRED": 1.2, "RATIONALE": "Season matchup estimate"},
            {"PLAY_TYPE": "Postup", "PPP_PRED": 1.1, "RATIONALE": "Season matchup estimate"},
        ]}
        self.assertTrue(valid_baseline(payload, matchup))
        self.assertFalse(valid_baseline({**payload, "opp_team": "NYK"}, matchup))
        self.assertFalse(valid_baseline({**payload, "rankings": payload["rankings"][::-1]}, matchup))
        self.assertFalse(valid_baseline({**payload, "rankings": [payload["rankings"][0]] * 2}, matchup))
        payload["rankings"][0]["PPP_PRED"] = float("nan")
        self.assertFalse(valid_baseline(payload, matchup))


if __name__ == "__main__":
    unittest.main()
