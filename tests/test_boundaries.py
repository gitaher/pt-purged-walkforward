import copy
import unittest

import test_verifier as helpers

from plugin.logic import solve
from verifier import verify_input


class BoundaryTests(unittest.TestCase):
    setUp = helpers.VerifierTests.setUp
    check = helpers.VerifierTests.check

    def reject(self, x):
        self.assertFalse(verify_input("compute", x)["valid"])

    def accept(self, x, gold):
        self.assertTrue(self.check(gold, x))
        self.assertEqual(solve(x), gold)

    def test_incomplete_tail_yields_no_fold(self):
        self.accept(dict(self.x, sessions=self.x["sessions"][:-1]), {"folds": []})

    def test_no_sessions(self):
        self.accept(dict(self.x, sessions=[]), {"folds": []})

    def test_zero_gap_exact_minimum(self):
        x = dict(self.x, label_horizon=0, embargo=0, sessions=self.x["sessions"][:6])
        self.accept(
            x, {"folds": [{"train": x["sessions"][:3], "validation": x["sessions"][3:5], "test": x["sessions"][5:6]}]}
        )

    def test_multiple_rolling_folds(self):
        x = dict(self.x, label_horizon=0, embargo=0, sessions=self.x["sessions"][:7])
        self.accept(
            x,
            {
                "folds": [
                    {"train": x["sessions"][:3], "validation": x["sessions"][3:5], "test": x["sessions"][5:6]},
                    {"train": x["sessions"][1:4], "validation": x["sessions"][4:6], "test": x["sessions"][6:7]},
                ]
            },
        )

    def test_horizon_boundary_not_off_by_one(self):
        x = dict(self.x, label_horizon=1, embargo=0, sessions=self.x["sessions"][:8])
        self.accept(
            x, {"folds": [{"train": x["sessions"][:3], "validation": x["sessions"][4:6], "test": x["sessions"][7:8]}]}
        )

    def test_reversed_or_duplicate_sessions(self):
        self.reject(dict(self.x, sessions=list(reversed(self.x["sessions"]))))
        self.reject(dict(self.x, sessions=self.x["sessions"] + self.x["sessions"][-1:]))

    def test_invalid_sizes(self):
        for k, v in [
            ("train_size", 0),
            ("test_size", 0),
            ("validation_size", True),
            ("embargo", -1),
            ("label_horizon", -1),
        ]:
            self.reject(dict(self.x, **{k: v}))

    def test_bad_calendar_date(self):
        self.reject(dict(self.x, sessions=["2026-02-30"]))

    def test_missing_fold_rejected(self):
        self.assertFalse(self.check({"folds": []}))

    def test_reordered_session_rejected(self):
        out = copy.deepcopy(self.gold)
        out["folds"][0]["train"].reverse()
        self.assertFalse(self.check(out))
