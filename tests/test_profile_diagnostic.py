"""Tests for profile quality diagnostic."""

from __future__ import annotations

from theaios.trustgate.calibration import diagnose_profiles


class TestDiagnoseProfiles:
    def test_perfect_consensus(self) -> None:
        """All questions have unanimous agreement — warns about determinism."""
        profiles = {
            "q1": [("B", 1.0)],
            "q2": [("C", 1.0)],
            "q3": [("A", 1.0)],
        }
        diag = diagnose_profiles(profiles)
        assert diag.status == "good"
        assert diag.mean_consensus == 1.0
        assert diag.frac_all_unique == 0.0
        # Perfect consensus triggers deterministic endpoint warning
        assert len(diag.warnings) == 1
        assert "zero variance" in diag.warnings[0]

    def test_strong_consensus(self) -> None:
        """Most questions have a clear winner."""
        profiles = {
            "q1": [("B", 0.8), ("A", 0.2)],
            "q2": [("C", 0.9), ("D", 0.1)],
            "q3": [("A", 0.7), ("B", 0.2), ("C", 0.1)],
        }
        diag = diagnose_profiles(profiles)
        assert diag.status == "good"
        assert diag.mean_consensus > 0.7
        assert diag.warnings == []

    def test_all_unique_answers(self) -> None:
        """Every sample produces a unique canonical answer — canonicalization failure."""
        profiles = {
            "q1": [("a", 0.1), ("b", 0.1), ("c", 0.1), ("d", 0.1), ("e", 0.1),
                   ("f", 0.1), ("g", 0.1), ("h", 0.1), ("i", 0.1), ("j", 0.1)],
            "q2": [("x", 0.1), ("y", 0.1), ("z", 0.1), ("w", 0.1), ("v", 0.1),
                   ("u", 0.1), ("t", 0.1), ("s", 0.1), ("r", 0.1), ("q", 0.1)],
        }
        diag = diagnose_profiles(profiles)
        assert diag.status == "poor"
        assert diag.frac_all_unique == 1.0
        assert len(diag.warnings) > 0
        # Warning should mention decision point
        assert any("decision point" in w for w in diag.warnings)  # type: ignore[union-attr]

    def test_mixed_quality(self) -> None:
        """Some questions have good consensus, some don't."""
        profiles = {
            "q1": [("B", 0.8), ("A", 0.2)],  # good
            "q2": [("x", 0.2), ("y", 0.2), ("z", 0.2), ("w", 0.2), ("v", 0.2)],  # all equal
            "q3": [("C", 0.9), ("D", 0.1)],  # good
        }
        diag = diagnose_profiles(profiles)
        assert diag.frac_all_unique > 0

    def test_empty_profiles(self) -> None:
        diag = diagnose_profiles({})
        assert diag.status == "poor"
        assert len(diag.warnings) > 0

    def test_weak_consensus_warns(self) -> None:
        """Low mean consensus should produce a warning."""
        profiles = {
            f"q{i}": [(f"a{i}", 0.25), (f"b{i}", 0.25), (f"c{i}", 0.25), (f"d{i}", 0.25)]
            for i in range(10)
        }
        diag = diagnose_profiles(profiles)
        assert diag.mean_consensus == 0.25
        assert diag.status in ("weak", "poor")

    def test_partial_unique_warns(self) -> None:
        """30% all-unique should trigger a weak warning."""
        good = {
            f"q{i}": [("A", 0.8), ("B", 0.2)]
            for i in range(7)
        }
        # Each answer has equal frequency (1/5 = 0.2) → all-unique
        bad = {
            f"q{i+7}": [
                (f"x{i}a", 0.2), (f"x{i}b", 0.2), (f"x{i}c", 0.2),
                (f"x{i}d", 0.2), (f"x{i}e", 0.2),
            ]
            for i in range(3)
        }
        profiles = {**good, **bad}
        diag = diagnose_profiles(profiles)
        assert diag.frac_all_unique == 0.3
        assert diag.status == "weak"
        assert len(diag.warnings) > 0

    def test_single_question(self) -> None:
        profiles = {"q1": [("B", 0.7), ("A", 0.3)]}
        diag = diagnose_profiles(profiles)
        assert diag.status == "good"
        assert diag.mean_consensus == 0.7
