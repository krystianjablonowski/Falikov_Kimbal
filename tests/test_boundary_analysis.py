import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fk_transport.boundary_analysis import _match_intersections


class BoundaryAnalysisTests(unittest.TestCase):
    def test_geometry_matching_rejects_distant_branch(self):
        pairs, rejected, ambiguous = _match_intersections(
            [0.8, 1.8], [0.82, 2.4], maximum_separation=0.1,
            ambiguity_tolerance=0.01,
        )
        self.assertEqual(pairs, [(0.8, 0.82)])
        self.assertEqual(rejected, 2)
        self.assertEqual(ambiguous, 0)

    def test_geometry_matching_rejects_ambiguous_pair(self):
        pairs, _, ambiguous = _match_intersections(
            [1.0], [0.99, 1.01], maximum_separation=0.1,
            ambiguity_tolerance=0.025,
        )
        self.assertEqual(pairs, [])
        self.assertEqual(ambiguous, 1)


if __name__ == "__main__":
    unittest.main()
