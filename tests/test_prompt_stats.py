import unittest

from src.cli.prompt_stats import _percentile_nearest_rank
from src.cli.prompt_stats_replicas import _percentile_nearest_rank as _percentile_replica


class PromptStatsTests(unittest.TestCase):
    def test_percentile_nearest_rank(self) -> None:
        vals = [10, 20, 30, 40, 50]
        self.assertEqual(_percentile_nearest_rank(vals, 50.0), 30)
        self.assertEqual(_percentile_nearest_rank(vals, 95.0), 50)
        self.assertEqual(_percentile_nearest_rank(vals, 99.0), 50)

    def test_percentile_handles_empty(self) -> None:
        self.assertEqual(_percentile_nearest_rank([], 95.0), 0)

    def test_replica_percentile_matches(self) -> None:
        vals = [2, 4, 6, 8]
        self.assertEqual(_percentile_replica(vals, 50.0), 4)
        self.assertEqual(_percentile_replica(vals, 95.0), 8)


if __name__ == "__main__":
    unittest.main()
