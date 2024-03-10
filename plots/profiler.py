import pstats
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stats", help="Path to stats file")
    args = parser.parse_args()

    stats_path = args.stats

    # print cumulative time spent in functions part of the dp-streaming codebase
    stats = pstats.Stats(stats_path)
    stats.sort_stats("cumulative")
    stats.print_stats("/Users/aadyaamaddi/Desktop/Research/dp-streaming/")
