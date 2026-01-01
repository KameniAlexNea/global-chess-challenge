#!/usr/bin/env python3
"""
Lightweight inspection script for chess datasets.

This script ONLY analyzes and prints:
- file existence and size
- basic schema (columns/keys)
- a small sample of rows

It does not extract, filter, or write any new datasets.
"""

import csv
import json
from collections import Counter
from io import TextIOWrapper
from pathlib import Path
from typing import Optional

import zstandard as zstd


def _format_bytes(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def analyze_puzzles(file_path: str, sample_size: int = 1000) -> None:
    """
    Analyze the puzzle dataset structure and content.

    Args:
        file_path: Path to lichess_db_puzzle.csv.zst
        sample_size: Number of puzzles to sample for analysis
    """
    path = Path(file_path)
    print("=" * 80)
    print("PUZZLE DATASET (lichess_db_puzzle.csv.zst)")
    print("=" * 80)
    print(f"Path: {path}")
    if not path.exists():
        print("Status: missing")
        print()
        return
    print("Status: found")
    print(f"Compressed size: {_format_bytes(path.stat().st_size)}")
    print(f"Sample rows: {sample_size}")

    dctx = zstd.ZstdDecompressor()

    ratings = []
    themes_counter = Counter()
    move_counts = []

    examples_printed = 0
    with open(path, "rb") as compressed:
        with dctx.stream_reader(compressed) as reader:
            text = TextIOWrapper(reader, encoding="utf-8")
            csv_reader = csv.DictReader(text)

            print("Columns:")
            print(", ".join(csv_reader.fieldnames or []))
            print()

            for i, row in enumerate(csv_reader):
                if i >= sample_size:
                    break

                rating = row.get("Rating")
                if rating:
                    try:
                        ratings.append(int(rating))
                    except ValueError:
                        pass

                themes = row.get("Themes")
                if themes:
                    for theme in themes.split():
                        themes_counter[theme] += 1

                moves = row.get("Moves")
                if moves:
                    move_counts.append(len(moves.split()))

                if examples_printed < 3:
                    examples_printed += 1
                    print(f"Example {examples_printed}:")
                    print(f"  PuzzleId: {row.get('PuzzleId')}")
                    print(f"  FEN: {row.get('FEN')}")
                    print(f"  Moves: {row.get('Moves')}")
                    print(f"  Rating: {row.get('Rating')}")
                    print(f"  Themes: {row.get('Themes')}")
                    print(f"  GameUrl: {row.get('GameUrl')}")
                    print()

    print("Summary:")
    if ratings:
        ratings_sorted = sorted(ratings)
        print(
            f"  Ratings: n={len(ratings)} min={min(ratings)} max={max(ratings)} mean={sum(ratings)/len(ratings):.0f} median={ratings_sorted[len(ratings_sorted)//2]}"
        )
    if move_counts:
        print(
            f"  Moves per puzzle: n={len(move_counts)} min={min(move_counts)} max={max(move_counts)} mean={sum(move_counts)/len(move_counts):.1f}"
        )
    if themes_counter:
        top = themes_counter.most_common(10)
        print("  Top themes:")
        for theme, count in top:
            print(f"    {theme}: {count}")
    print()


def analyze_evaluations(file_path: str, sample_size: int = 100) -> None:
    """
    Analyze the evaluation dataset structure and content.

    Args:
        file_path: Path to lichess_db_eval.jsonl.zst
        sample_size: Number of evaluations to sample
    """
    path = Path(file_path)
    print("=" * 80)
    print("EVALUATION DATASET (lichess_db_eval.jsonl.zst)")
    print("=" * 80)
    print(f"Path: {path}")
    if not path.exists():
        print("Status: missing")
        print()
        return
    print("Status: found")
    print(f"Compressed size: {_format_bytes(path.stat().st_size)}")
    print(f"Sample rows: {sample_size}")

    dctx = zstd.ZstdDecompressor()

    depths = []
    knodes = []
    pv_counts = []
    eval_types = {"cp": 0, "mate": 0}

    examples_printed = 0
    line_count = 0

    with open(path, "rb") as compressed:
        with dctx.stream_reader(compressed) as reader:
            text = TextIOWrapper(reader, encoding="utf-8")
            for line in text:
                if line_count >= sample_size:
                    break
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if examples_printed < 3:
                    examples_printed += 1
                    print(f"Example {examples_printed}:")
                    print(f"  Keys: {', '.join(sorted(data.keys()))}")
                    print(f"  FEN: {data.get('fen')}")
                    evals = data.get("evals") or []
                    print(f"  evals: {len(evals)}")
                    if evals:
                        first = evals[0]
                        pvs = first.get("pvs") or []
                        print(
                            f"  first eval: depth={first.get('depth')} knodes={first.get('knodes')} pvs={len(pvs)}"
                        )
                        if pvs:
                            pv0 = pvs[0]
                            ev_type = (
                                "cp"
                                if "cp" in pv0
                                else "mate" if "mate" in pv0 else "unknown"
                            )
                            ev_value = pv0.get("cp", pv0.get("mate"))
                            print(
                                f"  first PV: {ev_type}={ev_value} line={(pv0.get('line') or '')[:80]}..."
                            )
                    print()

                for eval_data in data.get("evals", []):
                    if "depth" in eval_data:
                        depths.append(eval_data["depth"])
                    if "knodes" in eval_data:
                        knodes.append(eval_data["knodes"])
                    if "pvs" in eval_data:
                        pv_counts.append(len(eval_data["pvs"]))
                        for pv in eval_data["pvs"]:
                            if "cp" in pv:
                                eval_types["cp"] += 1
                            if "mate" in pv:
                                eval_types["mate"] += 1

                line_count += 1

    print("Summary:")
    print(f"  Rows read: {line_count}")
    if depths:
        print(
            f"  Depth: n={len(depths)} min={min(depths)} max={max(depths)} mean={sum(depths)/len(depths):.1f}"
        )
    if knodes:
        print(
            f"  Knodes: n={len(knodes)} min={min(knodes)} max={max(knodes)} mean={sum(knodes)/len(knodes):.0f}"
        )
    if pv_counts:
        print(
            f"  PVs per eval: n={len(pv_counts)} min={min(pv_counts)} max={max(pv_counts)} mean={sum(pv_counts)/len(pv_counts):.1f}"
        )
    print(f"  PV eval types: cp={eval_types['cp']} mate={eval_types['mate']}")
    print()


def main(
    puzzle_file: Optional[str],
    eval_file: Optional[str],
    puzzle_samples: int,
    eval_samples: int,
) -> None:
    base_dir = Path(__file__).parent

    puzzle_path = (
        Path(puzzle_file) if puzzle_file else (base_dir / "lichess_db_puzzle.csv.zst")
    )
    eval_path = (
        Path(eval_file) if eval_file else (base_dir / "lichess_db_eval.jsonl.zst")
    )

    analyze_puzzles(str(puzzle_path), sample_size=puzzle_samples)
    analyze_evaluations(str(eval_path), sample_size=eval_samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze Lichess puzzle/eval datasets (no extraction)."
    )
    parser.add_argument(
        "--puzzle_file",
        type=str,
        default=None,
        help="Path to lichess_db_puzzle.csv.zst",
    )
    parser.add_argument(
        "--eval_file", type=str, default=None, help="Path to lichess_db_eval.jsonl.zst"
    )
    parser.add_argument(
        "--puzzle_samples",
        type=int,
        default=1000,
        help="Number of puzzle rows to sample",
    )
    parser.add_argument(
        "--eval_samples", type=int, default=100, help="Number of eval rows to sample"
    )
    args = parser.parse_args()

    main(
        puzzle_file=args.puzzle_file,
        eval_file=args.eval_file,
        puzzle_samples=args.puzzle_samples,
        eval_samples=args.eval_samples,
    )
