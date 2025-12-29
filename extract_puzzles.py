#!/usr/bin/env python3
"""
Extract chess puzzle data from Lichess database.

This script extracts raw puzzle data without formatting, returning structured
dictionaries matching the competition format.
"""

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import zstandard as zstd
from tqdm import tqdm


@dataclass
class PuzzleData:
    """Represents a single chess puzzle."""

    puzzle_id: str
    fen: str
    moves: List[str]
    rating: int
    themes: List[str]
    game_url: str
    opening_tags: List[str]


class PuzzleExtractor:
    """Extracts and processes chess puzzles."""

    def __init__(
        self,
        puzzle_file: Path,
        output_dir: Path,
        min_rating: int = 1000,
        max_rating: int = 2500,
        max_puzzles: Optional[int] = None,
        filter_ratings: bool = False,
    ):
        self.puzzle_file = puzzle_file
        self.output_dir = output_dir
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.max_puzzles = max_puzzles
        self.filter_ratings = filter_ratings

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self) -> Tuple[List[Dict], Dict]:
        """Extract puzzles and convert to structured data."""
        print(f"📦 Extracting puzzles from {self.puzzle_file}")
        print(f"   Rating range: {self.min_rating} - {self.max_rating}")
        print(f"   Max puzzles: {self.max_puzzles or 'unlimited'}")

        puzzles = self._read_puzzles()
        print(f"\n✅ Read {len(puzzles)} puzzles matching criteria")

        print("\n🔄 Converting to structured format...")
        examples, stats = self._convert_to_examples(puzzles)
        print(f"✅ Created {len(examples)} examples")

        return examples, stats

    def _read_puzzles(self) -> List[PuzzleData]:
        """Read and filter puzzles from compressed CSV."""
        puzzles = []
        skipped = 0
        errors = 0

        dctx = zstd.ZstdDecompressor()

        with open(self.puzzle_file, "rb") as compressed:
            with dctx.stream_reader(compressed) as reader:
                text_stream = reader.read().decode("utf-8").splitlines()
                csv_reader = csv.DictReader(text_stream)

                for row in tqdm(csv_reader, desc="Reading puzzles"):
                    try:
                        rating = int(row["Rating"]) if row["Rating"] else 0

                        if self.filter_ratings and (
                            rating < self.min_rating or rating > self.max_rating
                        ):
                            skipped += 1
                            continue

                        puzzle = PuzzleData(
                            puzzle_id=row["PuzzleId"],
                            fen=row["FEN"],
                            moves=row["Moves"].split(),
                            rating=rating,
                            themes=row["Themes"].split(),
                            game_url=row["GameUrl"],
                            opening_tags=(
                                row.get("OpeningTags", "").split()
                                if row.get("OpeningTags")
                                else []
                            ),
                        )

                        puzzles.append(puzzle)

                        if self.max_puzzles and len(puzzles) >= self.max_puzzles:
                            break

                    except Exception as e:
                        errors += 1
                        if errors < 10:
                            print(f"⚠️  Error parsing row: {e}")
                        continue

        print(f"\n   Matched: {len(puzzles)}")
        print(f"   Skipped (rating): {skipped}")
        print(f"   Errors: {errors}")

        return puzzles

    def _convert_to_examples(
        self, puzzles: List[PuzzleData]
    ) -> Tuple[List[Dict], Dict]:
        """Convert puzzles to structured format, expanding each puzzle into multiple training examples."""
        examples = []
        stats = {
            "total": len(puzzles),
            "total_positions": 0,
            "successful": 0,
            "failed": 0,
            "themes": Counter(),
            "rating_buckets": Counter(),
            "move_counts": [],
        }

        for puzzle in tqdm(puzzles, desc="Converting"):
            try:
                # Expand puzzle into multiple positions (one for each move in the solution)
                puzzle_examples = self._puzzle_to_dict(puzzle)
                examples.extend(puzzle_examples)
                stats["successful"] += 1
                stats["total_positions"] += len(puzzle_examples)

                for theme in puzzle.themes:
                    stats["themes"][theme] += 1
                stats["rating_buckets"][puzzle.rating // 100 * 100] += 1
                stats["move_counts"].append(len(puzzle.moves))

            except Exception as e:
                stats["failed"] += 1
                if stats["failed"] < 10:
                    print(f"⚠️  Failed to convert puzzle {puzzle.puzzle_id}: {e}")

        return examples, stats

    def _puzzle_to_dict(self, puzzle: PuzzleData) -> List[Dict]:
        """Convert a single puzzle to multiple training examples (one per position in the solution).

        For a puzzle where the player is Black with solution [f2g3, e6e7, b2b1, b3c1, b1c1, h6c1]:
        - Position 0: Black to move, should play f2g3
        - Position 2: Black to move (after White's e6e7), should play b2b1
        - Position 4: Black to move (after White's b3c1), should play b1c1

        Returns a list of training examples, one for each position where the puzzle-solving side moves.
        """
        board = chess.Board(puzzle.fen)
        puzzle_side = board.turn  # The side that's solving the puzzle
        examples = []

        # Play through each move in the solution
        for move_index, move_uci in enumerate(puzzle.moves):
            # Only create training examples for positions where puzzle-solving side moves
            if board.turn == puzzle_side:
                legal_moves = [move.uci() for move in board.legal_moves]
                side_to_move = "White" if board.turn == chess.WHITE else "Black"

                # Verify the move is legal
                if move_uci not in legal_moves:
                    raise ValueError(
                        f"Move {move_uci} not in legal moves at position {move_index}"
                    )

                # Remaining moves in the solution from this point
                remaining_solution = puzzle.moves[move_index:]

                examples.append(
                    {
                        "input": {
                            "fen": board.fen(),
                            "legal_moves": legal_moves,
                            "side_to_move": side_to_move,
                        },
                        "output": {"move": move_uci, "solution": remaining_solution},
                        "metadata": {
                            "puzzle_id": puzzle.puzzle_id,
                            "rating": puzzle.rating,
                            "themes": puzzle.themes,
                            "game_url": puzzle.game_url,
                            "opening_tags": puzzle.opening_tags,
                            "move_number": move_index + 1,
                            "total_moves": len(puzzle.moves),
                        },
                    }
                )

            # Apply the move to get to the next position
            board.push_uci(move_uci)

        return examples

    def save(self, examples: List[Dict], stats: Dict, format: str = "jsonl"):
        """Save examples to file."""
        output_file = self.output_dir / f"chess_puzzles_{len(examples)}.{format}"

        print(f"\n💾 Saving to {output_file}")

        if format == "jsonl":
            with open(output_file, "w") as f:
                for example in examples:
                    f.write(json.dumps(example) + "\n")
        elif format == "json":
            with open(output_file, "w") as f:
                json.dump(examples, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

        stats_file = self.output_dir / f"stats_{len(examples)}.json"
        with open(stats_file, "w") as f:
            stats_copy = stats.copy()
            stats_copy["themes"] = dict(stats["themes"].most_common(50))
            stats_copy["rating_buckets"] = dict(sorted(stats["rating_buckets"].items()))

            if stats["move_counts"]:
                stats_copy["move_stats"] = {
                    "min": min(stats["move_counts"]),
                    "max": max(stats["move_counts"]),
                    "mean": sum(stats["move_counts"]) / len(stats["move_counts"]),
                }

            json.dump(stats_copy, f, indent=2)

        print(f"✅ Saved {len(examples)} examples")
        print(f"✅ Saved statistics to {stats_file}")


def print_sample_examples(examples: List[Dict], num_samples: int = 3):
    """Print sample examples for inspection."""
    print("\n" + "=" * 80)
    print(f"📋 SAMPLE EXAMPLES ({num_samples} of {len(examples)})")
    print("=" * 80)

    for i, example in enumerate(examples[:num_samples], 1):
        print(f"\n{'─' * 80}")
        print(f"Example {i}:")
        print(f"{'─' * 80}")
        print(f"\n📥 INPUT:")
        print(f"   FEN: {example['input']['fen']}")
        print(f"   Side to move: {example['input']['side_to_move']}")
        print(
            f"   Legal moves ({len(example['input']['legal_moves'])}): {' '.join(example['input']['legal_moves'][:10])}..."
        )
        print(f"\n📤 OUTPUT:")
        print(f"   Best move: {example['output']['move']}")
        print(f"   Full solution: {' '.join(example['output']['solution'])}")
        print(f"\n📊 METADATA:")
        print(f"   Puzzle ID: {example['metadata']['puzzle_id']}")
        print(f"   Rating: {example['metadata']['rating']}")
        print(f"   Themes: {', '.join(example['metadata']['themes'][:5])}")


def print_statistics(stats: Dict):
    """Print extraction statistics."""
    print("\n" + "=" * 80)
    print("📊 EXTRACTION STATISTICS")
    print("=" * 80)

    print(
        f"\n✅ Success rate: {stats['successful']}/{stats['total']} ({stats['successful']/stats['total']*100:.1f}%)"
    )
    print(f"❌ Failed: {stats['failed']}")
    print(
        f"📍 Total positions extracted: {stats['total_positions']} (avg {stats['total_positions']/max(stats['successful'], 1):.1f} per puzzle)"
    )

    print(f"\n🎯 Top 15 Themes:")
    for theme, count in stats["themes"].most_common(15):
        print(f"   {theme:25s} : {count:5d} ({count/stats['successful']*100:.1f}%)")

    print(f"\n📈 Rating Distribution:")
    for rating, count in sorted(stats["rating_buckets"].items()):
        bar = "█" * (count // 100)
        print(f"   {rating:4d}: {bar} {count}")

    if stats["move_counts"]:
        print(f"\n♟️  Solution Complexity:")
        print(f"   Min moves: {min(stats['move_counts'])}")
        print(f"   Max moves: {max(stats['move_counts'])}")
        print(
            f"   Avg moves: {sum(stats['move_counts']) / len(stats['move_counts']):.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Extract chess puzzles from Lichess database"
    )
    parser.add_argument(
        "--puzzle-file", type=Path, default=Path("lichess_db_puzzle.csv.zst")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-rating", type=int, default=1000)
    parser.add_argument("--max-rating", type=int, default=2500)
    parser.add_argument("--max-puzzles", type=int, default=100000)
    parser.add_argument("--format", choices=["json", "jsonl"], default="jsonl")
    parser.add_argument("--show-samples", type=int, default=3)

    args = parser.parse_args()

    if not args.puzzle_file.exists():
        print(f"❌ Error: Puzzle file not found: {args.puzzle_file}")
        return 1

    print("=" * 80)
    print("🎯 CHESS PUZZLE EXTRACTOR")
    print("=" * 80)

    extractor = PuzzleExtractor(
        puzzle_file=args.puzzle_file,
        output_dir=args.output_dir,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        max_puzzles=args.max_puzzles,
    )

    examples, stats = extractor.extract()
    print_statistics(stats)

    if args.show_samples > 0:
        print_sample_examples(examples, args.show_samples)

    extractor.save(examples, stats, format=args.format)

    print("\n" + "=" * 80)
    print("🎉 EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Output: {args.output_dir}/chess_puzzles_{len(examples)}.{args.format}")

    return 0


if __name__ == "__main__":
    try:
        import chess
        import zstandard
        from tqdm import tqdm
    except ImportError as e:
        print(f"❌ Error: {e}\n📦 Install with: uv sync")
        exit(1)

    exit(main())
