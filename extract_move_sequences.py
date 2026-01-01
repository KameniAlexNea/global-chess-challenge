"""
Extract move sequence training data from Stockfish evaluations.

For each position with a line (sequence of moves), we:
1. Randomly select a point in the sequence
2. Apply moves up to that point to get new position
3. Create training example: current position → next move

This teaches the model to predict moves in expert sequences.
"""

import json
from pathlib import Path
from typing import Dict

import chess
import zstandard as zstd
from tqdm import tqdm


def extract_line_data(record: Dict) -> Dict | None:
    """
    Extract the full line from an eval record.
    Training will randomly pick split points on-the-fly.

    Returns one record with full line, or None if invalid.
    """
    try:
        eval_data = record["evals"][0]
        best_pv = eval_data["pvs"][0]
        line = best_pv["line"]
        moves = line.split()

        # Skip very short lines (need at least 2 moves)
        if len(moves) < 2:
            return None

        # Validate that all moves are legal
        board = chess.Board(record["fen"])
        valid_moves = []
        for move_uci in moves:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    valid_moves.append(move_uci)
                    board.push(move)
                else:
                    break
            except:
                break

        # Need at least 2 valid moves
        if len(valid_moves) < 2:
            return None

        return {
            "fen": record["fen"],
            "line": " ".join(valid_moves),
            "depth": eval_data["depth"],
            "knodes": eval_data.get("knodes", 0),
        }

    except Exception:
        return None


def process_eval_file(
    input_file: Path,
    output_file: Path,
    target_size_mb: float = 500,
    skip_size_mb: float = 0,
):
    """
    Process eval file and extract lines (full move sequences).

    Args:
        input_file: Path to lichess_db_eval.jsonl.zst
        output_file: Path to save extracted lines
        target_size_mb: Stop after processing ~this much compressed data
        skip_size_mb: Skip this many MB before starting extraction
    """
    print(f"Extracting move sequences from {input_file}")
    if skip_size_mb > 0:
        print(f"Skipping first {skip_size_mb}MB of compressed data")
        print(f"Extracting from {skip_size_mb}MB to {skip_size_mb + target_size_mb}MB")
    else:
        print(f"Target: ~{target_size_mb}MB of compressed data")

    dctx = zstd.ZstdDecompressor()
    all_lines = []

    bytes_processed = 0
    bytes_skipped = 0
    skip_bytes = skip_size_mb * 1024 * 1024
    target_bytes = target_size_mb * 1024 * 1024
    records_processed = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "rb") as compressed:
        with dctx.stream_reader(compressed) as reader:
            buffer = ""
            chunk_size = 1024 * 1024  # 1MB chunks

            # Skip phase
            if skip_bytes > 0:
                pbar_skip = tqdm(
                    total=skip_bytes, unit="B", unit_scale=True, desc="Skipping"
                )
                while bytes_skipped < skip_bytes:
                    chunk = reader.read(chunk_size)
                    if not chunk:
                        print(
                            f"\nWarning: File ended at {bytes_skipped/1024/1024:.1f}MB (before skip target)"
                        )
                        pbar_skip.close()
                        return

                    bytes_skipped += len(chunk)
                    pbar_skip.update(len(chunk))

                    # Process buffer to stay aligned with record boundaries
                    buffer += chunk.decode("utf-8")
                    lines = buffer.split("\n")
                    buffer = lines[-1]  # Keep incomplete line

                pbar_skip.close()
                print(f"Skipped {bytes_skipped/1024/1024:.1f}MB")

            # Extraction phase
            pbar = tqdm(
                total=target_bytes, unit="B", unit_scale=True, desc="Processing"
            )

            while bytes_processed < target_bytes:
                chunk = reader.read(chunk_size)
                if not chunk:
                    break

                bytes_processed += len(chunk)
                pbar.update(len(chunk))

                buffer += chunk.decode("utf-8")
                lines = buffer.split("\n")
                buffer = lines[-1]  # Keep incomplete line

                for line in lines[:-1]:
                    if line.strip():
                        try:
                            record = json.loads(line.strip())
                            line_data = extract_line_data(record)
                            if line_data:
                                all_lines.append(line_data)
                            records_processed += 1

                            # Periodically update progress
                            if records_processed % 1000 == 0:
                                pbar.set_postfix(
                                    {
                                        "records": records_processed,
                                        "valid_lines": len(all_lines),
                                    }
                                )
                        except:
                            continue

            pbar.close()

    # Save all lines
    print(f"\nSaving {len(all_lines)} lines to {output_file}")
    with open(output_file, "w") as f:
        for line_data in all_lines:
            f.write(json.dumps(line_data) + "\n")

    # Print statistics
    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Records processed: {records_processed:,}")
    print(f"Valid lines extracted: {len(all_lines):,}")
    if records_processed > 0:
        print(f"Success rate: {len(all_lines)/records_processed*100:.1f}%")
    print(f"Compressed data skipped: {bytes_skipped/1024/1024:.1f} MB")
    print(f"Compressed data processed: {bytes_processed/1024/1024:.1f} MB")
    print(f"Total data read: {(bytes_skipped + bytes_processed)/1024/1024:.1f} MB")
    print(f"Output file: {output_file}")

    # Calculate statistics
    line_lengths = [len(l["line"].split()) for l in all_lines]
    depths = [l["depth"] for l in all_lines]

    print(f"\n{'='*80}")
    print(f"LINE STATISTICS")
    print(f"{'='*80}")
    print(f"Line lengths:")
    print(f"  Min: {min(line_lengths)}")
    print(f"  Max: {max(line_lengths)}")
    print(f"  Avg: {sum(line_lengths)/len(line_lengths):.1f}")
    print(f"Depths:")
    print(f"  Min: {min(depths)}")
    print(f"  Max: {max(depths)}")
    print(f"  Avg: {sum(depths)/len(depths):.1f}")

    # Show sample
    print(f"\n{'='*80}")
    print(f"SAMPLE LINES")
    print(f"{'='*80}")
    for i, line_data in enumerate(all_lines[:3], 1):
        moves = line_data["line"].split()
        print(f"\nLine {i}:")
        print(f"  FEN: {line_data['fen'][:60]}...")
        print(f"  Moves: {' '.join(moves[:10])}{'...' if len(moves) > 10 else ''}")
        print(f"  Length: {len(moves)} moves")
        print(f"  Depth: {line_data['depth']}")
        print(f"  Knodes: {line_data['knodes']}")


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract move sequences from Stockfish eval data"
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("lichess_db_eval.jsonl.zst"),
        help="Path to input lichess_db_eval.jsonl.zst file",
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=Path("data/processed"),
        help="Path to output extracted move sequences",
    )
    parser.add_argument(
        "--target_size_mb",
        type=float,
        default=500,
        help="Target size of compressed data to process (in MB)",
    )
    parser.add_argument(
        "--skip_size_mb",
        type=float,
        default=0,
        help="Size of compressed data to skip before extraction (in MB)",
    )
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    input_file = Path(args.input_file)
    output_file = (
        Path(args.output_folder)
        / f"move_sequences_{args.skip_size_mb}-{int(args.target_size_mb)}mb.jsonl"
    )

    # Extract from ~500MB of compressed data
    # Store full lines - training will pick split points on-the-fly
    # To extract from 500MB-1000MB range, set skip_size_mb=500
    process_eval_file(
        input_file=input_file,
        output_file=output_file,
        target_size_mb=args.target_size_mb,
        skip_size_mb=args.skip_size_mb,  # Set to 500 to extract the next 500MB (500-1000MB range)
    )
