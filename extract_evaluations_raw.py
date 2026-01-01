"""Extract (decompress) the Lichess evaluation dataset.

This script simply decompresses `lichess_db_eval.jsonl.zst` into a plain JSONL file
(without parsing, filtering, or modifying the content).

It is intended for cases where you want the raw `.jsonl` on disk for tooling that
cannot read `.zst` directly.

Examples
--------
Decompress the full file:

    uv run python extract_evaluations_raw.py \
      --input_file lichess_db_eval.jsonl.zst \
      --output_file data/raw/lichess_db_eval.jsonl

Decompress only the first ~2GB of decompressed output (useful for testing):

    uv run python extract_evaluations_raw.py \
      --input_file lichess_db_eval.jsonl.zst \
      --output_file data/raw/lichess_db_eval_2gb.jsonl \
      --max_output_mb 2048
"""

from __future__ import annotations

import argparse
from pathlib import Path

import zstandard as zstd
from tqdm import tqdm


def decompress_zst_to_file(
    input_file: Path,
    output_file: Path,
    *,
    chunk_size: int = 64 * 1024 * 1024,
    max_output_bytes: int | None = None,
) -> None:
    """Stream-decompress a .zst file to an output file.

    This performs a raw byte-for-byte decompression of the underlying stream.
    It does not parse JSON and does not change line endings.

    Args:
        input_file: Path to `.zst` file.
        output_file: Path to write decompressed bytes.
        chunk_size: Read size (decompressed bytes) per iteration.
        max_output_bytes: Optional cap on decompressed output bytes.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    dctx = zstd.ZstdDecompressor()
    written = 0

    with open(input_file, "rb") as compressed, open(output_file, "wb") as out:
        with dctx.stream_reader(compressed) as reader:
            pbar_total = max_output_bytes if max_output_bytes is not None else None
            pbar = tqdm(
                total=pbar_total, unit="B", unit_scale=True, desc="Decompressing"
            )

            while True:
                if max_output_bytes is not None:
                    remaining = max_output_bytes - written
                    if remaining <= 0:
                        break
                    read_size = min(chunk_size, remaining)
                else:
                    read_size = chunk_size

                chunk = reader.read(read_size)
                if not chunk:
                    break

                out.write(chunk)
                written += len(chunk)
                pbar.update(len(chunk))

            pbar.close()

    print(f"Wrote {written:,} bytes to {output_file}")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompress lichess_db_eval.jsonl.zst to a raw .jsonl file (no processing)."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("lichess_db_eval.jsonl.zst"),
        help="Path to input lichess_db_eval.jsonl.zst",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        default=Path("data/raw/lichess_db_eval.jsonl"),
        help="Path to write decompressed JSONL",
    )
    parser.add_argument(
        "--max_output_mb",
        type=int,
        default=0,
        help="Optional cap for decompressed output size in MB (0 = no limit).",
    )
    parser.add_argument(
        "--chunk_mb",
        type=int,
        default=64,
        help="Decompressed chunk size in MB (default: 64).",
    )
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    max_output_bytes = None
    if args.max_output_mb and args.max_output_mb > 0:
        max_output_bytes = int(args.max_output_mb) * 1024 * 1024

    decompress_zst_to_file(
        args.input_file,
        args.output_file,
        chunk_size=int(args.chunk_mb) * 1024 * 1024,
        max_output_bytes=max_output_bytes,
    )
