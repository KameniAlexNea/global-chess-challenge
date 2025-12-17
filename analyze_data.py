#!/usr/bin/env python3
"""
Quick analysis script for chess datasets.
Explores the Lichess puzzle and evaluation datasets.
"""

import zstandard as zstd
import csv
import json
from pathlib import Path
from collections import Counter
import sys


def analyze_puzzles(file_path: str, sample_size: int = 1000):
    """
    Analyze the puzzle dataset structure and content.
    
    Args:
        file_path: Path to lichess_db_puzzle.csv.zst
        sample_size: Number of puzzles to sample for analysis
    """
    print("=" * 80)
    print("ANALYZING PUZZLE DATASET")
    print("=" * 80)
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"📁 File: {file_path}")
    print(f"📊 Sampling first {sample_size} puzzles...\n")
    
    dctx = zstd.ZstdDecompressor()
    
    ratings = []
    themes_counter = Counter()
    move_counts = []
    
    with open(file_path, 'rb') as compressed:
        with dctx.stream_reader(compressed) as reader:
            text_stream = reader.read().decode('utf-8').splitlines()
            csv_reader = csv.DictReader(text_stream)
            
            print("📋 Column names:")
            print(f"   {', '.join(csv_reader.fieldnames)}\n")
            
            for i, row in enumerate(csv_reader):
                if i >= sample_size:
                    break
                
                # Collect statistics
                if row['Rating']:
                    ratings.append(int(row['Rating']))
                
                if row['Themes']:
                    for theme in row['Themes'].split():
                        themes_counter[theme] += 1
                
                if row['Moves']:
                    move_counts.append(len(row['Moves'].split()))
                
                # Print first 5 as examples
                if i < 5:
                    print(f"Example {i+1}:")
                    print(f"  PuzzleId: {row['PuzzleId']}")
                    print(f"  FEN: {row['FEN']}")
                    print(f"  Moves: {row['Moves']}")
                    print(f"  Rating: {row['Rating']}")
                    print(f"  Themes: {row['Themes']}")
                    print(f"  GameUrl: {row['GameUrl']}")
                    print()
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    if ratings:
        print(f"\n📊 Rating Distribution (n={len(ratings)}):")
        print(f"   Min: {min(ratings)}")
        print(f"   Max: {max(ratings)}")
        print(f"   Mean: {sum(ratings) / len(ratings):.0f}")
        print(f"   Median: {sorted(ratings)[len(ratings)//2]}")
    
    if themes_counter:
        print(f"\n🎯 Top 20 Most Common Themes:")
        for theme, count in themes_counter.most_common(20):
            print(f"   {theme:25s} : {count:4d} ({count/sample_size*100:.1f}%)")
    
    if move_counts:
        print(f"\n♟️  Moves per Puzzle:")
        print(f"   Min: {min(move_counts)}")
        print(f"   Max: {max(move_counts)}")
        print(f"   Mean: {sum(move_counts) / len(move_counts):.1f}")
    
    print()


def analyze_evaluations(file_path: str, sample_size: int = 100):
    """
    Analyze the evaluation dataset structure and content.
    
    Args:
        file_path: Path to lichess_db_eval.jsonl.zst
        sample_size: Number of evaluations to sample
    """
    print("=" * 80)
    print("ANALYZING EVALUATION DATASET")
    print("=" * 80)
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"📁 File: {file_path}")
    print(f"📊 Sampling first {sample_size} positions...")
    print(f"⚠️  Warning: This is a large file (17GB), decompression may take time...\n")
    
    dctx = zstd.ZstdDecompressor()
    
    depths = []
    knodes = []
    pv_counts = []
    eval_types = {'cp': 0, 'mate': 0}
    
    with open(file_path, 'rb') as compressed:
        with dctx.stream_reader(compressed) as reader:
            # Read in chunks to avoid memory issues
            buffer = b''
            line_count = 0
            
            for chunk in iter(lambda: reader.read(100 * 1024 * 1024), b''):  # 100MB chunks
                buffer += chunk
                lines = buffer.split(b'\n')
                buffer = lines[-1]  # Keep incomplete line in buffer
                
                for line in lines[:-1]:
                    if line_count >= sample_size:
                        break
                    
                    if not line.strip():
                        continue
                    
                    try:
                        data = json.loads(line.decode('utf-8'))
                    
                        # Print first 3 as examples
                        if line_count < 3:
                            print(f"Example {line_count+1}:")
                            print(f"  FEN: {data['fen']}")
                            print(f"  Number of evaluations: {len(data['evals'])}")
                            for j, eval_data in enumerate(data['evals'][:2]):  # Show first 2 evals
                                print(f"  Eval {j+1}:")
                                print(f"    Depth: {eval_data['depth']}")
                                print(f"    Knodes: {eval_data['knodes']}")
                                print(f"    PVs: {len(eval_data['pvs'])}")
                                for k, pv in enumerate(eval_data['pvs'][:2]):  # Show first 2 PVs
                                    eval_type = 'cp' if 'cp' in pv else 'mate'
                                    eval_value = pv.get('cp', pv.get('mate', 'N/A'))
                                    print(f"      PV {k+1}: {eval_type}={eval_value}, line={pv['line'][:50]}...")
                            print()
                        
                        # Collect statistics
                        for eval_data in data['evals']:
                            depths.append(eval_data['depth'])
                            knodes.append(eval_data['knodes'])
                            pv_counts.append(len(eval_data['pvs']))
                            
                            for pv in eval_data['pvs']:
                                if 'cp' in pv:
                                    eval_types['cp'] += 1
                                if 'mate' in pv:
                                    eval_types['mate'] += 1
                        
                        line_count += 1
                    
                    except json.JSONDecodeError:
                        continue
                
                if line_count >= sample_size:
                    break
    
    # Statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    
    if depths:
        print(f"\n📊 Depth Distribution (n={len(depths)}):")
        print(f"   Min: {min(depths)}")
        print(f"   Max: {max(depths)}")
        print(f"   Mean: {sum(depths) / len(depths):.1f}")
    
    if knodes:
        print(f"\n🖥️  Knodes Distribution:")
        print(f"   Min: {min(knodes)}")
        print(f"   Max: {max(knodes)}")
        print(f"   Mean: {sum(knodes) / len(knodes):.0f}")
    
    if pv_counts:
        print(f"\n🎯 Principal Variations per Evaluation:")
        print(f"   Min: {min(pv_counts)}")
        print(f"   Max: {max(pv_counts)}")
        print(f"   Mean: {sum(pv_counts) / len(pv_counts):.1f}")
    
    print(f"\n📈 Evaluation Types:")
    print(f"   Centipawn (cp): {eval_types['cp']}")
    print(f"   Mate: {eval_types['mate']}")
    
    print()


def main():
    """Run analysis on both datasets."""
    base_dir = Path(__file__).parent
    
    puzzle_file = base_dir / "lichess_db_puzzle.csv.zst"
    eval_file = base_dir / "lichess_db_eval.jsonl.zst"
    
    # Analyze puzzles (fast)
    analyze_puzzles(str(puzzle_file), sample_size=1000)
    
    # Analyze evaluations (slower, smaller sample)
    analyze_evaluations(str(eval_file), sample_size=100)
    
    print("=" * 80)
    print("🎉 ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n💡 Key Takeaways:")
    print("   1. Puzzle dataset is structured and ready for quick extraction")
    print("   2. Evaluation dataset is large but contains high-quality Stockfish analysis")
    print("   3. Both datasets use standard formats (FEN for positions, UCI for moves)")
    print("   4. Puzzles include difficulty ratings and tactical themes")
    print("   5. Evaluations include multiple PVs and depth information")
    print("\n🚀 Next Steps:")
    print("   1. Extract puzzle positions for initial training")
    print("   2. Filter evaluations by depth (>=20) for high quality")
    print("   3. Create unified training format with both sources")
    print("   4. Generate rationales from themes and PVs")


if __name__ == "__main__":
    try:
        import zstandard
    except ImportError:
        print("❌ Error: zstandard library not installed")
        print("📦 Install it with: pip install zstandard")
        sys.exit(1)
    
    main()
