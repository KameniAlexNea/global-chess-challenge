#!/usr/bin/env python3
"""
Quick demo showing exactly what the model receives and must produce.
Run this to see real examples!
"""

import chess


def show_example_1():
    """Simple example with a basic position."""
    print("=" * 80)
    print("EXAMPLE 1: Simple Opening Position")
    print("=" * 80)
    
    # Create a position after 1.e4 e5
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
    
    print("\n📥 INPUT (What the model receives):\n")
    print(f"Position (FEN): {board.fen()}")
    print(f"\nSide to move: {'White' if board.turn else 'Black'}")
    
    legal_moves = [move.uci() for move in board.legal_moves]
    print(f"\nLegal moves ({len(legal_moves)} total):")
    print(" ".join(legal_moves))
    
    print("\n" + "-" * 80)
    print("\n📤 OUTPUT (What the model must produce):\n")
    print("<think>Developing the knight and preparing to control the center</think>")
    print("<uci_move>g1f3</uci_move>")
    
    print("\n" + "=" * 80)


def show_example_2():
    """Tactical puzzle example."""
    print("\n" * 2)
    print("=" * 80)
    print("EXAMPLE 2: Tactical Puzzle (Fork)")
    print("=" * 80)
    
    # Position from actual puzzle database
    board = chess.Board("6k1/5p1p/4p3/4q3/3nN3/2Q3P1/PP3P1P/6K1 w - -")
    
    print("\n📥 INPUT (What the model receives):\n")
    print(f"Position (FEN): {board.fen()}")
    print(f"\nSide to move: {'White' if board.turn else 'Black'}")
    
    legal_moves = [move.uci() for move in board.legal_moves]
    print(f"\nLegal moves ({len(legal_moves)} total):")
    # Print in groups of 10 for readability
    for i in range(0, len(legal_moves), 10):
        print(" ".join(legal_moves[i:i+10]))
    
    print("\n" + "-" * 80)
    print("\n📤 OUTPUT (What the model must produce):\n")
    print("<think>This move creates a fork, attacking both the queen and knight</think>")
    print("<uci_move>e4d2</uci_move>")
    
    print("\n✅ Validation:")
    print(f"   - Move 'e4d2' is in legal moves list: {('e4d2' in legal_moves)}")
    print(f"   - Found <uci_move> tag: ✓")
    print(f"   - Found <think> tag: ✓")
    
    print("\n" + "=" * 80)


def show_example_3():
    """Endgame example."""
    print("\n" * 2)
    print("=" * 80)
    print("EXAMPLE 3: Endgame Position")
    print("=" * 80)
    
    # Rook endgame
    board = chess.Board("8/4R3/1p2P3/p4r2/P6p/1P3Pk1/4K3/8 w - -")
    
    print("\n📥 INPUT (What the model receives):\n")
    print(f"Position (FEN): {board.fen()}")
    print(f"\nSide to move: {'White' if board.turn else 'Black'}")
    
    legal_moves = [move.uci() for move in board.legal_moves]
    print(f"\nLegal moves ({len(legal_moves)} total):")
    for i in range(0, len(legal_moves), 10):
        print(" ".join(legal_moves[i:i+10]))
    
    print("\n" + "-" * 80)
    print("\n📤 OUTPUT (What the model must produce):\n")
    print("<think>Capturing the rook wins material and simplifies to a winning endgame</think>")
    print("<uci_move>e7f7</uci_move>")
    
    print("\n" + "=" * 80)


def show_data_conversion():
    """Show how we convert puzzle data to training format."""
    print("\n" * 2)
    print("=" * 80)
    print("DATA CONVERSION: Puzzle → Training Example")
    print("=" * 80)
    
    print("\n📦 RAW PUZZLE DATA (from lichess_db_puzzle.csv.zst):\n")
    puzzle = {
        'PuzzleId': '000Pw',
        'FEN': '6k1/5p1p/4p3/4q3/3nN3/2Q3P1/PP3P1P/6K1 w - -',
        'Moves': 'e4d2 d4e2 g1f1 e2c3',
        'Rating': '1422',
        'Themes': 'crushing endgame fork short'
    }
    
    for key, value in puzzle.items():
        print(f"  {key}: {value}")
    
    print("\n" + "-" * 80)
    print("\n🔄 PROCESSING STEPS:\n")
    
    board = chess.Board(puzzle['FEN'])
    legal_moves = [move.uci() for move in board.legal_moves]
    first_move = puzzle['Moves'].split()[0]
    themes = puzzle['Themes'].split()
    
    print("  1. Load position from FEN ✓")
    print("  2. Generate legal moves using python-chess ✓")
    print(f"     → Found {len(legal_moves)} legal moves")
    print("  3. Extract first move from solution ✓")
    print(f"     → Best move: {first_move}")
    print("  4. Generate rationale from themes ✓")
    print(f"     → Themes: {', '.join(themes)}")
    
    # Simple rationale generation logic
    if 'fork' in themes:
        rationale = "This move creates a fork, attacking multiple pieces"
    elif 'mate' in themes:
        rationale = "This move leads to checkmate"
    else:
        rationale = "This move gains a winning advantage"
    
    print(f"     → Rationale: {rationale}")
    
    print("\n" + "-" * 80)
    print("\n📝 FINAL TRAINING EXAMPLE:\n")
    
    training_input = f"""Position (FEN): {puzzle['FEN']}

Side to move: White

Legal moves: {' '.join(legal_moves)}

What is your move?"""
    
    training_output = f"""<think>{rationale}</think>
<uci_move>{first_move}</uci_move>"""
    
    print("INPUT:")
    print(training_input)
    print("\nOUTPUT:")
    print(training_output)
    
    print("\n" + "=" * 80)


def show_stats():
    """Show some statistics about typical positions."""
    print("\n" * 2)
    print("=" * 80)
    print("STATISTICS: Typical Position Characteristics")
    print("=" * 80)
    
    positions = [
        ("Opening", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("Middlegame", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5"),
        ("Endgame", "8/8/4k3/8/8/3K4/8/8 w - - 0 1"),
        ("Tactical", "6k1/5p1p/4p3/4q3/3nN3/2Q3P1/PP3P1P/6K1 w - -"),
    ]
    
    print("\n| Phase      | Legal Moves | Complexity |")
    print("|------------|-------------|------------|")
    
    for phase, fen in positions:
        board = chess.Board(fen)
        num_moves = len(list(board.legal_moves))
        print(f"| {phase:10s} | {num_moves:11d} | {'High' if num_moves > 30 else 'Medium' if num_moves > 20 else 'Low':10s} |")
    
    print("\n💡 Key Insights:")
    print("   - Opening: Many legal moves (~20), need positional understanding")
    print("   - Middlegame: Most complex (30-40 moves), requires calculation")
    print("   - Endgame: Fewer moves (10-20), technique-heavy")
    print("   - Tactical: Finding the ONE best move matters most")
    
    print("\n" + "=" * 80)


def main():
    """Run all examples."""
    print("\n")
    print("🎯 CHESS AI INPUT/OUTPUT DEMONSTRATION")
    print("Understanding what the model receives and must produce")
    print("\n")
    
    show_example_1()
    show_example_2()
    show_example_3()
    show_data_conversion()
    show_stats()
    
    print("\n" * 2)
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The model's job is simple:
1. READ: Position + Legal Moves (as text)
2. THINK: What's the best move and why?
3. WRITE: <think>reason</think> <uci_move>move</uci_move>

We have 5.6M puzzles to train on. Each puzzle can be converted to a training
example in seconds. We have MORE than enough data!

Next step: Extract 100K puzzles and start training! 🚀
""")
    print("=" * 80)


if __name__ == "__main__":
    try:
        import chess
    except ImportError:
        print("❌ Error: python-chess not installed")
        print("📦 Install it with: uv sync")
        exit(1)
    
    main()
