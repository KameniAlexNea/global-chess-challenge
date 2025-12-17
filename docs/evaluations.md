**329,127,411** chess positions evaluated with Stockfish.
[Produced by, and for, the Lichess analysis board](https://lichess.org/analysis), running various flavours of Stockfish within user browsers.

[Download lichess_db_eval.jsonl.zst](https://database.lichess.org/lichess_db_eval.jsonl.zst)

This file was last updated on 2025-12-03.

### Format

Evaluations are formatted as JSON; one position per line.

The schema of a position looks like this:

```
{
  "fen":          // the position FEN only contains pieces, active color, castling rights, and en passant square.
  "evals": [      // a list of evaluations, ordered by number of PVs.
      "knodes":   // number of kilo-nodes searched by the engine
      "depth":    // depth reached by the engine
      "pvs": [    // list of principal variations
        "cp":     // centipawn evaluation. Omitted if mate is certain.
        "mate":   // mate evaluation. Omitted if mate is not certain.
        "line":   // principal variation, in UCI_Chess960 format.
}
```

Each position can have multiple evaluations, each with a different number of [PVs](https://www.chessprogramming.org/Principal_Variation).

### Sample

```
{
  "fen": "2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -",
  "evals": [
    {
      "pvs": [
        {
          "cp": 311,
          "line": "g2e4 f7f5 e4b7 c8b7 f2f3 b7f3 e1e6 d8h4 c2h2 h4g4"
        }
      ],
      "knodes": 206765,
      "depth": 36
    },
    {
      "pvs": [
        {
          "cp": 292,
          "line": "g2e4 f7f5 e4b7 c8b7 f2f3 b7f3 e1e6 d8h4 c2h2 h4g4"
        },
        {
          "cp": 277,
          "line": "f4g3 f7f5 e1e5 d8f6 a1e1 b7f7 g2c6 f8d8 d4d5 e6d5"
        }
      ],
      "knodes": 92958,
      "depth": 34
    },
    {
      "pvs": [
        {
          "cp": 190,
          "line": "h5h6 d8h4 h6g7 f8d8 f4g3 h4g4 c2e4 g4e4 g2e4 g8g7"
        },
        {
          "cp": 186,
          "line": "g2e4 f7f5 e4b7 c8b7 f2f3 b7f3 e1e6 d8h4 c2h2 h4g4"
        },
        {
          "cp": 176,
          "line": "f4g3 f7f5 e1e5 f5f4 g2e4 h7f6 e4b7 c8b7 g3f4 f6g4"
        }
      ],
      "knodes": 162122,
      "depth": 31
    }
  ]
}
```

### Notes
