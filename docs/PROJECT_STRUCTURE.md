# Project Structure & Development Plan

## 📁 Current Project Organization

```
global-chess-challenge/
├── 📄 Competition Documentation
│   ├── ChessPlayer.md              # Official challenge description
│   ├── puzzles.md                  # Puzzle dataset format doc
│   ├── evaluations.md              # Evaluation dataset format doc
│   ├── COMPETITION_STRATEGY.md     # Our comprehensive strategy (NEW)
│   └── README.md                   # Project readme
│
├── 💾 Datasets (Downloaded)
│   ├── lichess_db_puzzle.csv.zst   # 267 MB - 5.6M puzzles ✅
│   └── lichess_db_eval.jsonl.zst   # 17 GB - 329M evaluations ✅
│
├── 🎮 Starter Kit
│   └── global-chess-challenge-2025-starter-kit/
│       ├── chess-env/              # Core chess environment
│       │   ├── env.py              # Chess game logic
│       │   ├── run_game.py         # Tournament runner
│       │   ├── chess_renderer.py   # Board visualization
│       │   └── agents/             # Agent implementations
│       │       ├── base.py         # Base agent class
│       │       ├── random_agent.py
│       │       ├── stockfish_agent.py
│       │       └── template_agent.py  # For custom agents
│       │
│       ├── player_agents/          # Submission agents
│       │   ├── llm_agent_prompt_template.jinja
│       │   ├── random_agent_flask_server.py
│       │   └── README.md           # How to create agents
│       │
│       ├── local_evaluation.py     # Testing script
│       └── requirements.txt        # Dependencies
│
├── 🛠️ Our Tools (NEW)
│   ├── analyze_data.py             # Dataset exploration script ✅
│   └── main.py                     # Entry point (placeholder)
│
└── ⚙️ Configuration
    ├── pyproject.toml              # Python dependencies
    └── uv.lock                     # Dependency lock file
```

---

## 🎯 Development Phases

### Phase 1: Foundation (Days 1-5) - IN PROGRESS
**Status:** Strategy complete, starting implementation

#### Tasks:
- [x] Understand competition rules and requirements
- [x] Download datasets (puzzles + evaluations)
- [x] Create strategy document
- [x] Create data analysis script
- [ ] **NEXT:** Run data analysis
- [ ] Test starter kit locally
- [ ] Extract first training dataset (10K puzzles)
- [ ] Design prompt template v1
- [ ] Setup training environment
- [ ] First model training
- [ ] First submission

#### Deliverables:
- Working environment
- Sample training data
- Baseline model submission

---

### Phase 2: Optimization (Days 6-10)
**Status:** Planned

#### Tasks:
- [ ] Process evaluation dataset
- [ ] Expand training dataset (100K+ samples)
- [ ] Implement rationale generation
- [ ] Second training iteration
- [ ] Local tournament testing
- [ ] Improved submission
- [ ] Analyze leaderboard results

#### Deliverables:
- Large-scale training dataset
- Improved model (v2)
- Performance metrics

---

### Phase 3: Advanced (Days 11-14)
**Status:** Planned

#### Tasks:
- [ ] RLVR implementation (if time)
- [ ] Prompt engineering optimization
- [ ] Ensemble approaches
- [ ] Opening book integration
- [ ] Final testing and validation
- [ ] Multiple submission variants
- [ ] Documentation and writeup

#### Deliverables:
- Final competition submission
- Complete documentation
- Lessons learned

---

## 🗂️ Planned File Structure (To Create)

```
global-chess-challenge/
│
├── 📊 data/                        # Data processing outputs
│   ├── raw/                        # Decompressed datasets
│   │   ├── puzzles.csv
│   │   └── evaluations.jsonl
│   │
│   ├── processed/                  # Cleaned and formatted
│   │   ├── train_puzzles_10k.jsonl
│   │   ├── train_evals_100k.jsonl
│   │   ├── val_puzzles_1k.jsonl
│   │   └── val_evals_10k.jsonl
│   │
│   └── analysis/                   # Analysis outputs
│       ├── puzzle_stats.json
│       └── eval_stats.json
│
├── 🔧 scripts/                     # Development scripts
│   ├── 01_extract_puzzles.py      # Extract puzzle data
│   ├── 02_extract_evaluations.py  # Extract evaluation data
│   ├── 03_generate_rationales.py  # Create explanations
│   ├── 04_prepare_training.py     # Format for training
│   └── 05_test_agent.py           # Local testing
│
├── 🧠 models/                      # Model development
│   ├── prompts/                    # Prompt templates
│   │   ├── v1_basic.jinja
│   │   ├── v2_enhanced.jinja
│   │   └── v3_optimized.jinja
│   │
│   ├── training/                   # Training scripts
│   │   ├── train_sft.py           # Supervised fine-tuning
│   │   ├── train_rlvr.py          # RLVR training
│   │   └── config.yaml            # Training configs
│   │
│   └── checkpoints/                # Model checkpoints
│       ├── baseline_v1/
│       ├── improved_v2/
│       └── final_v3/
│
├── 🎮 agents/                      # Our custom agents
│   ├── llm_chess_agent_v1.py      # First agent
│   ├── llm_chess_agent_v2.py      # Improved agent
│   └── hybrid_agent.py            # Advanced approaches
│
├── 📈 evaluation/                  # Testing and results
│   ├── local_results/             # Local tournament results
│   │   ├── v1_results.json
│   │   └── v2_results.json
│   │
│   ├── game_logs/                 # PGN game files
│   │   └── *.pgn
│   │
│   └── metrics/                   # Performance tracking
│       └── metrics.csv
│
├── 📝 notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_prompt_testing.ipynb
│   ├── 03_model_analysis.ipynb
│   └── 04_results_visualization.ipynb
│
└── 🚀 submission/                  # Submission files
    ├── submission_v1/
    ├── submission_v2/
    └── final_submission/
```

---

## 🔨 Next Immediate Steps

### TODAY (Priority 1):
1. ✅ Create strategy document
2. ✅ Create data analysis script
3. ⏳ **Run data analysis on both datasets**
4. ⏳ Test local evaluation script
5. ⏳ Create data extraction pipeline

### Commands to Run:
```bash
# 1. Run data analysis
python analyze_data.py

# 2. Test starter kit
cd global-chess-challenge-2025-starter-kit
python local_evaluation.py --help

# 3. Test a baseline agent
cd chess-env
python run_game.py --white random --black stockfish --stockfish-skill 1

# 4. Create directories
mkdir -p data/{raw,processed,analysis}
mkdir -p scripts models/prompts evaluation notebooks submission
```

---

## 📋 Key Dependencies Status

### Installed (from pyproject.toml):
- ✅ python-chess (board logic)
- ✅ stockfish (engine)
- ✅ trueskill (rating)
- ✅ jinja2 (templates)
- ✅ rich (terminal UI)
- ✅ flask (API server)
- ✅ openai (API client)
- ✅ huggingface-hub (model hosting)

### Need to Add:
- ⚠️ zstandard (for dataset decompression) - **CRITICAL**
- ⚠️ pandas (data processing)
- ⚠️ torch (PyTorch for training)
- ⚠️ transformers (HuggingFace models)
- ⚠️ peft (LoRA/QLoRA)
- ⚠️ trl (RLVR training)
- ⚠️ datasets (HuggingFace datasets)
- ⚠️ accelerate (distributed training)
- ⚠️ bitsandbytes (quantization)

---

## 🎓 Learning Resources

### Must Read:
1. ✅ [ChessPlayer.md](ChessPlayer.md) - Challenge description
2. ⏳ [Starter Kit README](../global-chess-challenge-2025-starter-kit/README.md)
3. ⏳ [Player Agents README](../global-chess-challenge-2025-starter-kit/player_agents/README.md)
4. ⏳ AWS Trainium Tutorial: https://www.youtube.com/watch?v=9ihlYCzEuLQ

### Reference Docs:
- Python-chess: https://python-chess.readthedocs.io/
- Stockfish protocol: UCI specification
- TrueSkill: https://trueskill.org/
- Hugging Face TRL: https://huggingface.co/docs/trl/

---

## 💰 Prize Structure

1st Place: **$10,000** + $5,000 credits  
2nd Place: **$5,000** + $2,000 credits  
3rd Place: **$2,000** + $1,000 credits  

**Total:** $17,000 cash + $8,000 credits

---

## ⏰ Timeline

- **Competition Launched:** December 2, 2025
- **Round 1 Deadline:** December 31, 2025 (23:55 UTC)
- **Days Remaining:** ~14 days
- **Today:** December 17, 2025

---

## 🎯 Success Criteria

### Minimum Goal:
- Submit at least one working agent
- Beat random baseline consistently
- Learn the competition framework

### Realistic Goal:
- Submit 3+ agent variants
- Achieve ACPL < 300
- TrueSkill rating ~ Stockfish depth 2-3
- Top 50% of leaderboard

### Stretch Goal:
- Top 10 finish
- ACPL < 200
- High-quality rationales
- Publication-worthy approach

---

## 🤝 Collaboration Notes

This is a team effort! Key responsibilities:

### Data Science Tasks:
- Dataset extraction and processing
- Feature engineering
- Model training and evaluation
- Performance analysis

### ML Engineering Tasks:
- Training pipeline setup
- Model serving infrastructure
- Submission automation
- AWS Trainium integration

### Chess Domain Tasks:
- Opening book creation
- Tactical pattern analysis
- Rationale generation
- Game analysis

---

**Status:** Ready to begin implementation 🚀  
**Next Action:** Run data analysis script  
**Owner:** Team  
**Last Updated:** December 17, 2025
