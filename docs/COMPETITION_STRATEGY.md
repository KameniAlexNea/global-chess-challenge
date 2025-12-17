# Global Chess Challenge - Competition Strategy & Roadmap

**Team:** Data Science Collaboration  
**Challenge:** Global Chess Challenge 2025  
**Round 1 Deadline:** December 31, 2025  
**Days Remaining:** ~14 days  

---

## 🎯 Competition Summary

**Objective:** Build a text-only chess agent that:
1. Plays **legal chess moves** in UCI format
2. Provides **one-sentence rationale** explaining each move
3. Achieves high **TrueSkill rating** in tournament play
4. Minimizes **ACPL** (Average Centipawn Loss) vs Stockfish

**Evaluation Metrics:**
- **Game-level:** TrueSkill rating (μ - 3σ), Win/Draw/Loss vs baselines
- **Move-level:** Parse success (>98%), Legality rate (>99%), Top-K alignment, ACPL
- **Rationale:** Presence and quality of explanations

---

## 📊 Current Assets Analysis

### ✅ What We Have:
1. **Data Resources:**
   - ✅ Lichess puzzles: `lichess_db_puzzle.csv.zst` (267 MB, ~5.6M puzzles)
   - ✅ Stockfish evaluations: `lichess_db_eval.jsonl.zst` (17 GB, ~329M positions)
   - ✅ Both datasets are CC0 licensed and competition-approved

2. **Starter Kit:**
   - ✅ Complete chess environment (`chess-env/`)
   - ✅ Multiple baseline agents (Random, First, Last, Stockfish)
   - ✅ Template agent structure
   - ✅ Local evaluation script with tournament system
   - ✅ Example prompt templates (Jinja2)

3. **Infrastructure:**
   - ✅ Python environment setup (pyproject.toml with dependencies)
   - ✅ Stockfish integration
   - ✅ TrueSkill rating system
   - ✅ PGN game logging

### ❓ What We Need:
1. **Model Selection:** Choose base LLM (7-8B range: Llama-3.1-8B, Qwen2.5-7B, Mistral-7B)
2. **Training Data:** Extract and prepare positions from Lichess datasets
3. **Compute:** AWS Trainium instances (provided by competition)
4. **Fine-tuning Pipeline:** LoRA/QLoRA setup with Hugging Face
5. **Submission Pipeline:** Hugging Face model hosting for evaluation

---

## 🗺️ Three-Phase Strategic Approach

### **Phase 1: Foundation & Quick Win (Days 1-5)**
*Goal: Get a working baseline submission above random play*

#### 1.1 Environment Setup ✓
- [x] Verify starter kit installation
- [ ] Test local evaluation against baseline agents
- [ ] Understand input/output format thoroughly
- [ ] Run sample games: Random vs Stockfish

#### 1.2 Data Preparation (Priority: Puzzles First)
**Why puzzles first?**
- Smaller dataset (267 MB vs 17 GB)
- Pre-labeled with correct moves
- Rated by difficulty
- Tagged with tactical themes

**Tasks:**
```python
# Extract puzzle data
- Decompress lichess_db_puzzle.csv.zst
- Parse CSV: PuzzleId, FEN, Moves, Rating, Themes
- Create training samples:
  * Input: FEN + legal moves + side to move
  * Target: first move from puzzle solution
  * Rationale: generate from theme tags
- Split by rating: 1000-1500 (easy), 1500-2000 (medium), 2000+ (hard)
- Sample ~100K diverse positions for initial training
```

#### 1.3 Baseline Model Selection
**Recommended: Qwen2.5-7B-Instruct**
- Reasoning: Strong instruction-following, good for structured output
- Alternative: Llama-3.1-8B-Instruct
- Backup: Mistral-7B-Instruct-v0.3

#### 1.4 First Fine-tuning Pass
```yaml
Approach: Supervised Fine-Tuning (SFT) with QLoRA
Dataset: 100K puzzle positions
Epochs: 2-3
Learning Rate: 2e-4
Batch Size: 4-8 (gradient accumulation)
Target: >95% legality, >90% parse success
```

**Deliverable:** First submission by Day 5

---

### **Phase 2: Optimization & Scaling (Days 6-10)**
*Goal: Improve playing strength and move quality*

#### 2.1 Evaluation Analysis Pipeline
**Extract from lichess_db_eval.jsonl.zst:**
```python
- Decompress evaluation data (careful: 17GB!)
- Filter positions by:
  * Depth >= 20 (high quality)
  * Multiple PVs available
  * Clear evaluation (avoid equal positions)
- Extract:
  * FEN -> Best move (cp evaluation)
  * Top-3 moves with evaluations
  * Principal variation lines
- Create 500K-1M training samples
```

#### 2.2 Enhanced Training Dataset
**Combine sources:**
1. Puzzles: Tactical patterns (100K samples)
2. Evaluations: Best moves from real positions (500K samples)
3. Augmentation: 
   - Different game phases (opening/middle/endgame)
   - Balanced positions and tactical positions
   - Various material distributions

#### 2.3 Rationale Generation Strategy
**Three approaches to test:**

**Option A: Template-based (Fast)**
```python
if "fork" in puzzle_themes:
    rationale = "This move creates a fork attacking multiple pieces"
elif "pin" in puzzle_themes:
    rationale = "This pins the opponent's piece to their king/queen"
# ... more patterns
```

**Option B: Stockfish PV-based**
```python
# Use principal variation to explain
pv = ["e2e4", "e7e5", "g1f3"]
rationale = "Developing knight and controlling center squares"
```

**Option C: LLM-generated**
```python
# Use GPT-4 to generate rationales offline
# Add to training data as labels
```

**Recommendation:** Start with A, migrate to C for quality

#### 2.4 Second Training Round
```yaml
Dataset: 600K combined samples (puzzles + evaluations)
Model: Continue from Phase 1 checkpoint
Training: Additional 2-3 epochs
Focus: Minimize ACPL, improve Top-3 move alignment
```

**Deliverable:** Improved submission by Day 10

---

### **Phase 3: Advanced Techniques & Final Push (Days 11-14)**
*Goal: Squeeze every Elo point before deadline*

#### 3.1 RLVR (Reinforcement Learning with Verifiable Rewards)
**If time permits:**
```python
Setup:
- Use TRL (Transformers Reinforcement Learning)
- Reward function:
  * +1.0 for legal move
  * +0.5 for top-3 move alignment
  * +evaluation_gain / 100 (normalized centipawn improvement)
  * -0.5 for illegal/unparseable
  
Algorithm: PPO or GRPO
Rollouts: 1000-5000 positions
Iterations: 3-5
```

#### 3.2 Ensemble & Hybrid Approaches
**Experiment with:**
1. **Opening Book Integration:** 
   - Hard-code first 3-5 moves from strong opening lines
   - Switch to LLM after move 6

2. **Endgame Tablebase:**
   - Detect tablebase positions (≤6 pieces)
   - Use Syzygy tables for perfect play

3. **Stockfish Verification:**
   - Run Stockfish depth 5 on LLM suggestion
   - If evaluation drops >100cp, pick Stockfish move instead

#### 3.3 Prompt Engineering
**Optimize prompt template:**
```jinja2
Test variations:
- Include move history (last 5 moves)
- Add piece count and material balance
- Highlight checks/captures in legal moves
- Different instruction phrasings
- Chain-of-thought prompting
```

#### 3.4 Testing & Validation
**Run extensive local tournaments:**
```python
Opponents:
- Stockfish Depth 1, 3, 5
- Random Agent
- First Move Agent
- Previous checkpoint models

Games: 100+ per opponent
Measure: TrueSkill μ-3σ, ACPL, legality rate
```

**Deliverable:** Final optimized submission by Day 14

---

## 🔧 Technical Implementation Plan

### Data Pipeline
```python
# data_preparation.py
class ChessDataPipeline:
    1. decompress_datasets()
    2. extract_positions()
    3. augment_with_stockfish()
    4. generate_rationales()
    5. create_train_test_split()
    6. save_to_huggingface_format()
```

### Training Pipeline
```python
# train_chess_llm.py
1. Load base model (Qwen2.5-7B)
2. Setup QLoRA config (r=16, alpha=32)
3. Prepare dataset with prompt template
4. Train with SFTTrainer
5. Save checkpoints to HuggingFace
6. Validate with local evaluation
```

### Evaluation Pipeline
```python
# evaluate_agent.py
1. Load trained model
2. Run local_evaluation.py script
3. Play N games vs each baseline
4. Compute metrics (TrueSkill, ACPL, legality)
5. Log results and PGN files
```

---

## 📈 Success Metrics & Milestones

### Minimum Viable Submission (Day 5):
- ✅ Legality rate: >95%
- ✅ Parse success: >95%
- ✅ TrueSkill: Above Random agent
- ✅ Can complete 10 games without crashes

### Competitive Submission (Day 10):
- ✅ Legality rate: >98%
- ✅ Parse success: >98%
- ✅ TrueSkill: Between Stockfish depth 1-2
- ✅ ACPL: <300 (vs Stockfish depth 10)
- ✅ Top-3 alignment: >30%

### Strong Submission (Day 14):
- ✅ Legality rate: >99%
- ✅ Parse success: >99%
- ✅ TrueSkill: ~Stockfish depth 3
- ✅ ACPL: <200
- ✅ Top-3 alignment: >40%
- ✅ Quality rationales (human evaluation)

---

## 🚧 Risk Mitigation

### Risk 1: AWS Trainium Learning Curve
**Mitigation:** 
- Start training on local GPU with smaller model
- Watch AWS Trainium tutorial videos in parallel
- Switch to Trainium only after code is working

### Risk 2: Illegal Move Generation
**Mitigation:**
- Use constrained decoding (force output from legal move list)
- Add post-processing filter
- Penalize heavily in RLVR

### Risk 3: Poor Rationale Quality
**Mitigation:**
- Rationales are secondary to move quality
- Use simple template-based approach initially
- Improve gradually, don't block on this

### Risk 4: Data Processing Time
**Mitigation:**
- Start with puzzle data (small, fast)
- Process evaluation data in background
- Use parallel processing (p-tqdm)
- Sample strategically (don't need all 329M positions)

---

## 🎓 Key Insights & Best Practices

### 1. **Legality is Non-Negotiable**
- Invalid moves = instant disqualification from game
- Always provide legal moves in prompt
- Consider constrained decoding

### 2. **Quality Over Quantity**
- 100K high-quality positions > 10M noisy positions
- Focus on diverse, interesting positions
- Balance puzzle (tactics) + evaluation (strategy) data

### 3. **Incremental Improvement**
- Submit early and often
- Each submission teaches us something
- Don't wait for "perfect" solution

### 4. **Leverage Domain Knowledge**
- Chess has known patterns (openings, tactics, endgames)
- Don't reinvent the wheel - use opening books
- Endgame tablebases are perfect play (use them!)

### 5. **Test Locally First**
- Local evaluation script is our friend
- Fast iteration = faster learning
- Debug everything locally before submission

---

## 📚 Resources & References

### Documentation:
- Challenge page: https://www.aicrowd.com/challenges/global-chess-challenge-2025
- Starter kit: `global-chess-challenge-2025-starter-kit/README.md`
- Agent creation: `player_agents/README.md`

### Datasets:
- Lichess puzzles: Downloaded (267 MB)
- Lichess evaluations: Downloaded (17 GB)
- Format docs: `puzzles.md`, `evaluations.md`

### Tools:
- `python-chess`: Board representation, move validation
- `stockfish`: Position evaluation
- `trueskill`: Rating system
- Hugging Face: Model hosting & training

### AWS Trainium:
- Tutorial series: https://www.youtube.com/watch?v=9ihlYCzEuLQ
- Optimum Neuron docs: https://huggingface.co/docs/optimum-neuron/

---

## 🎯 Next Immediate Actions

### TODAY:
1. ✅ Review competition rules and starter kit (DONE)
2. [ ] Test local evaluation: `python local_evaluation.py`
3. [ ] Decompress puzzle dataset
4. [ ] Write data extraction script for puzzles
5. [ ] Design prompt template (v1)

### TOMORROW:
6. [ ] Extract 10K sample positions from puzzles
7. [ ] Test prompt with OpenAI API (quick validation)
8. [ ] Setup training environment (choose base model)
9. [ ] Write training script with QLoRA
10. [ ] Start first training run

### THIS WEEK:
11. [ ] Complete Phase 1 (first submission)
12. [ ] Begin evaluation data processing
13. [ ] Analyze first submission results
14. [ ] Iterate on prompt and training data

---

## 💡 Innovative Ideas to Explore

### Advanced Techniques:
1. **Multi-phase Training:**
   - Phase 1: Puzzles (tactics)
   - Phase 2: Evaluations (strategy)
   - Phase 3: Full games (planning)

2. **Difficulty Curriculum:**
   - Start with easy puzzles (1000-1500 rating)
   - Progress to hard (2000+)
   - Model learns incrementally

3. **Synthetic Data Generation:**
   - Use Stockfish to generate positions
   - Play random games, extract interesting positions
   - Create "what if" variations

4. **Rationale Quality Metrics:**
   - Train separate classifier for rationale quality
   - Use as auxiliary loss during training
   - Ensure explanations match moves

5. **Hybrid Architecture:**
   - Small tactical network for pattern recognition
   - LLM for rationale and move selection
   - Combine predictions

---

## 🏆 Competition Strategy

### Positioning:
- **Goal:** Top 3 finish (prizes: $10K / $5K / $2K + compute credits)
- **Strategy:** Fast iteration, solid engineering, leverage data
- **Differentiator:** High-quality rationales + strong play

### Timeline Awareness:
- Round 1 ends Dec 31 (14 days remaining)
- Need multiple submissions for learning
- Don't save "best" for last - learn early

### Community:
- Monitor competition forum
- Share learnings (after securing baseline)
- Learn from others' approaches

---

**Status:** Strategy document complete, ready to execute Phase 1 🚀

**Owner:** Data Science Team  
**Last Updated:** December 17, 2025
