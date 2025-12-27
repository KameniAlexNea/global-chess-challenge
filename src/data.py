from datasets import load_dataset
import json
from src.prompts import user_msg
from transformers import Qwen2TokenizerFast


def load_chess_dataset(
    tokenizer: Qwen2TokenizerFast,
):
    # Load your extracted puzzles - proper streaming, not amateur list loading
    # puzzle_file = "data/processed/chess_puzzles_231961.jsonl"
    # dataset = load_dataset("json", data_files=puzzle_file, split="train")
    dataset = load_dataset("alexneakameni/chess-puzzles", split="train")

    print(f"Loaded {len(dataset)} chess puzzle positions")

    # Sample puzzle
    print("\nSample puzzle:")
    print(json.dumps(dataset[0], indent=2))

    def format_chess_prompt(examples):
        """
        Format chess puzzles into GRPO training format.
        Model must infer tactical patterns from position alone.
        
        Args:
            examples: Batch of samples where each field is a list
        """
        # Extract batched data - each field is a list of dicts
        inputs = examples["input"]
        outputs = examples["output"]
        metadatas = examples["metadata"]
        
        # Process each sample in the batch
        prompts = []
        for inp in inputs:
            prompt = user_msg.format(
                FEN=inp["fen"],
                side_to_move=inp["side_to_move"],
                legal_moves_uci=" ".join(inp["legal_moves"]),
            )
            messages = [{"role": "user", "content": prompt}]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        
        return {
            "prompt": prompts,
            "correct_move": [out["move"] for out in outputs],
            "legal_moves": [inp["legal_moves"] for inp in inputs],
            "puzzle_id": [meta["puzzle_id"] for meta in metadatas],
            "rating": [meta["rating"] for meta in metadatas],
        }

    # Test the formatting with a single sample wrapped as a batch
    test_sample = {
        "input": [dataset[0]["input"]],
        "output": [dataset[0]["output"]],
        "metadata": [dataset[0]["metadata"]],
    }
    sample = format_chess_prompt(test_sample)
    print("Sample prompt:")
    print(sample["prompt"][0])
    print(f"\nCorrect move: {sample['correct_move'][0]}")

    dataset =  dataset.train_test_split(test_size=0.01, seed=42)

    train_dataset = dataset["train"].with_transform(format_chess_prompt)
    eval_dataset = dataset["test"].with_transform(format_chess_prompt)
    return train_dataset, eval_dataset
