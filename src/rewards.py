from src.utils import extract_xml_answer

def format_reward_func(completions, **kwargs):
    """
    Reward for correct XML format: <rationale>...</rationale><uci_move>...</uci_move>
    Strict on format to teach proper structure.
    """
    rewards = []
    for completion in completions:
        try:
            _, _, has_format = extract_xml_answer(completion)
            rewards.append(1.0 if has_format else 0.0)
        except:
            rewards.append(0.0)
    return rewards


def legality_reward_func(completions, legal_moves, **kwargs):
    """
    Reward if the move is legal. Heavy penalty for illegal moves.
    This is a hard constraint - model must learn chess rules first.
    """
    rewards = []
    for completion, legal in zip(completions, legal_moves):
        try:
            _, move, _ = extract_xml_answer(completion)
            if move is None:
                rewards.append(-1.0)  # Failed to extract any move
                continue

            if move in legal:
                rewards.append(1.0)
            else:
                rewards.append(-2.0)  # Heavy penalty for illegal moves
        except:
            rewards.append(-1.0)
    return rewards


def correctness_reward_func(completions, correct_move, **kwargs):
    """
    Reward if the move is correct (matches puzzle solution).
    Disentangled from format - we try to extract move even if XML is wrong.
    """
    rewards = []
    for completion, correct in zip(completions, correct_move):
        try:
            _, move, _ = extract_xml_answer(completion)
            if move is None:
                rewards.append(-0.5)
                continue

            if move == correct:
                rewards.append(3.0)  # High reward for correct solution
            else:
                rewards.append(-0.5)  # Small penalty for wrong move
        except:
            rewards.append(-0.5)
    return rewards
