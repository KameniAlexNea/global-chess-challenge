"""
Utility functions for tokenizer setup and configuration.
"""


def get_model_template(tokenizer, model_config=None):
    """
    Infer the appropriate chat template based on model type and special tokens.

    Args:
        tokenizer: HuggingFace tokenizer instance
        model_config: Optional model config to help identify model type

    Returns:
        str: Chat template string
    """
    # Get model name/type from config or tokenizer
    model_type = None
    if model_config and hasattr(model_config, "model_type"):
        model_type = model_config.model_type.lower()
    elif hasattr(tokenizer, "name_or_path"):
        model_name = tokenizer.name_or_path.lower()
        # Infer type from name
        if "llama" in model_name:
            model_type = "llama"
        elif "qwen" in model_name:
            model_type = "qwen"
        elif "gemma" in model_name:
            model_type = "gemma"
        elif "granite" in model_name:
            model_type = "granite"
        elif "mistral" in model_name:
            model_type = "mistral"

    # Check for specific special tokens in tokenizer
    special_tokens = {token for token in tokenizer.all_special_tokens}

    # Qwen/ChatML format: <|im_start|> and <|im_end|>
    if "<|im_start|>" in special_tokens or model_type == "qwen":
        return (
            "{%- set default_system = 'You are a chess expert playing a game. Analyze positions carefully and respond with both a rationale explaining your reasoning and your move in UCI notation using <rationale></rationale> and <uci_move></uci_move> tags.' %}"
            "{%- if messages[0]['role'] == 'system' %}"
            "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
            "{%- else %}"
            "{{- '<|im_start|>system\\n' + default_system + '<|im_end|>\\n' }}"
            "{%- endif %}"
            "{%- for message in messages %}"
            "{%- if (message['role'] == 'user') or (message['role'] == 'system' and not loop.first) %}"
            "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
            "{%- elif message['role'] == 'assistant' %}"
            "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}"
            "{%- endif %}"
            "{%- endfor %}"
            "{%- if add_generation_prompt %}"
            "{{- '<|im_start|>assistant\\n' }}"
            "{%- endif %}"
        )

    # Gemma format: <start_of_turn> and <end_of_turn>
    if "<start_of_turn>" in special_tokens or model_type == "gemma":
        return (
            "{%- set default_system = 'You are a chess expert playing a game. Analyze positions carefully and respond with both a rationale explaining your reasoning and your move in UCI notation using <rationale></rationale> and <uci_move></uci_move> tags.' %}"
            "{{ bos_token }}"
            "{%- if messages[0]['role'] == 'system' %}"
            "{%- set first_user_prefix = messages[0]['content'] + '\\n' %}"
            "{%- set loop_messages = messages[1:] %}"
            "{%- else %}"
            "{%- set first_user_prefix = default_system + '\\n' %}"
            "{%- set loop_messages = messages %}"
            "{%- endif %}"
            "{%- for message in loop_messages %}"
            "{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{%- endif %}"
            "{%- if (message['role'] == 'assistant') %}"
            "{%- set role = 'model' %}"
            "{%- else %}"
            "{%- set role = message['role'] %}"
            "{%- endif %}"
            "{{ '<start_of_turn>' + role + '\\n' + (first_user_prefix if loop.first else '') }}"
            "{{ message['content'] | trim }}"
            "{{ '<end_of_turn>\\n' }}"
            "{%- endfor %}"
            "{%- if add_generation_prompt %}"
            "{{'<start_of_turn>model\\n'}}"
            "{%- endif %}"
        )

    # Llama 3 format: <|begin_of_text|><|start_header_id|>
    if "<|start_header_id|>" in special_tokens or model_type == "llama":
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set system_message = messages[0]['content'] %}"
            "{% set messages = messages[1:] %}"
            "{% else %}"
            "{% set system_message = '' %}"
            "{% endif %}"
            "{{ bos_token }}"
            "{% if system_message %}"
            "<|start_header_id|>system<|end_header_id|>\n\n{{ system_message }}<|eot_id|>"
            "{% endif %}"
            "{% for message in messages %}"
            "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            "{% endif %}"
        )

    # Granite format: <|start_of_role|> and <|end_of_role|>
    if "<|start_of_role|>" in special_tokens or model_type == "granite":
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|start_of_role|>system<|end_of_role|>{{ message['content'] }}<|end_of_text|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|start_of_role|>user<|end_of_role|>{{ message['content'] }}<|end_of_text|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|start_of_role|>assistant<|end_of_role|>{{ message['content'] }}<|end_of_text|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|start_of_role|>assistant<|end_of_role|>"
            "{% endif %}"
        )

    # Generic fallback using BOS/EOS tokens
    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""

    print(f"⚠️  Using generic template. BOS='{bos}', EOS='{eos}'")
    return (
        "{% for message in messages %}"
        f"{bos}{{{{ message['role'] }}}}: {{{{ message['content'] }}}}{eos}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        f"{bos}assistant: "
        "{% endif %}"
    )


def ensure_chat_template(tokenizer, model_config=None):
    """
    Check if tokenizer has a chat template and set an appropriate one if absent.

    This function intelligently selects the right chat template based on:
    1. Model type from config
    2. Special tokens present in tokenizer
    3. Model name patterns

    Args:
        tokenizer: HuggingFace tokenizer instance
        model_config: Optional model config to help identify model type

    Returns:
        tokenizer: The tokenizer with chat_template set
    """
    if tokenizer.chat_template is None: # Always resetting the chat_template : tokenizer.chat_template is None
        template = get_model_template(tokenizer, model_config)
        tokenizer.chat_template = template

        # Identify which template was used
        if "<|im_start|>" in template:
            template_name = "Qwen/ChatML"
        elif "<start_of_turn>" in template:
            template_name = "Gemma"
        elif "<|start_header_id|>" in template:
            template_name = "Llama 3"
        elif "<|start_of_role|>" in template:
            template_name = "Granite"
        else:
            template_name = "Generic"

        print(f"⚠️  Tokenizer was missing chat_template. Set {template_name} template.")
    else:
        print("✓ Tokenizer already has chat_template configured.")

    return tokenizer


def setup_tokenizer(model_name_or_path, model_config=None, **kwargs):
    """
    Load a tokenizer and ensure it has a chat template configured.

    Args:
        model_name_or_path: Model name or path to load tokenizer from
        model_config: Optional model config to help identify model type
        **kwargs: Additional arguments to pass to AutoTokenizer.from_pretrained

    Returns:
        tokenizer: Configured tokenizer with chat_template
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    tokenizer = ensure_chat_template(tokenizer, model_config)

    return tokenizer
