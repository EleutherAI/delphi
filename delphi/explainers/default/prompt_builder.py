from .prompts import example, system, system_single_token


def build_examples(
    **kwargs,
):
    examples = []

    for i in range(1, 4):
        prompt, response = example(i, **kwargs)

        messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]

        examples.extend(messages)

    return examples

def build_feature_logits(top_logits: list[str], bot_logits: list[str]):
    prompt = ""
    if top_logits:
        prompt +="Here are the top logits PROMOTED by this pattern:\n"
        prompt += ", ".join(top_logits)
    if bot_logits:
        prompt +="Here are the top logits SUPRESSED by this pattern:\n"
        prompt += ", ".join(bot_logits)

    print(prompt)
    return prompt

def build_prompt(
    examples: str,
    top_logits: list[str],
    bot_logits: list[str],
    activations: bool = False,
    cot: bool = False,
    
) -> list[dict]:
    messages = system(
        cot=cot,
        top_logits= (len(top_logits) > 0),
        bot_logits= (len(bot_logits) > 0),
    )

    few_shot_examples = build_examples(
        activations=activations,
        cot=cot,
    )

    messages.extend(few_shot_examples)
    logits = build_feature_logits(top_logits, bot_logits)

    user_start = f"\n{examples}\n{logits}\n"

    messages.append(
        {
            "role": "user",
            "content": user_start,
        }
    )

    return messages


def build_single_token_prompt(
    examples,
):
    messages = system_single_token()

    user_start = f"WORDS: {examples}"

    messages.append(
        {
            "role": "user",
            "content": user_start,
        }
    )

    return messages
