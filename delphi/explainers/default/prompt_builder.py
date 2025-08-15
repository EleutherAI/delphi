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


def build_prompt(
    examples: str,
    activations: bool = False,
    cot: bool = False,
) -> list[dict]:
    messages = system(
        cot=cot,
    )

    few_shot_examples = build_examples(
        activations=activations,
        cot=cot,
    )

    messages.extend(few_shot_examples)

    user_start = f"\n{examples}\n"

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
