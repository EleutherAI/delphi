DSCORER_SYSTEM_PROMPT = """You are an intelligent and meticulous linguistics researcher doing a "intruder detection" task.

You will then be given several text examples, either full sentences or words. Your task is to determine which examples should be classified as "intruder".
Some sentences will have words highlighted with <> tags. Do not overthink.

There is only ever one intruder in the examples.

You should write [RESPONSE]: followed by the index of the intruder.

"""

# https://www.neuronpedia.org/gpt2-small/6-res-jb/6048
DSCORER_EXAMPLE_ONE = """Examples:

Example 0:<|endoftext|>Getty ImagesĊĊPatriots tight end Rob Gronkowski had his bossâĢĻ
Example 1: Media Day 2015ĊĊLSU defensive end Isaiah Washington (94) speaks to the
Example 2: shown, is generally not eligible for ads. For example, videos about recent tragedies,
Example 3: line, with the left side âĢĶ namely tackle Byron Bell at tackle and guard Amini
"""
DSCORER_RESPONSE_ONE_COT = "There are 3 examples that are related to American football. The intruder is 2 because it is about videos and ads."
DSCORER_RESPONSE_ONE = "[RESPONSE]: 2"


# https://www.neuronpedia.org/gpt2-small/8-res-jb/12654
DSCORER_EXAMPLE_TWO = """Examples:

Example 0: enact an individual health insurance mandate?âĢĿ, Pelosi's response was to dismiss both
Example 1: climate, TomblinâĢĻs Chief of Staff Charlie Lorensen said.Ċ
Example 2: no wonderworking relics, no true Body and Blood of Christ, no true Baptism
Example 3:ĊĊIt has been devised by Director of Public Prosecutions (DPP)
Example 4: and fair investigation not even include the Director of Athletics? Â· Finally, we believe the
"""
DSCORER_RESPONSE_TWO_COT = "I can see that there are several examples that have the word 'of' before a capital letter. The intruder is 0 because it does not."
DSCORER_RESPONSE_TWO = "[RESPONSE]: 0"

# https://www.neuronpedia.org/gpt2-small/8-res-jb/12654
DSCORER_EXAMPLE_THREE = """Examples:

Example 0: Climbing
Example 1: running
Example 2: swim
Example 3: eating
Example 4: cycling
"""
DSCORER_RESPONSE_THREE_COT = "All examples are related to activities, the first 3 and the last one being about sports and physical activities. Eating is not a sport or physical activity so it is the intruder."

DSCORER_RESPONSE_THREE = "[RESPONSE]: 3"

DSCORER_EXAMPLE_FOUR = """
Example 0: You<< guys>> support me in many other ways already and
Example 1: birth control access but I assure that << guys>> in Kentucky
Example 2: gig! I hope you <<guys>> LOVE her, and please be nice,
Example 3:American, told Hannity that you<< guys>> are playing the race card.
"""

DSCORER_RESPONSE_FOUR_COT = "I see that all the examples have the word 'guys' highlighted. All examples except example 1 have the word 'you' before 'guys', therefore, example 1 is the intruder."

DSCORER_RESPONSE_FOUR = "[RESPONSE]: 1"

GENERATION_PROMPT = """Examples:

{examples}
"""

default = [
    {"role": "user", "content": DSCORER_EXAMPLE_ONE},
    {"role": "assistant", "content": DSCORER_RESPONSE_ONE},
    {"role": "user", "content": DSCORER_EXAMPLE_TWO},
    {"role": "assistant", "content": DSCORER_RESPONSE_TWO},
    {"role": "user", "content": DSCORER_EXAMPLE_THREE},
    {"role": "assistant", "content": DSCORER_RESPONSE_THREE},
]

default_cot = [
    {"role": "user", "content": DSCORER_EXAMPLE_ONE},
    {"role": "assistant", "content": DSCORER_RESPONSE_ONE_COT + DSCORER_RESPONSE_ONE},
    {"role": "user", "content": DSCORER_EXAMPLE_TWO},
    {"role": "assistant", "content": DSCORER_RESPONSE_TWO_COT + DSCORER_RESPONSE_TWO},
    {"role": "user", "content": DSCORER_EXAMPLE_THREE},
    {
        "role": "assistant",
        "content": DSCORER_RESPONSE_THREE_COT + DSCORER_RESPONSE_THREE,
    },
]


def prompt(examples: str, cot: bool = False) -> list[dict]:
    generation_prompt = GENERATION_PROMPT.format(examples=examples)

    if cot:
        examples_to_use = default_cot
    else:
        examples_to_use = default

    prompt = [
        {"role": "system", "content": DSCORER_SYSTEM_PROMPT},
        *examples_to_use,
        {"role": "user", "content": generation_prompt},
    ]

    return prompt
