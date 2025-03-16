### SYSTEM PROMPT ###

SYSTEM_SINGLE_TOKEN = """Your job is to look for patterns in text. You will be given a list of WORDS, your task is to provide an explanation for what pattern best describes them. Here are some guidelines:
- Produce a specific final description for the latents common in the examples, and what patterns you found.
- Don't focus on giving examples of important tokens, if the examples are uninformative, you don't need to mention them.
- Do not make lists of possible explanations. Keep your explanations short and concise.
- The last line of your response must be the formatted explanation, using [EXPLANATION]:

Here are some example:

WORDS: ['Thomas Edison', 'Steve Jobs', 'Alexander Graham Bell']
[EXPLANATION]: Names of people who are inventors of technical fields

WORDS: ['over the moon', 'till the cows come home', 'than meets the eye']
[EXPLANATION]: Common idioms in text conveying positive sentiment.

WORDS: ['er', 'er', 'er']
[EXPLANATION]: The token "er".

WORDS: ['house', 'a box', 'smoking area', 'way']
[EXPLANATION]: Nouns representing a distinct objects that contains something, sometimes preciding a quotation mark.

{prompt}
"""

SYSTEM = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>. How important each token is for the behavior is listed after each example in parentheses.

- Try to produce a concise final description. Simply describe the text latents that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not mention the marker tokens (<< >>) in your explanation.
- Do not make lists of possible explanations. Keep your explanations short and concise.
- The last line of your response must be the formatted explanation, using [EXPLANATION]:

{prompt}
"""

SYSTEM_CONTRASTIVE = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>. How important each token is for the behavior is listed after each example in parentheses.

- Try to produce a concise final description. Simply describe the text latents that are common in the examples, and what patterns you found.
- Counterexamples where no special words are present are also provided to help you understand the patterns' edge cases.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not mention the marker tokens (<< >>) in your explanation.
- Do not make lists of possible explanations. Keep your explanations short and concise.
- The last line of your response must be the formatted explanation, using [EXPLANATION]:
"""


COT = """
To better find the explanation for the language patterns go through the following stages:

1.Find the special words that are selected in the examples and list a couple of them. Search for patterns in these words, if there are any. Don't list more than 5 words.

2. Write down general shared latents of the text examples. This could be related to the full sentence or to the words surrounding the marked words.

3. Formulate an hypothesis and write down the final explanation using [EXPLANATION]:.

"""


### EXAMPLE 1 ###

EXAMPLE_1 = """
Example 1:  and he was <<over the moon>> to find
Example 2:  we'll be laughing <<till the cows come home>>! Pro
Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd
"""

EXAMPLE_1_ACTIVATIONS = """
Example 1:  and he was <<over the moon>> to find
Activations: ("over", 5), (" the", 6), (" moon", 9)
Example 2:  we'll be laughing <<till the cows come home>>! Pro
Activations: ("till", 5), (" the", 5), (" cows", 8), (" come", 8), (" home", 8)
Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd
Activations: ("than", 5), (" meets", 7), (" the", 6), (" eye", 8)
"""


EXAMPLE_1_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "over the moon", "than meets the eye".
SURROUNDING TOKENS: No interesting patterns.

Step 1.
- The activating tokens are all parts of common idioms.
- The surrounding tokens have nothing in common.

Step 2.
- The examples contain common idioms.
- In some examples, the activating tokens are followed by an exclamation mark.

Step 3.
- The activation values are the highest for the more common idioms in examples 1 and 3.

Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities?
- Yes, I missed one: The text examples all convey positive sentiment.
"""

EXAMPLE_1_EXPLANATION = """
[EXPLANATION]: Common idioms in text conveying positive sentiment.
"""

### EXAMPLE 2 ###

EXAMPLE_2 = """
Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
Example 2:  every year you get tall<<er>>," she
Example 3:  the hole was small<<er>> but deep<<er>> than the
"""

EXAMPLE_2_ACTIVATIONS = """
Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
Activations: ("er", 8)
Example 2:  every year you get tall<<er>>," she
Activations: ("er", 2)
Example 3:  the hole was small<<er>> but deep<<er>> than the
Activations: ("er", 9), ("er", 9)
"""

EXAMPLE_2_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "er", "er", "er".
SURROUNDING TOKENS: "wid", "tall", "small", "deep".

Step 1.
- The activating tokens are mostly "er".
- The surrounding tokens are mostly adjectives, or parts of adjectives, describing size.
- The neuron seems to activate on, or near, the tokens in comparative adjectives describing size.

Step 2.
- In each example, the activating token appeared at the end of a comparative adjective.
- The comparative adjectives ("wider", "tallish", "smaller", "deeper") all describe size.

Step 3.
- Example 2 has a lower activation value. It doesn't compare sizes as directly as the other examples.

Let me look again for patterns in the examples. Are there any links or hidden linguistic commonalities that I missed?
- I can't see any.
"""


EXAMPLE_2_EXPLANATION = """
[EXPLANATION]: The token "er" at the end of a comparative adjective describing size.
"""

### EXAMPLE 3 ###

EXAMPLE_3 = """
Example 1:  something happening inside my <<house>>", he
Example 2:  presumably was always contained in <<a box>>", according
Example 3:  people were coming into the <<smoking area>>".

However he
Example 4:  Patrick: "why are you getting in the << way?>>" Later,
"""

EXAMPLE_3_ACTIVATIONS = """
Example 1:  something happening inside my <<house>>", he
Activations: ("house", 7)
Example 2:  presumably was always contained in <<a box>>", according
Activations: ("a", 5), ("box", 9)
Example 3:  people were coming into the <<smoking area>>".

However he
Activations: ("smoking", 2), ("area", 4)
Example 4:  Patrick: "why are you getting in the << way?>>" Later,
Activations: ("way", 4), ("?", 2)
"""

EXAMPLE_3_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "house", "a box", "smoking area", " way?".
SURROUNDING TOKENS: No interesting patterns.

Step 1.
- The activating tokens are all things that one can be in.
- The surrounding tokens have nothing in common.

Step 2.
- The examples involve being inside something, sometimes figuratively.
- The activating token is a thing which something else is inside of.

STEP 3.
- The activation values are highest for the examples where the token is a distinctive object or space.

Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities?
- Yes, I missed one: The activating token is followed by a quotation mark, suggesting it occurs within speech.
"""

EXAMPLE_3_EXPLANATION = """
[EXPLANATION]: Nouns representing a distinct objects that contains something, sometimes preciding a quotation mark.
"""


def get(item):
    return globals()[item]


def _prompt(n, activations=False, **kwargs):
    starter = (
        get(f"EXAMPLE_{n}") if not activations else get(f"EXAMPLE_{n}_ACTIVATIONS")
    )

    prompt_atoms = [starter]

    return "".join(prompt_atoms)


def _response(n, cot=False, **kwargs):
    response_atoms = []
    if cot:
        response_atoms.append(get(f"EXAMPLE_{n}_COT_ACTIVATION_RESPONSE"))

    response_atoms.append(get(f"EXAMPLE_{n}_EXPLANATION"))

    return "".join(response_atoms)


def example(n, **kwargs):
    prompt = _prompt(n, **kwargs)
    response = _response(n, **kwargs)

    return prompt, response


def system(cot=False):
    prompt = ""

    if cot:
        prompt += COT

    return [
        {
            "role": "system",
            "content": SYSTEM.format(prompt=prompt),
        }
    ]


def system_single_token():
    return [{"role": "system", "content": SYSTEM_SINGLE_TOKEN}]


def system_contrastive():
    return [{"role": "system", "content": SYSTEM_CONTRASTIVE}]
