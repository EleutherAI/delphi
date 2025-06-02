SYSTEM_PROMPT = """
You are an interpretability‑labeling assistant.

INPUT
• One prompt block with examples from a single latent feature from a sparse auto‑encoder. Activating tokens are marked like <token|score>; all other tokens are plain text.
• You will also receive non-activating examples to help narrow down the concept. These examples are not highlighted.

TASK
1. **Read all information in the block** (highlighted tokens, examples).
2. **Infer the concept the latent detects** – e.g. a grammatical pattern, semantic theme, named entity, punctuation, etc.
   Focus on what *causes* large activations, but also consider the context of the tokens. Sometimes the context is more important than the tokens themselves.
3. **Output exactly three lines** (no extra words, bullets, or explanations):

   • **Line 1 – Concept description:** One clear sentence in plain English that best describes the latent’s pattern.
     – Be specific: mention relationships, tense, or semantic role if relevant.
     – If nothing coherent emerges, write “uncertain”.

   • **Line 2 – Short name:** A concise `snake_case` identifier (3–6 lowercase tokens) capturing the essence of the concept.
     – Use only letters, digits, and underscores.
     – If uncertain, write `unknown`.

   • **Line 3 – Confidence:** A label denoting how confident you are in this description
     - You can choose from: very high, high, medium high, medium, medium low, low, and very low

STYLE & RULES
• Do **not** quote the examples verbatim. Paraphrase.
• Return **only** the three lines—no markdown, no JSON, no headings.
"""

FEW_SHOT_EXAMPLE_1 = """
ACTIVATING EXAMPLES:
Example 0:  coastal communities on how to adapt<< to|9.0>> risings seas: move away from the shore. The Post report explores the implications of sea-level rise for the Hampton Roads
Example 1:  to adjust<< to|9.0>> an ostomy. But you will be able to
work, participate in sports and physical activities, be intimate with your
partner, and
Example 2:  .
Getting to Know the Place
The process of adjusting<< to|10.0>> life following repatriation was extremely difficult during the first year. Returnees sold whatever assets they
Example 3:  whether it be demining, helping youngsters born in camps to adapt<< to|8.0>> rural areas, setting up micro-credit systems, or similar programs. Provisions also need
Example 4:  .
Tall fescue has a lot of favorable traits. It is persistent, adapted<< to a|6.0,2.0>> variety of soil conditions, is compatible in mixtures with other grasses
Example 5:  code requirements change over time (as code organizations adapt<< to|7.0>> new information and technologies), buildings are usually not required to modify their structure or operation to conform to the new
Example 6:  your average house cat.
Over thousands of years, we’ve learned to cope<< with|4.0>> our night blindness, mainly by using artificial light to make up for the
Example 7:  (Marquardt, 1996) and on sustaining momentum and adapting<< to|7.0>> change in the corporate marketplace (Senge, 199
Example 8:  medical library at the University of Wisconsin at Madison.
Helen's biggest professional challenge at Wisconsin was coping<< with|3.0>> lack of space. After several plans for expansion failed
Example 9:  art tools and the open-source feature means it can be adapted<< to|3.0>> new problems," Santiago said.
That 75-fold faster performance, enabled by
Example 10:  economy. In fact, I have been critical of many aspects of the Japanese economy. But they do seem to have made a better adjustment than Western nations<< to|3.0>> the

NON-ACTIVATING EXAMPLES:
Example 0:  to survive change.definition of environmental attributes. However, some also include a reference to the type of environmental attribute of primary interest, such as a particular state
Example 1:  quarter of a mile. The quarter master stood aft, near the taffrail, and kept her constantly astern, by which means they were enabled to steer
Example 2:  Lubbock, Texas, used a meta-analysis of heat and water-deficit stress expression profiles in cotton, peanut, and Arabidopsis to identify candidate genes
Example 3:  wood was used for comparisons with Jurassic fossil forests at Kawhia Harbour, New Zealand.Orphanages on Trial: Hundreds of thousands of children in Russia are
Example 4:  though to symbolize that God was again at peace with the earth (see Genesis 8:11). . . . There is further symbolic significance in the cultivation
Example 5:  are offensive or obvious attempts at advertising, please submit the information below.|Date:||7/29/2014 6:27
"""

FEW_SHOT_RESPONSE_1 = """
Concept description: The latent detects the use of the verb "to adapt" or related words, often with the preposition "to" or "for", indicating a process of change or adjustment.
Short name: adapt_to
Confidence: high
"""

FEW_SHOT_EXAMPLE_2 = """
ACTIVATING EXAMPLES:
Example 0:  juice,<< wine|9.0>>, tea, candy, and
even ice cream. The berries are also known as "Siberian pineapple"
Some folks (in New England
Example 1:  a simple color-change test. I’ve paraphrased this from a<< wine|8.0>> text: Titration is the process of determining the concentration of a substance
Example 2:  and the conservation of soil and water. Organic<< wine|6.0>>growing recognizes that healthy soils grow balanced<< grapes|4.0>> that produce the best<< wines|6.0>>. Organic<< vineyards|3.0>> are often small, family-
Example 3:  We want to make sure everyone who works for our company, whether in IT, marketing or finance, has knowledge about<< wine|7.0>> and a passion about<< wine|6.0>>.”
Trans
Example 4:  thou be destroyed: which also shall not leave thee either corn,<< wine|6.0>>, or oil, or the increase of thy kine, or flocks of thy sheep,
Example 7:  . From what I’ve read, it seems that citric acid is often used by home<< winem|4.0,7.0>>akers who want to add a bit more kick to their
Example 8:  Busch registered its trademark in 1876 , after the<< Bud|2.0>>weiser had been<< brewed|4.0>> there, but before the particular Budvar<< brew|5.0>>ery was incorporated
Example 9:  All foods made with raw or lightly cooked eggs
- Unpasteurized milk products and foods made from them
- Soft and semi-soft<< cheeses|5.0>> like Feta
Example 10:  1900. The thief is sitting on a horse below the "Ice<< Beer|4.0>>" sign, with his hands behind his back.
A posse that
Example 11:  I got a drink<< whiskey|4.0>> from Adj. of Gen. Schutz afternoon on fatigue on gurilla. For supper fishes. On guard from 1
Example 12:  is immense,"<< brew|4.0>>ers association head Walter Koenig said. "The increasing costs of raw materials may become a serious threat for many<< brewer|4.0>>ies."
We
Example 13:  , as well as the various stages of the disease.
- 10 Warning Signs – The ten basic warning signs that someone you<< love|4.0>> may have Alzheimer’
Example 14:  rootes of diuers kindes, and diuers fruites: their drinke is commonly water, but while the<< grape|3.0>> lasteth, they
Example 15:  of<< liquor|3.0>> for<< beverage|2.0>> purposes. It did provide however, for the sale at wholesale, by persons known as "Vendors" to physicians, chemists, dru
Example 16:  in same-sex flirting lure in female fish;<< cheese|3.0>> dates back 7.5 millennia; Americans love public transportation once they give it a chance;
Example 17:  time, Coke was sold at soda fountains. But the lawyers were interested in this new idea: selling<< drinks|3.0>> in bottles. The lawyers wanted to buy the<< bott|3.0>>ling
Example 18:  ; for example, the sticky remnants within<< drinks|3.0>> cans is appealing to mice (harvest, wood and others) which crawl in and become trapped and must suffer an agon
Example 19:  << bottled|3.0>> water, when linked with a WBDO, was classified as an individual water system; these WBDOs are now classified as<< bottled|2.0>> water. WN
Example 20:  nineteenth century to produce the story of a<< drink|3.0>> that came to symbolize both the high points of art and the depths of degeneration.
Jad Adams looks at the

NON-ACTIVATING EXAMPLES:
Example 0:  .The term herb-chronology is referring to dendrochronology because of the similarity of the structures investigated. The term was introduced in the late
Example 1:  third example is a nonlinear ordinary differential that we analyze using its two symmetries. The final example is the partial differential equation known as the cubic nonlinear Schrö
Example 2:  an active open source community is growing and providing more free drivers and software libraries for linux.This talk aims to introduce the audience to virtual reality development on
Example 3:  and her daughters, Eudocia and Placidia. The mother and Placidia were sent to Constantinople, later, under Leo I, but G
Example 4:  Since the symptoms of histamine intolerance look like an allergy but aren’t, researchers tend to describe the problem as a “pseudoallergy:” all the
Example 5:  passed, the Martello towers in England met a variety of fates. The Coastguard took over many to aid in the fight against smuggling.Fifteen
Example 6:  01. And the White House had its first hydraulic elevator installed in 1881, its first electric elevator in 1898.
Example 7:  own ethnic group, country, region, class, or whatever. That’s the sort of history that one “owns.”But there’s an equally
Example 8:  I attended an astrophysics talk last night. The guest speaker said that it was the ‘pattern’ of the CMBR (cosmic microwave background radiation) that was
Example 9:  and connect the card reader to a USB port.Launch the file recovery software on your computer. Many programs have this feature including Norton 360
Example 10:  informed, effective, and responsible participants in the profession and society. It is also an education that is designed to challenge the intellect of the student and help him/
"""

FEW_SHOT_RESPONSE_2 = """
Concept description: The latent identifies words related to alcoholic beverages, drinks and some types of food eaten with these beverages.
Short name: alchol_drink_food_related
Confidence: medium
"""


def prompt(examples: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_EXAMPLE_1},
        {"role": "assistant", "content": FEW_SHOT_RESPONSE_1},
        {"role": "user", "content": FEW_SHOT_EXAMPLE_2},
        {"role": "assistant", "content": FEW_SHOT_RESPONSE_2},
        {"role": "user", "content": examples},
    ]
