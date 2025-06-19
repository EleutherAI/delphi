class DspyFuzzer(DspyClassifier):
    name = "fuzzing"

    def __init__(
        self,
        client: Client,
        n_examples_shown: int = 5,
        seed: int = 42,
        fuzz_type: Literal["default", "active"] = "default",
    ):
        self.client = client
        self.n_examples_shown = n_examples_shown
        client = self.client.client
        dspy.configure(lm=client)
        self.scorer = (dspy.Predict)(ExampleDetector)
        self.scorer = dspy.LabeledFewShot().compile(
            self.scorer,
            trainset=FEW_SHOT,
        )
        self.rng = random.Random(seed)
        self.fuzz_type = fuzz_type
        super().__init__(self.scorer)

    def _highlight_tokens(
        self,
        tokens: list[str],
        activations: list[float],
        positions: list[tuple[int, int]] = [],
    ) -> tuple[str, list[tuple[int, int]]]:
        output = ""
        # branch if highlighting positions are provided
        if len(positions) > 0:
            starts, ends = zip(*positions)
            for i, token in enumerate(tokens):
                if i in starts:
                    output += "<"
                elif i in ends:
                    output += ">"
                else:
                    output += token
        # branch if using activations to highlight tokens
        else:
            positions = []
            i = 0
            while i < len(tokens):
                if activations[i] > 0:
                    start = i
                    output += "<"

                    while i < len(tokens) and activations[i] > 0:
                        output += tokens[i]
                        i += 1
                    end = i
                    positions.append((start, end))

                    output += ">"
                else:
                    output += tokens[i]
                    i += 1

        return output, positions

    def _perturb_positions(
        self, positions: list[tuple[int, int]], length: int
    ) -> list[tuple[int, int]]:
        new_positions = []
        for position in positions:
            max_len = length - 1

            if self.rng.random() < 0.5:
                # move one forward if possible
                if position[1] < max_len:
                    new_positions.append((position[0] + 1, position[1] + 1))
                else:
                    new_positions.append((position[0] - 1, position[1] - 1))
            else:
                # move one backward if possible
                if position[0] > 0:
                    new_positions.append((position[0] - 1, position[1] - 1))
                else:
                    new_positions.append((position[0] + 1, position[1] + 1))
            new_positions.append(position)
        return new_positions

    def _prepare_and_batch(self, record: LatentRecord) -> list[list[FeatureExample]]:
        activating_examples = record.test
        non_activating_examples = record.not_active
        all_examples = []
        activating_positions = []
        for example in activating_examples:
            # making type checking happy
            assert example.str_tokens is not None
            text, positions = self._highlight_tokens(
                example.str_tokens, example.activations.tolist()
            )
            all_examples.append(
                FeatureExample(text=text, activating=True, correct=None)
            )
            activating_positions.append(positions)
        if self.fuzz_type == "default":
            for example in non_activating_examples:
                assert example.str_tokens is not None

                if sum(example.activations.tolist()) == 0:
                    # sample one of activating positions if example is not activating
                    random_position = self.rng.choice(activating_positions)
                else:
                    # if from a neighbour, we highlight the activating example
                    random_position = []

                text, positions = self._highlight_tokens(
                    example.str_tokens, example.activations.tolist(), random_position
                )
                all_examples.append(
                    FeatureExample(text=text, activating=False, correct=None)
                )
        elif self.fuzz_type == "active":
            for example in activating_examples:
                assert example.str_tokens is not None
                random_position = self.rng.choice(activating_positions)
                # perturb the positions
                new_positions = self._perturb_positions(
                    random_position, len(example.str_tokens)
                )
                text, positions = self._highlight_tokens(
                    example.str_tokens, example.activations.tolist(), new_positions
                )
                all_examples.append(
                    FeatureExample(text=text, activating=False, correct=None)
                )

        self.rng.shuffle(all_examples)

        batched = [
            all_examples[i : i + self.n_examples_shown]
            for i in range(0, len(all_examples), self.n_examples_shown)
        ]

        return batched

    def _check_correct(self, example: FeatureExample, is_feature: bool) -> bool:
        return is_feature == example.activating
