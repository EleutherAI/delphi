import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from vllm import LLM, SamplingParams


def parse_neuronpedia_features(neuronpedia_file: str) -> Dict[str, dict]:
    """
    Parse Neuronpedia graph JSON and extract features.

    Returns dict mapping layer_feature -> {explanation, raw_data}
    """
    features = {}

    with open(neuronpedia_file, "r") as f:
        data = json.load(f)

    for node in data.get("nodes", []):
        node_id = node.get("node_id", "")
        clerp = node.get("clerp", "")

        # Extract layer_feature by removing ctx_idx (everything after 2nd underscore)
        parts = node_id.split("_")
        if len(parts) >= 2:
            feature_id = f"{parts[0]}_{parts[1]}"
            features[feature_id] = {
                "explanation": clerp,
                "layer": parts[0],
                "feature": parts[1],
                "raw_data": node,
            }

    return features


def parse_delphi_features(delphi_file: str) -> Dict[str, dict]:
    """
    Parse Delphi JSONL explanations and extract features.

    Returns dict mapping layer_feature -> {explanation, raw_data}
    """
    features = {}

    with open(delphi_file, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line.strip())

                feature_field = entry.get("feature", "")
                output_field = entry.get("explanation", "")

                # Parse feature field like "layers.0.mlp_411"
                # Extract layer and feature number
                if "layers." in feature_field and ".mlp_" in feature_field:
                    # Extract layer number from "layers.X.mlp_Y"
                    match = re.match(r"layers\.(\d+)\.mlp_(\d+)", feature_field)
                    if match:
                        layer = match.group(1)
                        feature = match.group(2)
                        feature_id = f"{layer}_{feature}"

                        explanation = output_field.strip()

                        features[feature_id] = {
                            "explanation": explanation,
                            "layer": layer,
                            "feature": feature,
                            "raw_data": entry,
                        }

    return features


def setup_vllm_client(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> Optional[LLM]:
    """
    Set up VLLM client for explanation comparison.
    """
    try:
        print(f"Loading model {model_name}...")
        llm = LLM(
            model=model_name,
            max_model_len=1024,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
        )
        return llm
    except Exception as e:
        print(f"Failed to load VLLM model: {e}")
        return None


def parse_rationale_and_score(response: str) -> tuple[str, int]:
    """
    Parse both rationale and similarity score from LLM response.
    Returns tuple of (rationale, score)
    """
    import re

    # Try to extract rationale and score using the expected format
    rationale_pattern = r"rationale:\s*(.+?)(?=score:|$)"
    score_pattern = r"score:\s*([012])"

    rationale = "DNE"
    score = -1  # Default score

    # Extract rationale
    rationale_match = re.search(rationale_pattern, response, re.IGNORECASE | re.DOTALL)
    if rationale_match:
        rationale = rationale_match.group(1).strip()
        # Clean up rationale (remove extra whitespace, newlines)
        rationale = " ".join(rationale.split())

    # Extract score
    score_match = re.search(score_pattern, response, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))

    return rationale, score


def create_comparison_prompts(common_features: List[Dict]) -> List[str]:
    """
    Create comparison prompts for all features.
    """
    prompts = []
    for feature in common_features:
        prompt = f"""
### Task
You will be given two descriptions about some set of words.
Compare how similar Explanation B is to Explanation A (the ground truth)
You will use a scale from 0 to 2 for how similar they are:
0 = Not related at all
1 = Weakly related
2 = Very closely related

Be strict and sparingly give scores of 2
### Output format
First provide a one-sentence rationale for your rating, then provide the numeric score.
Rationale:
[your reasoning]
Score: [number]

### Examples
A: proper nouns, especially political figures
B: Names of notable people and places
Rationale:
B talks about proper nouns but does not mention political figures in particular.
Score: 1

A: of
B: Possessive or relational prepositions
Rationale:
B is only barely related to A but is very broad.
Score: 1

A: verbs
B: Verbs indicating actions or states
Rationale:
Both talk about verbs.
Score: 2

A: the phrase "we all know" and similar constructions
B: Knowing or understanding facts about sports
Rationale:
B does not mention the phrase "we all know" or mention expressions at all.
Score: 0

### Real query
Using the above information as guidance, answer for the following pair:
A: {feature['neuronpedia_explanation'].strip()}
B: {feature['delphi_explanation'].strip()}
"""
        prompts.append(prompt)
    return prompts


def process_explanation_comparisons(
    llm: LLM, common_features: List[Dict]
) -> Optional[Dict]:
    """
    Process all common features through LLM for similarity scoring using batch
    processing.
    """
    if not llm:
        print("LLM failed to initialize")
        return None

    print(
        f"Comparing {len(common_features)} feature explanations with LLM "
        f"(batch processing)..."
    )

    # Create all prompts at once
    prompts = create_comparison_prompts(common_features)

    # Set up sampling parameters
    sampling_params = SamplingParams(temperature=0.1, max_tokens=50, top_p=0.9)

    try:
        # Process all prompts in one batch - let VLLM handle the batching
        print("Generating LLM responses...")
        outputs = llm.generate(prompts, sampling_params)

        # Parse all responses
        individual_scores = []
        total_score = 0
        valid = 0

        for feature, output in zip(common_features, outputs):
            response = output.outputs[0].text.strip()
            rationale, score = parse_rationale_and_score(response)

            individual_scores.append(
                {
                    "layer_feature_id": feature["layer_feature_id"],
                    "score": score,
                    "rationale": rationale,
                    "neuronpedia_explanation": feature["neuronpedia_explanation"],
                    "delphi_explanation": feature["delphi_explanation"],
                }
            )
            if score != -1:
                total_score += score
                valid += 1

        max_possible = valid * 2
        average_score = total_score / max_possible if valid > 0 else 0.0

        print(f"Completed batch processing of {len(common_features)} comparisons")

        return {
            "total_score": total_score,
            "max_possible": max_possible,
            "average_score": average_score,
            "individual_scores": individual_scores,
        }

    except Exception as e:
        print(f"Error during batch LLM comparison: {e}")
        return None


def compare_features(
    neuronpedia_features: Dict[str, dict], delphi_features: Dict[str, dict]
) -> Dict:
    """
    Compare features from both sources and create comparison data.
    """
    neuronpedia_ids = set(neuronpedia_features.keys())
    delphi_ids = set(delphi_features.keys())

    common_ids = neuronpedia_ids & delphi_ids
    only_neuronpedia = neuronpedia_ids - delphi_ids
    only_delphi = delphi_ids - neuronpedia_ids

    # Create comparison data
    common_features = []
    for feature_id in sorted(common_ids):
        common_features.append(
            {
                "layer_feature_id": feature_id,
                "layer": neuronpedia_features[feature_id]["layer"],
                "feature": neuronpedia_features[feature_id]["feature"],
                "neuronpedia_explanation": neuronpedia_features[feature_id][
                    "explanation"
                ],
                "delphi_explanation": delphi_features[feature_id]["explanation"],
            }
        )

    comparison_data = {
        "summary": {
            "total_neuronpedia_features": len(neuronpedia_ids),
            "total_delphi_features": len(delphi_ids),
            "common_features": len(common_ids),
            "only_in_neuronpedia": len(only_neuronpedia),
            "only_in_delphi": len(only_delphi),
        },
        "common_features": common_features,
        "only_in_neuronpedia": sorted(list(only_neuronpedia)),
        "only_in_delphi": sorted(list(only_delphi)),
    }

    return comparison_data


def create_html_table(comparison_data: Dict, output_file: str):
    """
    Create interactive HTML table with row selection and view toggle.
    """
    import json
    import random

    # Create a shuffled copy of the common features
    shuffled_features = comparison_data["common_features"].copy()
    random.shuffle(shuffled_features)

    # Convert features to JSON for JavaScript
    features_json = json.dumps(shuffled_features)

    # Extract summary values for shorter lines
    summary = comparison_data["summary"]
    total_np = summary["total_neuronpedia_features"]
    total_delphi = summary["total_delphi_features"]
    common_features = summary["common_features"]
    only_np = summary["only_in_neuronpedia"]
    only_delphi = summary["only_in_delphi"]

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Feature Comparison: Neuronpedia vs Delphi</title>
    <style>
        body {{
            font-family: Arial;
            font-size: 14pt;
            margin: 20px;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
            padding: 0 40px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 30px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
            vertical-align: top;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .summary {{
            background-color: #e8f4f8;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        .section-header {{
            font-size: 24pt;
            font-weight: bold;
            color: #333;
            margin: 30px 0 15px 0;
        }}
        .unique-list {{
            background-color: #fff8dc;
            padding: 15px;
            border-radius: 5px;
        }}
        .feature-row {{
            cursor: pointer;
            transition: background-color 0.2s;
        }}
        .feature-row:hover {{
            background-color: #f0f8ff;
        }}
        .feature-row.selected {{
            background-color: #90EE90 !important;
        }}
        .controls {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 8px;
        }}
        .toggle-button {{
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 18pt;
            margin-right: 20px;
        }}
        .toggle-button:hover {{
            background-color: #45a049;
        }}
        .selection-counter {{
            display: inline-block;
            font-weight: bold;
            color: #333;
        }}
    </style>
</head>
<body>
    <h1>Feature Comparison: Neuronpedia vs Delphi</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total Neuronpedia Features:</strong> {total_np}</p>
        <p><strong>Total Delphi Features:</strong> {total_delphi}</p>
        <p><strong>Common Features:</strong> {common_features}</p>
        <p><strong>Only in Neuronpedia:</strong> {only_np}</p>
        <p><strong>Only in Delphi:</strong> {only_delphi}</p>"""

    # Add comparison scores if available
    if "llm_comparison" in comparison_data:
        comp = comparison_data["llm_comparison"]
        html_content += f"""
        <p><strong>LLM Similarity Score:</strong>
        {comp['total_score']}/{comp['max_possible']}
        (avg: {comp['average_score']:.2f})</p>
        """

    html_content += f"""
        <p class="selection-counter"><strong>
            Selected Rows:</strong> <span id="selectedCount">0</span>
        </p>
    </div>

    <div class="controls">
        <button id="viewToggle" class="toggle-button" onclick="toggleView()">
        Show Random 100
        </button>
        <span class="selection-counter">
            Currently showing: <span id="currentView">
                All {common_features} features
            </span>
        </span>
    </div>

    <div class="section-header">Common Features</div>
    <table id="featuresTable">
        <thead>
            <tr>
                <th>Layer</th>
                <th>Feature</th>
                <th>Neuronpedia Explanation</th>
                <th>Delphi Explanation</th>
                <th>LLM Score</th>
                <th>LLM Rationale</th>
            </tr>
        </thead>
        <tbody id="featuresTableBody">
        </tbody>
    </table>

    <script>
        // Global variables
        let allFeatures = {features_json};
        let currentFeatures = [...allFeatures];
        let selectedRows = new Set();
        let showingAll = true;
        let randomSubset = [];

        // Initialize the table
        function initializeTable() {{
            displayFeatures(currentFeatures);
            updateSelectionCounter();
        }}

        // Display features in the table
        function displayFeatures(features) {{
            const tbody = document.getElementById('featuresTableBody');
            tbody.innerHTML = '';

            features.forEach((feature, index) => {{
                const row = document.createElement('tr');
                row.className = 'feature-row';
                row.dataset.featureId = feature.layer_feature_id;
                row.onclick = () => toggleRowSelection(row);

                // Check if this row was previously selected
                if (selectedRows.has(feature.layer_feature_id)) {{
                    row.classList.add('selected');
                }}

                row.innerHTML = `
                    <td>${{feature.layer}}</td>
                    <td>${{feature.feature}}</td>
                    <td>${{feature.neuronpedia_explanation}}</td>
                    <td>${{feature.delphi_explanation}}</td>
                    <td>${{feature.llm_score !== null ?
                        feature.llm_score : 'N/A'}}</td>
                    <td>${{feature.llm_rationale !== null ?
                        feature.llm_rationale : 'N/A'}}</td>
                `;

                tbody.appendChild(row);
            }});
        }}

        // Toggle row selection
        function toggleRowSelection(row) {{
            const featureId = row.dataset.featureId;

            if (row.classList.contains('selected')) {{
                row.classList.remove('selected');
                selectedRows.delete(featureId);
            }} else {{
                row.classList.add('selected');
                selectedRows.add(featureId);
            }}

            updateSelectionCounter();
        }}

        // Update selection counter
        function updateSelectionCounter() {{
            document.getElementById('selectedCount').textContent = selectedRows.size;
        }}

        // Get random subset of features
        function getRandomSubset(features, count) {{
            const shuffled = [...features].sort(() => 0.5 - Math.random());
            return shuffled.slice(0, Math.min(count, features.length));
        }}

        // Toggle between all features and random 100
        function toggleView() {{
            const toggleButton = document.getElementById('viewToggle');
            const currentViewSpan = document.getElementById('currentView');

            if (showingAll) {{
                // Switch to random 100
                if (randomSubset.length === 0) {{
                    randomSubset = getRandomSubset(allFeatures, 100);
                }}
                currentFeatures = randomSubset;
                toggleButton.textContent = 'Show All Features';
                currentViewSpan.textContent = `Random 100 features`;
                showingAll = false;
            }} else {{
                // Switch to all features
                currentFeatures = allFeatures;
                toggleButton.textContent = 'Show Random 100';
                currentViewSpan.textContent = `All ${{allFeatures.length}} features`;
                showingAll = true;
            }}

            displayFeatures(currentFeatures);
        }}

        // Initialize when page loads
        window.onload = initializeTable;
    </script>
"""

    html_content += """

    <div class="section-header">Features Only in Neuronpedia</div>
    <div class="unique-list">
"""

    if comparison_data["only_in_neuronpedia"]:
        html_content += (
            "<p>" + ", ".join(comparison_data["only_in_neuronpedia"]) + "</p>"
        )
    else:
        html_content += "<p>None</p>"

    html_content += """
    </div>

    <div class="section-header">Features Only in Delphi</div>
    <div class="unique-list">
"""

    if comparison_data["only_in_delphi"]:
        html_content += "<p>" + ", ".join(comparison_data["only_in_delphi"]) + "</p>"
    else:
        html_content += "<p>None</p>"

    html_content += """
    </div>

</body>
</html>
"""

    with open(output_file, "w") as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Neuronpedia and Delphi feature explanations"
    )
    parser.add_argument("neuronpedia_file", help="Path to Neuronpedia graph JSON file")
    parser.add_argument("delphi_file", help="Path to Delphi explanations JSONL file")
    parser.add_argument("--output", help="Output directory path (default: results)")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Use VLLM to compare explanation similarity (requires VLLM)",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-3-12b-it",
        help="Model to use for comparison (default: meta-llama/Llama-3.1-8B-Instruct)",
    )

    args = parser.parse_args()

    print("Parsing Neuronpedia features...")
    neuronpedia_features = parse_neuronpedia_features(args.neuronpedia_file)
    print(f"Found {len(neuronpedia_features)} Neuronpedia features")

    print("Parsing Delphi features...")
    delphi_features = parse_delphi_features(args.delphi_file)
    print(f"Found {len(delphi_features)} Delphi features")

    print("Comparing features...")
    comparison_data = compare_features(neuronpedia_features, delphi_features)

    # Add LLM comparison if requested
    if args.compare:
        print("Setting up VLLM for explanation comparison...")
        llm = setup_vllm_client(args.model)
        llm_results = process_explanation_comparisons(
            llm, comparison_data["common_features"]
        )
        if llm_results:
            comparison_data["llm_comparison"] = llm_results

            # Merge LLM scores and rationales back into common_features
            score_lookup = {
                item["layer_feature_id"]: item
                for item in llm_results["individual_scores"]
            }
            for feature in comparison_data["common_features"]:
                feature_id = feature["layer_feature_id"]
                if feature_id in score_lookup:
                    feature["llm_score"] = score_lookup[feature_id]["score"]
                    feature["llm_rationale"] = score_lookup[feature_id]["rationale"]
                else:
                    feature["llm_score"] = None
                    feature["llm_rationale"] = None

            print("\nLLM Comparison Results:")
            print(
                f"  Total Score: {llm_results['total_score']}/"
                f"{llm_results['max_possible']}"
            )
            print(f"  Average Score: {llm_results['average_score']:.2f}")

    print("Saving results...")
    # Create output directory if it doesn't exist
    output_dir = Path(args.delphi_file).parent
    if args.output:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON output
    with open(f"{output_dir}/comparison_results.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    # Create HTML table
    create_html_table(comparison_data, f"{output_dir}/comparison_table.html")

    print("\nResults saved to:")
    print(f"  JSON: {output_dir}/comparison_results.json")
    print(f"  HTML: {output_dir}/comparison_table.html")

    print("\nSummary:")
    print(f"  Common features: {comparison_data['summary']['common_features']}")
    print(f"  Only in Neuronpedia: {comparison_data['summary']['only_in_neuronpedia']}")
    print(f"  Only in Delphi: {comparison_data['summary']['only_in_delphi']}")

    if "llm_comparison" in comparison_data:
        comp = comparison_data["llm_comparison"]
        print(
            f"  LLM Similarity Score: {comp['total_score']}/"
            f"{comp['max_possible']} (avg: {comp['average_score']:.2f})"
        )


if __name__ == "__main__":
    main()
