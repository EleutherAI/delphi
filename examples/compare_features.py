#!/usr/bin/env python3
"""
Script to compare Neuronpedia graph features with Delphi explanations.

This script takes two JSON files:
1. Neuronpedia graph JSON with nodes containing node_id, clerp explanations
2. Delphi explanations JSONL with feature IDs and explanation outputs

It finds common features and creates comparison outputs.
"""

import argparse
import json
import re
from typing import Dict


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
                output_field = entry.get("output", "")

                # Parse feature field like "layers.0.mlp_411"
                # Extract layer and feature number
                if "layers." in feature_field and ".mlp_" in feature_field:
                    # Extract layer number from "layers.X.mlp_Y"
                    match = re.match(r"layers\.(\d+)\.mlp_(\d+)", feature_field)
                    if match:
                        layer = match.group(1)
                        feature = match.group(2)
                        feature_id = f"{layer}_{feature}"

                        # Extract explanation from output
                        explanation = output_field.strip()

                        features[feature_id] = {
                            "explanation": explanation,
                            "layer": layer,
                            "feature": feature,
                            "raw_data": entry,
                        }

    return features


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

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Feature Comparison: Neuronpedia vs Delphi</title>
    <style>
        body {{
            font-family: Poppins;
            font-size: 24pt;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
            padding: 0 40px;
            margin: 20px;
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
        <p>
            <strong>Total Neuronpedia Features:</strong>
            {comparison_data["summary"]["total_neuronpedia_features"]}
        </p>
        <p><strong>Total Delphi Features:</strong>
          {comparison_data["summary"]["total_delphi_features"]}
        </p>
        <p><strong>Common Features:</strong>
            {comparison_data["summary"]["common_features"]}
        </p>
        <p><strong>Only in Neuronpedia:</strong>
            {comparison_data["summary"]["only_in_neuronpedia"]}
        </p>
        <p><strong>Only in Delphi:</strong>
            {comparison_data["summary"]["only_in_delphi"]}
        </p>
        <p class="selection-counter">
            <strong>Selected Rows:</strong> <span id="selectedCount">0</span>
            </p>
    </div>

    <div class="controls">
        <button id="viewToggle" class="toggle-button" onclick="toggleView()">
            Show Random 100
        </button>
        <span class="selection-counter">Currently showing: <span id="currentView">
            All {comparison_data["summary"]["common_features"]} features
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
    parser.add_argument(
        "--output", default="results", help="Output directory path (default: results)"
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

    print("Saving results...")
    # Save JSON output
    with open(f"{args.output}/comparison_results.json", "w") as f:
        json.dump(comparison_data, f, indent=2)

    # Create HTML table
    create_html_table(comparison_data, f"{args.output}/comparison_table.html")

    print("\nResults saved to:")
    print(f"  JSON: {args.output}/comparison_results.json")
    print(f"  HTML: {args.output}/comparison_table.html")

    print("\nSummary:")
    print(f"  Common features: {comparison_data['summary']['common_features']}")
    print(f"  Only in Neuronpedia: {comparison_data['summary']['only_in_neuronpedia']}")
    print(f"  Only in Delphi: {comparison_data['summary']['only_in_delphi']}")


if __name__ == "__main__":
    main()
