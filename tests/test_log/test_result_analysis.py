import numpy as np
import pandas as pd

from delphi.log.result_analysis import compute_confusion


def test_failed_count_counts_dropped_predictions():
    """
    compute_confusion must report how many examples the scorer failed to
    produce a (parseable) prediction for. Failed examples are stored with a
    None/NaN ``prediction`` by the classifier and are dropped by the
    ``prediction.notna()`` filter, so ``failed_count`` should equal the number
    of rows removed by that filter, i.e. ``len(df) - len(valid rows)``.
    """
    df = pd.DataFrame(
        {
            "activating": [True, False, True, False],
            # Two examples failed to be parsed -> NaN prediction.
            "prediction": [1.0, 0.0, np.nan, np.nan],
            "probability": [0.9, 0.1, np.nan, np.nan],
        }
    )

    conf = compute_confusion(df)

    assert conf["total_examples"] == 2
    assert conf["failed_count"] == 2

    # Mirror the "fraction of failed examples" computation in log_results().
    fraction_failed = conf["failed_count"] / (
        conf["total_examples"] + conf["failed_count"]
    )
    assert fraction_failed == 0.5


def test_failed_count_zero_when_all_predictions_valid():
    df = pd.DataFrame(
        {
            "activating": [True, False],
            "prediction": [1.0, 0.0],
            "probability": [0.9, 0.1],
        }
    )

    conf = compute_confusion(df)

    assert conf["total_examples"] == 2
    assert conf["failed_count"] == 0
