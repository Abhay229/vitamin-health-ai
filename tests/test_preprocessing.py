import pandas as pd

from src.preprocessing import TARGET_COLUMN, build_preprocessor, split_features_target


def test_target_derived_columns_are_never_features():
    data = pd.DataFrame(
        {
            "Age": [20, 30],
            "Gender": [0, 1],
            TARGET_COLUMN: [18.0, 32.0],
            "VitaminD_Category": ["Deficient", "Sufficient"],
            "Deficiency_Status": [1, 0],
            "Risk_Score": [90, 10],
        }
    )

    features, target = split_features_target(data)

    assert features.columns.tolist() == ["Age", "Gender"]
    assert target.tolist() == [18.0, 32.0]


def test_preprocessor_handles_unseen_categories():
    train = pd.DataFrame({"Age": [20, None], "Gender": ["F", "M"]})
    transformed = build_preprocessor(train).fit_transform(train)
    assert transformed.shape[0] == 2
