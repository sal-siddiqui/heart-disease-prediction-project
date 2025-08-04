def get_prediction(estimators, features):
    if features.shape != (1, 13):
        raise ValueError(f"Expecting a single row with 13 features. Got {features.shape} instead of (1, 13)")

    result = {}

    for name, estimator in estimators.items():
        prediction = estimator.predict(features).item()
        result[name] = prediction

    return result


def get_predictions(estimators, features):
    if features.ndim != 2 or features.shape[1] != 13 or features.shape[0] == 1:
        raise ValueError(f"Expecting atleast two rows with 13 features. Got {features.shape} instead of (* > 1, 13)")

    results = {}

    for idx, row in features.iterrows():
        row = row.to_frame().T
        results[idx] = get_prediction(estimators, row)

    return results
