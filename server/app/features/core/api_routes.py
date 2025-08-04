from pathlib import Path
import pickle

from flask import Blueprint, jsonify, request
import pandas as pd
from app.utils.helpers import get_prediction, get_predictions


# initialize blueprint
bp_core_api = Blueprint(
    name="core_api",
    import_name=__name__,
    template_folder="../../templates/core/",
    url_prefix="/api",
)

models = {}

for item in (Path.cwd() / "app/utils" / "classifiers").iterdir():
    name = item.stem.replace("-", "_")
    estimator = pickle.load(item.open("rb"))
    models[name] = estimator


@bp_core_api.route("/predict", methods=["POST"])
def predict():
    # Get input data as JSON
    data = request.get_json()

    # Convert data to DataFrame
    features = pd.DataFrame([data])

    # Get prediction based on the features
    predictions = get_prediction(estimators=models, features=features)

    # Return the prediction as a JSON response
    return jsonify(predictions)


# predict_batch route
@bp_core_api.route("/predict_batch", methods=["POST"])
def predict_batch():
    # Get input data as JSON
    data = request.get_json()

    # Convert data to DataFrame
    features = pd.DataFrame(data)

    # Get prediction based on the features
    predictions = get_predictions(estimators=models, features=features)

    # Return the prediction as a JSON response
    return jsonify(predictions)
