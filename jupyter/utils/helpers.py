from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from tqdm.notebook import tqdm
from sklearn.base import clone
from sklearn.metrics import get_scorer
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns


def grid_search(model, param_grid, X, y, metrics, *, k=3, random_state=42):
    # generate hyperparameters combinations
    param_grid = ParameterGrid(param_grid)

    # Initialize Stratified K-Fold cross-validation to maintain the distribution of target classes
    k_folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    # List to store results from each hyperparameter combination
    results = []

    # Iterate over all hyperparameter combinations in the grid
    for params in tqdm(param_grid):
        # Clone the model (creates a fresh copy) and set the current hyperparameters
        model_clone = clone(model.estimator).set_params(**params)

        # Initialize a dictionary to store the scores for each metric
        scores = {metric: [] for metric in metrics}

        # Perform K-fold cross-validation
        for train_idx, val_idx in k_folds.split(X, y):
            # Split data into training and validation sets for this fold
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

            # Train the model on the current fold's training data
            model_clone.fit(X_train_cv, y_train_cv)

            # Evaluate the model using each scoring metric
            for metric in metrics:
                # Get the scoring function for the current metric
                scorer = get_scorer(metric)
                # Compute the score on the validation set
                score = scorer(model_clone, X_val_cv, y_val_cv)
                # Store the score in the dictionary
                scores[metric].append(score)

        # Compute the average score across all folds for each metric
        avg_scores = {metric: np.mean(scores[metric]) for metric in scores}

        # Store the results for this hyperparameter combination
        results.append({"estimator": model_clone, "params": params, "scores": avg_scores})

    # Sort results based on all scoring metrics (descending order) to find the best model
    results.sort(
        key=lambda item: tuple(item["scores"][metric] for metric in metrics),
        reverse=True,
    )

    # Extract the best model, its parameters, and the associated scores
    best_estimator = results[0]["estimator"]
    best_params = results[0]["params"]
    best_scores_cv = results[0]["scores"]

    # Update the original model with the best estimator and parameters
    model.estimator = best_estimator
    model.params = best_params
    model.val_scores = best_scores_cv

    # Print model specifications
    model_specs(model)


def model_specs(model):
    # Initialize the PrettyTable with specific formatting options
    table = PrettyTable(
        float_format=".1",  # Float formatting (rounding to 1 decimal place)
        junction_char="•",  # Character used for joining rows
        horizontal_char="—",  # Character used for horizontal separators
    )

    # Set the table title using the model name
    table.title = f"{model.name} — Cross-Validation Results"

    # Define the table headers for the hyperparameter section
    table.field_names = ["Hyperparameter", "Value"]

    # Add rows for model hyperparameters
    for key, value in model.params.items():
        table.add_row([f"{key}", value])

    # Add a divider between the hyperparameters and the scores
    table.add_divider()

    # Add a row for the scores section, labeled with "Scoring Metric" and "Value (%)"
    table.add_row(["Scoring Metric", "Value (%)"], divider=True)

    # Add rows for each scoring metric and its corresponding percentage score
    for metric, score in model.val_scores.items():
        formatted_score = f"{score * 100:.1f} %"  # Convert score to percentage with 1 decimal place
        table.add_row([f"{metric}", formatted_score])

    # Print the table to display the results
    print(table)


def custom_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    *,
    cmap="Blues",
    subplots_kws=None,
    cm_kws=None,
    annot_kws=None,
):
    """
    Generates and displays a customized confusion matrix with a heatmap.

    Args:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - cmap: The color map to use for the heatmap (default is "Blues").
    - labels: List of labels to display on the x and y axes (default is None).
    - subplots_kws: Additional arguments to customize the plot (default is None).
    - cm_kws: Additional arguments for confusion matrix computation (default is None).
    - annot_kws: Additional arguments to customize annotation appearance (default is None).

    Returns:
    - None (Displays the plot).
    """

    # Set default values for None arguments
    subplots_kws = subplots_kws or {}
    cm_kws = cm_kws or {}
    annot_kws = annot_kws or {}

    # Create the figure for plotting with the given subplot options
    fig, ax = plt.subplots(**subplots_kws)

    # If 'labels' is provided, ensure it's a list (if it's a NumPy array or pandas Series)
    if labels is not None:
        labels = labels.tolist() if isinstance(labels, (np.ndarray, pd.Series)) else labels

    # Compute the confusion matrix and transpose it for correct orientation
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, **cm_kws).transpose()

    # Create the heatmap with the confusion matrix
    sns.heatmap(
        data=cm,
        annot=True,  # Annotate the cells with numeric values
        fmt=".1f",  # Format the annotation to 1 decimal place
        cmap=cmap,  # Use the specified colormap
        linewidths=3,  # Set the linewidth between cells
        annot_kws={"fontweight": "bold", **annot_kws},  # Bold text for annotations
        ax=ax,  # Plot on the specified axis
    )

    # Set labels for the axes
    ax.set_xlabel("GROUND TRUTH", labelpad=10)
    ax.set_ylabel("PREDICTIONS", labelpad=10)

    # Set the x and y tick labels, if 'labels' are provided
    if labels:
        ax.set_xticks(ax.get_xticks(), labels=labels)
        ax.set_yticks(ax.get_yticks(), labels=labels)

    # Adjust the spines to make the borders more visible
    ax.spines[["left", "top"]].set_position(("outward", 10))

    # Configure tick parameters for a better appearance
    ax.tick_params(direction="inout", length=0)

    # Move the x-axis to the top
    ax.tick_params(labeltop=True, labelbottom=False)
    ax.xaxis.set_label_position("top")

    # Invert the x-axis and y-axis to match the usual confusion matrix layout
    ax.invert_yaxis()
    ax.invert_xaxis()

    # Access and modify the color bar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(length=0)  # Remove ticks on the color bar

    # Display the plot
    plt.show()


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
