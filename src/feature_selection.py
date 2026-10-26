"""Feature-selection helpers for the CICIDS2017 IDS project."""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier


def remove_correlated_features(X_train: pd.DataFrame, X_val: pd.DataFrame,
                               X_test: pd.DataFrame, threshold: float = 0.90):
    """Drop highly correlated numerical features using training data only.

    For each highly correlated pair, this simple strategy drops the later
    column in the DataFrame's column order. It does not rank the pair's value.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be greater than 0 and at most 1.")

    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [
        column for column in upper.columns
        if (upper[column] > threshold).any()
    ]

    return (
        X_train.drop(columns=to_drop).copy(),
        X_val.drop(columns=to_drop).copy(),
        X_test.drop(columns=to_drop).copy(),
        to_drop,
    )


def select_top_mi_features(X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, X_test: pd.DataFrame,
                           k: int = 20, random_state: int = 42):
    """Rank features by mutual information and apply the top-k selection."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    if X_train.isna().any().any():
        raise ValueError("Impute missing feature values before calculating mutual information.")

    scores = mutual_info_classif(X_train, y_train, random_state=random_state)
    mi_scores = pd.Series(scores, index=X_train.columns, name="MI score").sort_values(ascending=False)
    selected_features = mi_scores.head(min(k, len(mi_scores))).index.tolist()

    return (
        X_train[selected_features].copy(),
        X_val[selected_features].copy(),
        X_test[selected_features].copy(),
        selected_features,
        mi_scores,
    )


def evaluate_feature_counts(X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series,
                            mi_scores: pd.Series,
                            feature_counts: tuple[int, ...] = (10, 15, 20, 25, 30),
                            max_depth: int = 10, random_state: int = 42) -> pd.Series:
    """Score top mutual-information feature counts with a validation decision tree."""
    if not feature_counts or any(count < 1 for count in feature_counts):
        raise ValueError("feature_counts must contain positive integers.")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1.")

    ranked_features = [feature for feature in mi_scores.index if feature in X_train.columns]
    if not ranked_features:
        raise ValueError("mi_scores must contain at least one feature from X_train.")

    scores = {}
    for count in feature_counts:
        selected = ranked_features[:count]
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        model.fit(X_train[selected], y_train)
        predictions = model.predict(X_val[selected])
        scores[count] = f1_score(y_val, predictions, average="macro")

    return pd.Series(scores, name="Validation macro F1")


def plot_feature_correlations(X: pd.DataFrame, figsize: tuple[int, int] = (16, 12)):
    """Plot absolute feature correlations and return the matplotlib axes."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    correlation_matrix = X.corr().abs()
    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(correlation_matrix, cmap="coolwarm", vmin=0, vmax=1, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    return ax
