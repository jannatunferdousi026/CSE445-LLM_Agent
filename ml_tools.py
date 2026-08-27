import json
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim


DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer
}


def load_dataset_summary(dataset_name: str) -> str:
    """Loads a standard benchmark dataset and returns summary statistics."""

    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({
            "error": f"Unknown dataset '{name}'. Options: {list(DATASETS.keys())}"
        })

    data = DATASETS[name]()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    summary = {
        "dataset": name,
        "n_samples": df.shape[0],
        "n_features": len(data.feature_names),
        "feature_names": list(data.feature_names),
        "classes": [str(c) for c in np.unique(data.target)],
        "missing_values": int(df.isnull().sum().sum())
    }

    return json.dumps(summary)


def train_sklearn_model(
    dataset_name: str,
    model_type: str,
    test_size: float = 0.2
) -> str:
    """Trains a Scikit-Learn model and returns evaluation results."""

    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({
            "error": f"Dataset '{name}' not found."
        })

    data = DATASETS[name]()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=test_size,
        random_state=42,
        stratify=data.target
    )

    model_type = model_type.lower().strip()

    if model_type == "decision_tree":
        clf = DecisionTreeClassifier(
            max_depth=4,
            random_state=42
        )

    elif model_type == "logistic_regression":
        clf = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=50,
            random_state=42
        )

    else:
        return json.dumps({
            "error": f"Unsupported model '{model_type}'."
        })

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    cv_scores = cross_val_score(
        clf,
        data.data,
        data.target,
        cv=5
    )

    result = {
        "dataset": name,
        "model": model_type,
        "test_size": test_size,
        "test_accuracy": float(accuracy),
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True
        )
    }

    return json.dumps(result, indent=2)


class MLP(nn.Module):
    """Simple PyTorch Multi-Layer Perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def train_pytorch_mlp(
    dataset_name: str,
    hidden_dim: int = 32,
    epochs: int = 50,
    lr: float = 0.01
) -> str:
    """Trains a simple PyTorch MLP classifier."""

    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({
            "error": f"Dataset '{name}' not found."
        })

    data = DATASETS[name]()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target
    )

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.long
    )

    input_dim = X_train.shape[1]
    output_dim = len(np.unique(data.target))

    model = MLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    model.train()

    for epoch in range(epochs):

        optimizer.zero_grad()

        outputs = model(X_train)

        loss = criterion(
            outputs,
            y_train
        )

        loss.backward()

        optimizer.step()

    model.eval()

    with torch.no_grad():

        outputs = model(X_test)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        accuracy = (
            predictions == y_test
        ).float().mean().item()

    result = {
        "dataset": name,
        "model": "pytorch_mlp",
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "learning_rate": lr,
        "test_accuracy": float(accuracy),
        "final_loss": float(loss.item())
    }
def tune_hyperparameters(dataset_name: str, model_name: str) -> str:
    """Tune model hyperparameters using GridSearchCV."""

    name = dataset_name.lower().strip()
    model = model_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({
            "error": f"Unknown dataset '{name}'. Options: {list(DATASETS.keys())}"
        })

    if model not in ["svc", "decision_tree"]:
        return json.dumps({
            "error": "Unsupported model. Options: ['svc', 'decision_tree']"
        })

    data = DATASETS[name]()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    if model == "svc":
        estimator = SVC()

        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }

    else:
        estimator = DecisionTreeClassifier(
            random_state=42
        )

        param_grid = {
            "max_depth": [None, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"]
        }

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    test_accuracy = best_model.score(
        X_test,
        y_test
    )

    result = {
        "dataset": name,
        "model": model,
        "method": "GridSearchCV",
        "cv_folds": 5,
        "best_parameters": grid_search.best_params_,
        "best_cv_accuracy": float(grid_search.best_score_),
        "test_accuracy": float(test_accuracy),
        "number_of_parameter_combinations": len(
            grid_search.cv_results_["params"]
        )
    }

    return json.dumps(result, indent=2)
def analyze_features(dataset_name: str) -> str:
    """Apply PCA and Sequential Feature Selection to a dataset."""

    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({
            "error": f"Unknown dataset '{name}'. Options: {list(DATASETS.keys())}"
        })

    data = DATASETS[name]()

    X = data.data
    y = data.target

    n_components = min(2, X.shape[1])

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    estimator = LogisticRegression(max_iter=1000)

    n_features_to_select = max(1, min(2, X.shape[1]))

    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features_to_select,
        direction="forward",
        scoring="accuracy",
        cv=5
    )

    selector.fit(X, y)

    selected_features = [
        data.feature_names[i]
        for i, selected in enumerate(selector.get_support())
        if selected
    ]

    result = {
        "dataset": name,
        "original_features": int(X.shape[1]),
        "pca_components": n_components,
        "explained_variance_ratio": [
            float(x) for x in pca.explained_variance_ratio_
        ],
        "selected_features": selected_features,
        "number_of_selected_features": len(selected_features)
    }

    return json.dumps(result, indent=2)
def train_advanced_pytorch_classifier(
    dataset_name: str,
    epochs: int = 80
) -> str:
    """Advanced PyTorch classifier with BatchNorm, Dropout and LR Scheduler."""

    name = dataset_name.lower().strip()

    if name not in DATASETS:
        return json.dumps({
            "error": f"Unknown dataset '{name}'. Options: {list(DATASETS.keys())}"
        })

    data = DATASETS[name]()

    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    class AdvancedMLP(nn.Module):

        def __init__(self, input_dim, num_classes):
            super().__init__()

            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(32, num_classes)
            )

        def forward(self, x):
            return self.network(x)

    model = AdvancedMLP(
        X.shape[1],
        len(np.unique(y))
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.01
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=25,
        gamma=0.5
    )

    model.train()

    final_loss = 0

    for _ in range(epochs):

        optimizer.zero_grad()

        outputs = model(X_train)

        loss = criterion(outputs, y_train)

        loss.backward()

        optimizer.step()

        scheduler.step()

        final_loss = loss.item()

    model.eval()

    with torch.no_grad():

        predictions = model(X_test).argmax(dim=1)

        accuracy = (
            predictions.eq(y_test)
            .float()
            .mean()
            .item()
        )

    result = {
        "dataset": name,
        "model": "advanced_pytorch_classifier",
        "epochs": epochs,
        "architecture": "64 → BatchNorm → Dropout → 32 → BatchNorm → Dropout",
        "optimizer": "Adam",
        "scheduler": "StepLR",
        "test_accuracy": accuracy,
        "final_loss": final_loss
    }

    return json.dumps(result, indent=2)
