import json
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

    return json.dumps(result, indent=2)
