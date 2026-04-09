"""
Model Trainer Module - Trains ML models for anomaly detection.

Applies Machine Learning concepts (Sem 4):
- Decision Trees (ID3 algorithm)
- K-Means Clustering
- Naive Bayes Classification
- Linear Regression
- Model evaluation metrics
"""

import pickle
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter, defaultdict
from pathlib import Path


@dataclass
class TrainingResult:
    """Results from model training."""

    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Dict[str, int]
    feature_importance: Optional[Dict[str, float]] = None


class DecisionTreeClassifier:
    """
    Decision Tree classifier using ID3 algorithm.

    ID3 Algorithm (Information Gain):
    1. Calculate entropy of target variable
    2. For each feature:
       - Calculate information gain
       - Information Gain = Entropy(S) - Sum(|Sv|/|S| * Entropy(Sv))
    3. Select feature with highest information gain
    4. Recursively build tree

    This implementation uses binned continuous features.
    """

    def __init__(
        self, max_depth: int = 10, min_samples_split: int = 5, n_bins: int = 10
    ):
        """
        Initialize Decision Tree.

        Args:
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples required to split
            n_bins: Number of bins for continuous features
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_bins = n_bins
        self.tree: Optional[Dict] = None
        self.feature_names: List[str] = []
        self.feature_thresholds: Dict[int, List[float]] = {}
        self.feature_importance_: Dict[str, float] = {}

    def _entropy(self, labels: List[str]) -> float:
        """
        Calculate entropy of a label distribution.

        Entropy = -Sum(p_i * log2(p_i))

        Higher entropy = more uncertainty
        """
        if not labels:
            return 0.0

        counter = Counter(labels)
        n = len(labels)

        entropy = 0.0
        for count in counter.values():
            if count > 0:
                p = count / n
                entropy -= p * math.log2(p)

        return entropy

    def _information_gain(
        self, X: List[List[float]], y: List[str], feature_idx: int, threshold: float
    ) -> float:
        """
        Calculate information gain for a binary split.

        IG = H(S) - |S_left|/|S| * H(S_left) - |S_right|/|S| * H(S_right)
        """
        n = len(y)
        if n == 0:
            return 0.0

        # Split samples
        left_indices = [i for i, x in enumerate(X) if x[feature_idx] <= threshold]
        right_indices = [i for i, x in enumerate(X) if x[feature_idx] > threshold]

        if not left_indices or not right_indices:
            return 0.0

        # Calculate entropies
        parent_entropy = self._entropy(y)

        left_labels = [y[i] for i in left_indices]
        right_labels = [y[i] for i in right_indices]

        left_entropy = self._entropy(left_labels)
        right_entropy = self._entropy(right_labels)

        # Weighted sum
        n_left = len(left_indices)
        n_right = len(right_indices)

        weighted_entropy = (n_left / n * left_entropy) + (n_right / n * right_entropy)

        return parent_entropy - weighted_entropy

    def _find_best_split(
        self, X: List[List[float]], y: List[str]
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find the best feature and threshold for splitting."""
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        n_features = len(X[0]) if X else 0

        for feature_idx in range(n_features):
            # Get unique values for thresholds
            values = sorted(set(x[feature_idx] for x in X))

            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                gain = self._information_gain(X, y, feature_idx, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: List[List[float]], y: List[str], depth: int = 0) -> Dict:
        """Recursively build the decision tree."""
        # Check stopping conditions
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(set(y)) == 1
        ):
            # Return leaf node with majority class
            return {
                "leaf": True,
                "class": Counter(y).most_common(1)[0][0],
                "samples": len(y),
            }

        # Find best split
        feature_idx, threshold, gain = self._find_best_split(X, y)

        if feature_idx is None or gain == 0:
            return {
                "leaf": True,
                "class": Counter(y).most_common(1)[0][0],
                "samples": len(y),
            }

        # Track feature importance
        feature_name = (
            self.feature_names[feature_idx]
            if feature_idx < len(self.feature_names)
            else f"feature_{feature_idx}"
        )
        self.feature_importance_[feature_name] = (
            self.feature_importance_.get(feature_name, 0) + gain
        )

        # Split data
        left_indices = [i for i, x in enumerate(X) if x[feature_idx] <= threshold]
        right_indices = [i for i, x in enumerate(X) if x[feature_idx] > threshold]

        X_left = [X[i] for i in left_indices]
        y_left = [y[i] for i in left_indices]
        X_right = [X[i] for i in right_indices]
        y_right = [y[i] for i in right_indices]

        # Build child nodes
        return {
            "leaf": False,
            "feature": feature_idx,
            "feature_name": feature_name,
            "threshold": threshold,
            "left": self._build_tree(X_left, y_left, depth + 1),
            "right": self._build_tree(X_right, y_right, depth + 1),
        }

    def fit(
        self,
        X: List[List[float]],
        y: List[str],
        feature_names: Optional[List[str]] = None,
    ):
        """
        Train the decision tree.

        Args:
            X: Feature vectors (list of lists)
            y: Labels (list of strings)
            feature_names: Names of features for interpretation
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(len(X[0]))]
        self.feature_importance_ = {}
        self.tree = self._build_tree(X, y)

        # Normalize feature importance
        total = sum(self.feature_importance_.values()) or 1
        self.feature_importance_ = {
            k: v / total for k, v in self.feature_importance_.items()
        }

    def _predict_one(self, x: List[float], node: Dict) -> str:
        """Predict label for a single sample."""
        if node["leaf"]:
            return node["class"]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X: List[List[float]]) -> List[str]:
        """Predict labels for multiple samples."""
        if not self.tree:
            raise RuntimeError("Model not trained. Call fit() first.")

        return [self._predict_one(x, self.tree) for x in X]

    def get_tree_structure(self) -> Dict:
        """Get the tree structure for visualization."""
        return self.tree


class KMeansCluster:
    """
    K-Means Clustering for unsupervised anomaly detection.

    Algorithm:
    1. Initialize K centroids randomly
    2. Assign each point to nearest centroid
    3. Update centroids as mean of assigned points
    4. Repeat until convergence

    Anomaly Detection:
    Points far from any centroid are considered anomalies.
    """

    def __init__(
        self, n_clusters: int = 2, max_iterations: int = 100, random_state: int = 42
    ):
        """
        Initialize K-Means.

        Args:
            n_clusters: Number of clusters (K)
            max_iterations: Maximum iterations
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids: List[List[float]] = []
        self.labels_: List[int] = []
        self.inertia_: float = 0.0  # Sum of squared distances

    def _euclidean_distance(self, a: List[float], b: List[float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def _find_nearest_centroid(self, point: List[float]) -> int:
        """Find the index of the nearest centroid."""
        distances = [self._euclidean_distance(point, c) for c in self.centroids]
        return distances.index(min(distances))

    def _calculate_centroid(self, points: List[List[float]]) -> List[float]:
        """Calculate centroid of a cluster."""
        if not points:
            return [0.0] * (len(self.centroids[0]) if self.centroids else 0)

        n_features = len(points[0])
        return [sum(p[i] for p in points) / len(points) for i in range(n_features)]

    def fit(self, X: List[List[float]]):
        """
        Fit K-Means to data.

        Args:
            X: Feature vectors
        """
        if not X:
            return

        import random

        random.seed(self.random_state)

        # Initialize centroids randomly (K-Means++)
        n_samples = len(X)
        n_features = len(X[0])

        # First centroid: random
        self.centroids = [X[random.randint(0, n_samples - 1)].copy()]

        # Remaining centroids: weighted by distance
        for _ in range(1, self.n_clusters):
            distances = []
            for x in X:
                min_dist = min(self._euclidean_distance(x, c) for c in self.centroids)
                distances.append(min_dist**2)

            total = sum(distances)
            if total > 0:
                probabilities = [d / total for d in distances]
                cumulative = 0
                r = random.random()
                for i, p in enumerate(probabilities):
                    cumulative += p
                    if cumulative > r:
                        self.centroids.append(X[i].copy())
                        break
            else:
                self.centroids.append(X[random.randint(0, n_samples - 1)].copy())

        # Iterative optimization
        for iteration in range(self.max_iterations):
            # Assign points to clusters
            self.labels_ = [self._find_nearest_centroid(x) for x in X]

            # Update centroids
            new_centroids = []
            for k in range(self.n_clusters):
                cluster_points = [
                    X[i] for i, label in enumerate(self.labels_) if label == k
                ]
                if cluster_points:
                    new_centroids.append(self._calculate_centroid(cluster_points))
                else:
                    new_centroids.append(self.centroids[k])

            # Check convergence
            movement = sum(
                self._euclidean_distance(old, new)
                for old, new in zip(self.centroids, new_centroids)
            )

            self.centroids = new_centroids

            if movement < 1e-6:
                break

        # Calculate inertia
        self.inertia_ = sum(
            self._euclidean_distance(X[i], self.centroids[self.labels_[i]]) ** 2
            for i in range(len(X))
        )

    def predict(self, X: List[List[float]]) -> List[int]:
        """Predict cluster labels."""
        return [self._find_nearest_centroid(x) for x in X]

    def distance_to_centroid(self, x: List[float]) -> float:
        """Get distance from point to its nearest centroid."""
        if not self.centroids:
            return 0.0
        return min(self._euclidean_distance(x, c) for c in self.centroids)

    def get_anomaly_scores(self, X: List[List[float]]) -> List[float]:
        """
        Calculate anomaly scores based on distance to centroids.

        Higher score = more likely to be an anomaly.
        """
        if not self.centroids:
            return [0.0] * len(X)

        distances = [self.distance_to_centroid(x) for x in X]

        # Normalize by max distance
        max_dist = max(distances) if distances else 1
        if max_dist > 0:
            return [d / max_dist for d in distances]
        return distances


class NaiveBayesClassifier:
    """
    Gaussian Naive Bayes Classifier.

    Bayes' Theorem:
    P(y|x) = P(x|y) * P(y) / P(x)

    Naive assumption: features are independent given the class.

    P(x|y) = Product of P(x_i|y) for all features

    For continuous features, assumes Gaussian distribution.
    """

    def __init__(self):
        self.class_priors: Dict[str, float] = {}
        self.feature_params: Dict[str, Dict[int, Tuple[float, float]]] = (
            {}
        )  # class -> feature_idx -> (mean, std)
        self.classes: List[str] = []

    def fit(self, X: List[List[float]], y: List[str]):
        """
        Train Naive Bayes classifier.

        Args:
            X: Feature vectors
            y: Labels
        """
        n_samples = len(y)
        n_features = len(X[0]) if X else 0

        # Calculate class priors
        class_counts = Counter(y)
        self.classes = list(class_counts.keys())
        self.class_priors = {c: count / n_samples for c, count in class_counts.items()}

        # Calculate feature statistics per class
        self.feature_params = {}

        for cls in self.classes:
            self.feature_params[cls] = {}
            indices = [i for i, label in enumerate(y) if label == cls]

            for feature_idx in range(n_features):
                values = [X[i][feature_idx] for i in indices]

                mean = sum(values) / len(values) if values else 0
                variance = (
                    sum((v - mean) ** 2 for v in values) / len(values)
                    if len(values) > 1
                    else 1e-6
                )
                std = math.sqrt(variance) if variance > 0 else 1e-6

                self.feature_params[cls][feature_idx] = (mean, std)

    def _gaussian_pdf(self, x: float, mean: float, std: float) -> float:
        """Calculate Gaussian probability density."""
        if std < 1e-6:
            std = 1e-6

        exponent = -((x - mean) ** 2) / (2 * std**2)
        return (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(exponent)

    def _predict_one(self, x: List[float]) -> Tuple[str, Dict[str, float]]:
        """Predict class for a single sample."""
        log_probs = {}

        for cls in self.classes:
            # Start with log prior
            log_prob = math.log(self.class_priors[cls])

            # Add log likelihoods
            for feature_idx, value in enumerate(x):
                mean, std = self.feature_params[cls].get(feature_idx, (0, 1))
                pdf = self._gaussian_pdf(value, mean, std)

                if pdf > 0:
                    log_prob += math.log(pdf)
                else:
                    log_prob += -100  # Very small probability

            log_probs[cls] = log_prob

        # Return class with highest probability
        predicted_class = max(log_probs.keys(), key=lambda k: log_probs[k])

        # Convert to actual probabilities
        max_log = max(log_probs.values())
        probs = {cls: math.exp(lp - max_log) for cls, lp in log_probs.items()}
        total = sum(probs.values())
        probs = {cls: p / total for cls, p in probs.items()}

        return predicted_class, probs

    def predict(self, X: List[List[float]]) -> List[str]:
        """Predict classes for multiple samples."""
        return [self._predict_one(x)[0] for x in X]

    def predict_proba(self, X: List[List[float]]) -> List[Dict[str, float]]:
        """Get class probabilities for multiple samples."""
        return [self._predict_one(x)[1] for x in X]


class LinearRegressionModel:
    """
    Linear Regression using Ordinary Least Squares (OLS).

    Used for predicting continuous values like:
    - Expected packet size
    - Expected traffic volume
    - Anomaly score regression

    Formula: y = X * w + b
    where w = (X^T X)^(-1) X^T y  (Normal Equation)

    Applies Machine Learning concepts (Sem 4):
    - Regression analysis
    - Cost function minimization (MSE)
    - Feature-target relationship modeling
    """

    def __init__(self):
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.r_squared: float = 0.0
        self.mse: float = 0.0
        self.feature_names: List[str] = []

    def fit(
        self,
        X: List[List[float]],
        y: List[float],
        feature_names: Optional[List[str]] = None,
    ):
        """
        Fit linear regression using gradient descent.

        Uses gradient descent rather than the Normal Equation to avoid
        matrix inversion, which is more numerically stable for
        high-dimensional data.

        Args:
            X: Feature matrix (n_samples x n_features)
            y: Target values (continuous)
            feature_names: Optional feature names
        """
        if not X or not y:
            return

        n_samples = len(X)
        n_features = len(X[0])
        self.feature_names = feature_names or [f"f{i}" for i in range(n_features)]

        # Normalize features for stable gradient descent
        self._means = [0.0] * n_features
        self._stds = [1.0] * n_features

        for j in range(n_features):
            col = [X[i][j] for i in range(n_samples)]
            self._means[j] = sum(col) / n_samples
            variance = sum((v - self._means[j]) ** 2 for v in col) / n_samples
            self._stds[j] = math.sqrt(variance) if variance > 0 else 1.0

        # Normalize X
        X_norm = []
        for i in range(n_samples):
            row = [
                (X[i][j] - self._means[j]) / self._stds[j] for j in range(n_features)
            ]
            X_norm.append(row)

        # Initialize weights
        self.weights = [0.0] * n_features
        self.bias = sum(y) / n_samples

        # Gradient descent
        lr = 0.01
        epochs = 1000

        for _ in range(epochs):
            # Predict
            predictions = [
                sum(X_norm[i][j] * self.weights[j] for j in range(n_features))
                + self.bias
                for i in range(n_samples)
            ]

            # Compute gradients
            errors = [predictions[i] - y[i] for i in range(n_samples)]

            grad_w = [0.0] * n_features
            for j in range(n_features):
                grad_w[j] = (
                    2
                    / n_samples
                    * sum(errors[i] * X_norm[i][j] for i in range(n_samples))
                )

            grad_b = 2 / n_samples * sum(errors)

            # Update
            for j in range(n_features):
                self.weights[j] -= lr * grad_w[j]
            self.bias -= lr * grad_b

        # Calculate metrics on training data
        predictions = self.predict(X)
        self.mse = (
            sum((predictions[i] - y[i]) ** 2 for i in range(n_samples)) / n_samples
        )

        y_mean = sum(y) / n_samples
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum((y[i] - predictions[i]) ** 2 for i in range(n_samples))
        self.r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def predict(self, X: List[List[float]]) -> List[float]:
        """Predict continuous values."""
        predictions = []
        n_features = len(self.weights)

        for sample in X:
            # Normalize using training statistics
            x_norm = [
                (sample[j] - self._means[j]) / self._stds[j] for j in range(n_features)
            ]
            y_pred = (
                sum(x_norm[j] * self.weights[j] for j in range(n_features)) + self.bias
            )
            predictions.append(y_pred)

        return predictions

    def get_coefficients(self) -> Dict[str, float]:
        """Get feature coefficients (weights) — shows feature importance."""
        return {name: weight for name, weight in zip(self.feature_names, self.weights)}


class ModelTrainer:
    """
    High-level interface for training and evaluating ML models.
    """

    def __init__(self, models_dir: str = "models"):
        """
        Initialize trainer.

        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train_decision_tree(
        self,
        X: List[List[float]],
        y: List[str],
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[DecisionTreeClassifier, TrainingResult]:
        """Train and evaluate Decision Tree."""
        # Split data (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train
        model = DecisionTreeClassifier(**kwargs)
        model.fit(X_train, y_train, feature_names)

        # Evaluate
        y_pred = model.predict(X_test)
        result = self._evaluate(y_test, y_pred, "DecisionTree")
        result.feature_importance = model.feature_importance_

        return model, result

    def train_naive_bayes(
        self, X: List[List[float]], y: List[str]
    ) -> Tuple[NaiveBayesClassifier, TrainingResult]:
        """Train and evaluate Naive Bayes."""
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train
        model = NaiveBayesClassifier()
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        result = self._evaluate(y_test, y_pred, "NaiveBayes")

        return model, result

    def train_kmeans(
        self, X: List[List[float]], n_clusters: int = 2
    ) -> Tuple[KMeansCluster, Dict]:
        """Train K-Means clustering."""
        model = KMeansCluster(n_clusters=n_clusters)
        model.fit(X)

        stats = {
            "model_type": "KMeans",
            "n_clusters": n_clusters,
            "inertia": model.inertia_,
            "cluster_sizes": dict(Counter(model.labels_)),
        }

        return model, stats

    def train_linear_regression(
        self,
        X: List[List[float]],
        y: List[float],
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[LinearRegressionModel, Dict]:
        """
        Train Linear Regression for anomaly score prediction.

        Unlike the classifiers, this predicts a continuous anomaly score.

        Args:
            X: Feature matrix
            y: Continuous target values (e.g., anomaly scores)
            feature_names: Optional feature names

        Returns:
            Trained model and evaluation statistics
        """
        # Split data (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train
        model = LinearRegressionModel()
        model.fit(X_train, y_train, feature_names)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        n_test = len(y_test)

        mse = (
            sum((y_test[i] - y_pred[i]) ** 2 for i in range(n_test)) / n_test
            if n_test > 0
            else 0
        )
        rmse = math.sqrt(mse)

        y_mean = sum(y_test) / n_test if n_test > 0 else 0
        ss_tot = sum((yi - y_mean) ** 2 for yi in y_test)
        ss_res = sum((y_test[i] - y_pred[i]) ** 2 for i in range(n_test))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        stats = {
            "model_type": "LinearRegression",
            "mse": round(mse, 6),
            "rmse": round(rmse, 6),
            "r_squared": round(r_squared, 4),
            "coefficients": model.get_coefficients(),
        }

        return model, stats

    def _evaluate(
        self, y_true: List[str], y_pred: List[str], model_type: str
    ) -> TrainingResult:
        """Calculate evaluation metrics."""
        # Confusion matrix
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == "anomaly" and p == "anomaly")
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == "normal" and p == "normal")
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == "normal" and p == "anomaly")
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == "anomaly" and p == "normal")

        # Metrics
        accuracy = (tp + tn) / len(y_true) if y_true else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return TrainingResult(
            model_type=model_type,
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            confusion_matrix={"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        )

    def save_model(self, model: Any, name: str, scaler: Any = None):
        """Save model to disk using pickle.

        Saves a dictionary containing the model, optional scaler, and
        model type so that MLDetector.load_model() can restore everything.
        """
        path = self.models_dir / f"{name}.pkl"
        data = {
            "model": model,
            "model_type": name,
            "scaler": scaler,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Model saved to {path}")

    def load_model(self, name: str) -> Any:
        """Load model from disk."""
        path = self.models_dir / f"{name}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)
