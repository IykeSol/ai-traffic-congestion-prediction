"""
AI Traffic Congestion Predictor
Model Definitions — XGBoost + RandomForest Soft-Voting Ensemble
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Label order for consistency
CONGESTION_CLASSES = ["low", "moderate", "high", "severe"]

CATEGORICAL_COLS = ["road_type", "city_zone", "weather_condition"]
FEATURE_COLS = [
    "hour", "hour_sin", "hour_cos",
    "day_of_week", "day_sin", "day_cos",
    "month", "month_sin", "month_cos",
    "is_weekend", "road_type", "city_zone", "weather_condition",
    "temperature_c", "vehicle_count", "average_speed_kmh",
    "signal_cycle_time", "road_capacity_utilization",
    "incident_reported", "school_zone", "construction_zone",
    "public_event_nearby",
]


class TrafficPredictor:
    """
    Ensemble traffic congestion predictor.
    Wraps preprocessing + XGBoost + RandomForest soft-voting.
    """

    def __init__(self):
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.target_encoder = LabelEncoder()
        self.model = None
        self.feature_names = FEATURE_COLS
        self.classes_ = CONGESTION_CLASSES

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def _encode_categoricals(self, df: pd.DataFrame,
                              fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_COLS:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen labels gracefully
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col])
        return df

    def preprocess(self, df: pd.DataFrame,
                   fit: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        df = df.copy()
        df = self._encode_categoricals(df, fit=fit)

        X = df[self.feature_names].values

        y = None
        if "congestion_level" in df.columns:
            if fit:
                y = self.target_encoder.fit_transform(df["congestion_level"])
            else:
                y = self.target_encoder.transform(df["congestion_level"])

        return X, y

    # ------------------------------------------------------------------ #
    # Model construction
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_ensemble(xgb_params: dict | None = None,
                       rf_params: dict | None = None) -> VotingClassifier:

        default_xgb = dict(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.80,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
        default_rf = dict(
            n_estimators=300,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

        if xgb_params:
            default_xgb.update(xgb_params)
        if rf_params:
            default_rf.update(rf_params)

        xgb = XGBClassifier(**default_xgb)
        rf  = RandomForestClassifier(**default_rf)

        ensemble = VotingClassifier(
            estimators=[("xgboost", xgb), ("random_forest", rf)],
            voting="soft",
            weights=[0.60, 0.40],
        )
        return ensemble

    # ------------------------------------------------------------------ #
    # Train / Predict
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray,
            xgb_params: dict | None = None,
            rf_params: dict | None = None):
        self.model = self.build_ensemble(xgb_params, rf_params)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        encoded = self.model.predict(X)
        return self.target_encoder.inverse_transform(encoded)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_single(self, feature_dict: dict) -> dict:
        """Predict from a raw feature dictionary."""
        df = pd.DataFrame([feature_dict])
        # fill any missing features with 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        X, _ = self.preprocess(df, fit=False)
        label = self.predict(X)[0]
        proba = self.predict_proba(X)[0]
        classes = self.target_encoder.classes_
        confidence = {
            cls: round(float(p), 4)
            for cls, p in zip(classes, proba)
        }
        return {
            "prediction": label,
            "confidence": confidence,
            "top_confidence": round(max(proba) * 100, 1),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, name: str = "traffic_predictor"):
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(self, path, compress=3)
        print(f"  Model saved >> {path}")
        return path

    @classmethod
    def load(cls, name: str = "traffic_predictor") -> "TrafficPredictor":
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        obj = joblib.load(path)
        print(f"  Model loaded << {path}")
        return obj

    # ------------------------------------------------------------------ #
    # Feature importance
    # ------------------------------------------------------------------ #

    def feature_importance_df(self) -> pd.DataFrame:
        xgb_model = self.model.estimators_[0]
        importances = xgb_model.feature_importances_
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df
