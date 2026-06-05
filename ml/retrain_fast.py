"""
Fast Retrain Script — skips Optuna, uses pre-found best hyperparameters.
Run: python ml/retrain_fast.py
"""
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

from ml.model import TrafficPredictor

console = Console()

DATA_PATH   = os.path.join("data", "traffic_dataset.csv")
METRICS_PATH = os.path.join("ml", "models", "metrics.json")

# Best hyperparameters from previous Optuna run
BEST_PARAMS = {
    "n_estimators":     312,
    "max_depth":        9,
    "learning_rate":    0.10455,
    "subsample":        0.82883,
    "colsample_bytree": 0.81106,
    "min_child_weight": 5,
    "gamma":            0.05274,
    "reg_alpha":        0.65119,
    "reg_lambda":       2.37772,
}


def main():
    console.print(Panel.fit(
        "[bold cyan]AI TRAFFIC CONGESTION PREDICTOR[/bold cyan]\n"
        "[dim]Fast Retrain — Using Pre-Tuned Hyperparameters[/dim]",
        border_style="cyan",
    ))
    t0 = time.time()

    # Load data
    console.print(Rule("[bold cyan]  Loading Dataset"))
    df = pd.read_csv(DATA_PATH)
    console.print(f"  Loaded [cyan]{len(df):,}[/cyan] rows.")

    # Preprocess
    predictor = TrafficPredictor()
    X, y = predictor.preprocess(df, fit=True)
    console.print(f"  Feature matrix: [cyan]{X.shape}[/cyan]")

    # Train/Test split BEFORE SMOTE
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    console.print(f"  Train: [cyan]{X_train_raw.shape[0]}[/cyan] | Test: [cyan]{X_test.shape[0]}[/cyan]")

    # SMOTE only on training set
    sm = SMOTE(sampling_strategy="not majority", random_state=42, k_neighbors=5)
    X_train, y_train = sm.fit_resample(X_train_raw, y_train_raw)
    console.print(f"  After SMOTE: [cyan]{X_train.shape}[/cyan]")

    # Train
    console.print(Rule("[bold cyan]  Training Ensemble (XGBoost + RandomForest)"))
    predictor.fit(X_train, y_train, xgb_params=BEST_PARAMS)
    console.print("  [bold green]Training complete.[/bold green]")

    # Evaluate on unseen test set
    console.print(Rule("[bold cyan]  Final Evaluation on Unseen Test Set"))
    y_pred      = predictor.model.predict(X_test)
    class_names = predictor.target_encoder.classes_

    acc         = accuracy_score(y_test, y_pred)
    f1_macro    = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    report      = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    # Metrics table
    tbl = Table(box=box.DOUBLE_EDGE, header_style="bold white",
                title="[bold cyan] Model Performance")
    tbl.add_column("Metric", style="dim white")
    tbl.add_column("Score",  justify="right", style="bold")

    def color(s):
        c = "green" if s >= 0.78 else "yellow" if s >= 0.65 else "red"
        return f"[{c}]{s*100:.2f}%[/{c}]"

    tbl.add_row("Accuracy",      color(acc))
    tbl.add_row("F1 (macro)",    color(f1_macro))
    tbl.add_row("F1 (weighted)", color(f1_weighted))
    console.print(tbl)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_tbl = Table(box=box.SIMPLE_HEAD, show_header=True)
    cm_tbl.add_column("Actual vs Predicted", style="bold dim")
    for cls in class_names:
        cm_tbl.add_column(cls, justify="center")
    for i, row in enumerate(cm):
        cm_tbl.add_row(class_names[i], *[str(v) for v in row])
    console.print("\n  [bold]Confusion Matrix:[/bold]")
    console.print(cm_tbl)

    # Per-class
    pc_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
    pc_tbl.add_column("Class")
    pc_tbl.add_column("Precision", justify="center")
    pc_tbl.add_column("Recall",    justify="center")
    pc_tbl.add_column("F1",        justify="center")
    pc_tbl.add_column("Support",   justify="center")
    for cls in class_names:
        r = report[cls]
        pc_tbl.add_row(
            cls,
            f"{r['precision']*100:.1f}%",
            f"{r['recall']*100:.1f}%",
            f"{r['f1-score']*100:.1f}%",
            str(int(r['support'])),
        )
    console.print("\n  [bold]Per-Class Metrics:[/bold]")
    console.print(pc_tbl)

    # Save metrics JSON
    fi = predictor.feature_importance_df()
    metrics = {
        "accuracy":           round(acc, 4),
        "f1_macro":           round(f1_macro, 4),
        "f1_weighted":        round(f1_weighted, 4),
        "per_class": {
            cls: {k: round(v, 4) for k, v in report[cls].items()}
            for cls in class_names
        },
        "feature_importances": fi.set_index("feature")["importance"].to_dict(),
        "classes":            class_names.tolist(),
    }
    os.makedirs("ml/models", exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model
    predictor.save("traffic_predictor")

    elapsed = time.time() - t0
    console.print(Panel.fit(
        f"[bold green]Training Complete![/bold green]\n"
        f"Final Accuracy : [bold]{acc*100:.2f}%[/bold]\n"
        f"F1 (macro)     : [bold]{f1_macro*100:.2f}%[/bold]\n"
        f"F1 (weighted)  : [bold]{f1_weighted*100:.2f}%[/bold]\n"
        f"Time elapsed   : [dim]{elapsed:.1f}s[/dim]\n\n"
        f"Run [cyan]python ml/predict.py[/cyan] for interactive prediction\n"
        f"Run [cyan]python app/app.py[/cyan] to start the web UI",
        border_style="green",
        title="[bold green] Done ",
    ))


if __name__ == "__main__":
    main()
