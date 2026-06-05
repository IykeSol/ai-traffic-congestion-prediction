"""
AI Traffic Congestion Predictor
Terminal Training Script — run: python ml/train.py

Shows live rich terminal output:
  - Data loading summary
  - Cross-validation fold results
  - Optuna hyperparameter tuning progress
  - Final model metrics table
  - Feature importance ranking
  - Saves best model to ml/models/
"""

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, MofNCompleteColumn
)
from rich.text import Text
from rich.rule import Rule
from rich import box

from ml.model import TrafficPredictor, FEATURE_COLS

# ─────────────────────────────────────────────
console = Console()
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "traffic_dataset.csv")
METRICS_PATH = os.path.join(os.path.dirname(__file__),
                            "models", "metrics.json")
N_TRIALS    = 30
N_FOLDS     = 5
SMOTE_RATIO = 0.85   # oversample minority classes toward majority


# ─────────────────────────────────────────────
def load_data():
    console.print(Rule("[bold cyan]  Loading Dataset"))
    if not os.path.exists(DATA_PATH):
        console.print(f"[red]  Dataset not found at {DATA_PATH}")
        console.print("[yellow]  Run: python data/generate_dataset.py  first.[/yellow]")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    tbl = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta")
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value",  style="white")
    tbl.add_row("Total samples",  f"{len(df):,}")
    tbl.add_row("Features",       str(len(FEATURE_COLS)))
    tbl.add_row("Target classes", str(df["congestion_level"].nunique()))
    tbl.add_row("Missing values", str(df.isnull().sum().sum()))
    console.print(tbl)

    dist = df["congestion_level"].value_counts().sort_index()
    console.print("\n  [bold]Class distribution:[/bold]")
    colors = {"low": "green", "moderate": "yellow",
              "high": "red",  "severe": "bold red"}
    for cls, cnt in dist.items():
        pct = cnt / len(df) * 100
        bar = "#" * int(pct / 2)
        color = colors.get(cls, "white")
        console.print(f"    [{color}]{cls:<10}[/{color}]  {bar}  {cnt:>5} ({pct:.1f}%)")

    return df


# ─────────────────────────────────────────────
def prepare_features(df: pd.DataFrame, predictor: TrafficPredictor):
    console.print(Rule("[bold cyan]  Feature Engineering"))
    X, y = predictor.preprocess(df, fit=True)
    console.print(f"  Feature matrix : [cyan]{X.shape}[/cyan]")
    console.print(f"  Label vector   : [cyan]{y.shape}[/cyan]  | classes: {list(predictor.target_encoder.classes_)}")
    return X, y


# ─────────────────────────────────────────────
def cross_validate(X, y):
    console.print(Rule("[bold cyan]  5-Fold Cross Validation  (default params)"))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.85, colsample_bytree=0.80,
        eval_metric="mlogloss", random_state=42,
        n_jobs=-1, tree_method="hist"
    )
    rf  = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42,
        n_jobs=-1, class_weight="balanced"
    )
    base_model = VotingClassifier(
        estimators=[("xgb", xgb), ("rf", rf)],
        voting="soft", weights=[0.6, 0.4]
    )

    fold_accs, fold_f1s = [], []
    tbl = Table(box=box.ROUNDED, header_style="bold blue")
    tbl.add_column("Fold", justify="center")
    tbl.add_column("Accuracy", justify="center")
    tbl.add_column("F1 (macro)", justify="center")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("  Cross-validating...", total=N_FOLDS)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            base_model.fit(X_tr, y_tr)
            preds = base_model.predict(X_val)

            acc = accuracy_score(y_val, preds)
            f1  = f1_score(y_val, preds, average="macro")
            fold_accs.append(acc)
            fold_f1s.append(f1)

            color = "green" if acc >= 0.75 else "yellow" if acc >= 0.65 else "red"
            tbl.add_row(
                str(fold),
                f"[{color}]{acc*100:.2f}%[/{color}]",
                f"{f1*100:.2f}%"
            )
            prog.advance(task)

    console.print(tbl)
    mean_acc = np.mean(fold_accs)
    mean_f1  = np.mean(fold_f1s)
    console.print(
        f"\n  Mean Accuracy  : [bold green]{mean_acc*100:.2f}%[/bold green]  "
        f"(± {np.std(fold_accs)*100:.2f}%)"
    )
    console.print(
        f"  Mean F1 macro  : [bold green]{mean_f1*100:.2f}%[/bold green]  "
        f"(± {np.std(fold_f1s)*100:.2f}%)"
    )
    return mean_acc, mean_f1


# ─────────────────────────────────────────────
def tune_hyperparams(X, y) -> dict:
    console.print(Rule("[bold cyan]  Optuna Hyperparameter Tuning"))
    console.print(f"  Running [bold]{N_TRIALS}[/bold] trials — optimising XGBoost params...\n")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    best_scores = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("  Tuning...", total=N_TRIALS)

        def objective(trial):
            params = {
                "n_estimators":    trial.suggest_int("n_estimators", 200, 600),
                "max_depth":       trial.suggest_int("max_depth", 4, 9),
                "learning_rate":   trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
                "subsample":       trial.suggest_float("subsample", 0.70, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.95),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
                "gamma":           trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha":       trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda":      trial.suggest_float("reg_lambda", 0.5, 3.0),
                "eval_metric": "mlogloss",
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",
            }
            xgb   = XGBClassifier(**params)
            scores = cross_val_score(xgb, X, y, cv=skf,
                                     scoring="accuracy", n_jobs=-1)
            score = scores.mean()
            best_scores.append(score)
            prog.advance(task)
            return score

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=N_TRIALS)

    best = study.best_params
    console.print(f"\n  Best trial accuracy : [bold green]{study.best_value*100:.2f}%[/bold green]")

    tbl = Table(box=box.SIMPLE_HEAD, header_style="bold magenta")
    tbl.add_column("Hyperparameter")
    tbl.add_column("Best Value", style="cyan")
    for k, v in best.items():
        tbl.add_row(k, str(round(v, 5) if isinstance(v, float) else v))
    console.print(tbl)
    return best


# ─────────────────────────────────────────────
def train_final(X, y, best_xgb_params: dict, predictor: TrafficPredictor):
    console.print(Rule("[bold cyan]  Training Final Ensemble Model"))
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("  Fitting XGBoost + RandomForest ensemble...", total=None)
        predictor.fit(X, y, xgb_params=best_xgb_params)
        prog.update(task, completed=True)

    console.print("  [bold green]Ensemble training complete.[/bold green]")


# ─────────────────────────────────────────────
def evaluate_final(X, y, predictor: TrafficPredictor):
    console.print(Rule("[bold cyan]  Final Evaluation on Unseen Test Set"))

    y_pred = predictor.model.predict(X)
    class_names = predictor.target_encoder.classes_

    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_weighted = f1_score(y, y_pred, average="weighted")

    # Metrics table
    tbl = Table(box=box.DOUBLE_EDGE, header_style="bold white",
                title="[bold cyan] Model Performance", show_footer=False)
    tbl.add_column("Metric",  style="dim white")
    tbl.add_column("Score",   justify="right", style="bold")

    def color_score(s):
        color = "green" if s >= 0.78 else "yellow" if s >= 0.70 else "red"
        return f"[{color}]{s*100:.2f}%[/{color}]"

    tbl.add_row("Accuracy",    color_score(acc))
    tbl.add_row("F1 (macro)",  color_score(f1_macro))
    tbl.add_row("F1 (weighted)", color_score(f1_weighted))
    console.print(tbl)

    # Confusion matrix
    console.print("\n  [bold]Confusion Matrix:[/bold]")
    cm = confusion_matrix(y, y_pred)
    cm_tbl = Table(box=box.SIMPLE_HEAD, show_header=True)
    cm_tbl.add_column("Actual \\ Predicted", style="bold dim")
    for cls in class_names:
        cm_tbl.add_column(cls, justify="center")
    for i, row in enumerate(cm):
        cm_tbl.add_row(class_names[i], *[str(v) for v in row])
    console.print(cm_tbl)

    # Per-class report
    report = classification_report(y, y_pred, target_names=class_names,
                                   output_dict=True)
    console.print("\n  [bold]Per-Class Metrics:[/bold]")
    cls_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold blue")
    cls_tbl.add_column("Class")
    cls_tbl.add_column("Precision", justify="right")
    cls_tbl.add_column("Recall",    justify="right")
    cls_tbl.add_column("F1",        justify="right")
    cls_tbl.add_column("Support",   justify="right")
    colors = {"low": "green", "moderate": "yellow",
              "high": "red", "severe": "bold red"}
    for cls in class_names:
        r = report[cls]
        c = colors.get(cls, "white")
        cls_tbl.add_row(
            f"[{c}]{cls}[/{c}]",
            f"{r['precision']*100:.1f}%",
            f"{r['recall']*100:.1f}%",
            f"{r['f1-score']*100:.1f}%",
            str(int(r["support"])),
        )
    console.print(cls_tbl)

    # Feature importances
    console.print("\n  [bold]Top 10 Feature Importances (XGBoost):[/bold]")
    fi_df = predictor.feature_importance_df().head(10)
    fi_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold magenta")
    fi_tbl.add_column("Rank", justify="center")
    fi_tbl.add_column("Feature")
    fi_tbl.add_column("Importance", justify="right")
    for rank, row in fi_df.iterrows():
        bar = "#" * int(row["importance"] * 200)
        fi_tbl.add_row(str(rank+1), row["feature"],
                       f"[cyan]{bar}[/cyan] {row['importance']:.4f}")
    console.print(fi_tbl)

    # Save metrics JSON
    metrics = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class": {
            cls: {k: round(v, 4) for k, v in report[cls].items()}
            for cls in class_names
        },
        "feature_importances": fi_df.set_index("feature")["importance"].to_dict(),
        "classes": list(class_names),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    console.print(f"  Metrics saved >> {METRICS_PATH}")
    return metrics


# ─────────────────────────────────────────────
def main():
    console.print(Panel.fit(
        "[bold cyan]AI TRAFFIC CONGESTION PREDICTOR[/bold cyan]\n"
        "[dim]Senior ML Engineer Training Pipeline v1.0[/dim]",
        border_style="cyan",
    ))

    t0 = time.time()

    df        = load_data()
    predictor = TrafficPredictor()
    X, y      = prepare_features(df, predictor)

    # CRITICAL: split BEFORE SMOTE so test set has no synthetic samples
    from sklearn.model_selection import train_test_split
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    console.print(f"  Train split : {X_train_raw.shape[0]} samples | Test split : {X_test.shape[0]} samples")

    # Apply SMOTE only to training data
    console.print(f"\n  Applying SMOTE only to training set (ratio={SMOTE_RATIO})...")
    sm = SMOTE(sampling_strategy="not majority", random_state=42, k_neighbors=5)
    X_train, y_train = sm.fit_resample(X_train_raw, y_train_raw)
    console.print(f"  After SMOTE (train) : [cyan]{X_train.shape}[/cyan]")

    cv_acc, _   = cross_validate(X_train, y_train)
    best_params = tune_hyperparams(X_train, y_train)
    train_final(X_train, y_train, best_params, predictor)
    metrics     = evaluate_final(X_test, y_test, predictor)
    predictor.save("traffic_predictor")

    elapsed = time.time() - t0
    console.print(Panel.fit(
        f"[bold green]Training Complete![/bold green]\n"
        f"Final Accuracy : [bold]{metrics['accuracy']*100:.2f}%[/bold]\n"
        f"F1 (macro)     : [bold]{metrics['f1_macro']*100:.2f}%[/bold]\n"
        f"Time elapsed   : [dim]{elapsed:.1f}s[/dim]\n\n"
        f"Run [bold cyan]python ml/evaluate.py[/bold cyan] for full evaluation\n"
        f"Run [bold cyan]python ml/predict.py[/bold cyan] for interactive prediction",
        border_style="green",
        title="[bold green] Done ",
    ))


if __name__ == "__main__":
    main()
