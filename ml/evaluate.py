"""
AI Traffic Congestion Predictor
Terminal Evaluation Script — run: python ml/evaluate.py

Loads the saved model and runs a comprehensive hold-out evaluation.
Prints full classification report, ROC-AUC, and saves evaluation JSON.
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

from ml.model import TrafficPredictor

console = Console()

DATA_PATH    = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "data", "traffic_dataset.csv")
REPORT_PATH  = os.path.join(os.path.dirname(__file__),
                             "models", "eval_report.json")
CM_PLOT_PATH = os.path.join(os.path.dirname(__file__),
                             "models", "confusion_matrix.png")


def load_model_and_data():
    console.print(Rule("[bold cyan]  Loading Model & Test Data"))
    predictor = TrafficPredictor.load("traffic_predictor")

    df = pd.read_csv(DATA_PATH)
    X, y = predictor.preprocess(df, fit=False)

    # 80/20 stratified split — held-out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    console.print(f"  Test set size  : [cyan]{X_test.shape[0]:,}[/cyan] samples (20%)")
    return predictor, X_test, y_test


def evaluate(predictor, X_test, y_test):
    console.print(Rule("[bold cyan]  Running Evaluation on Hold-Out Test Set"))

    y_pred      = predictor.model.predict(X_test)
    y_proba     = predictor.model.predict_proba(X_test)
    class_names = predictor.target_encoder.classes_

    acc      = accuracy_score(y_test, y_pred)
    f1_mac   = f1_score(y_test, y_pred, average="macro")
    f1_wt    = f1_score(y_test, y_pred, average="weighted")
    prec_mac = precision_score(y_test, y_pred, average="macro")
    rec_mac  = recall_score(y_test, y_pred, average="macro")

    # ROC-AUC (one-vs-rest)
    y_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    roc   = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")

    # ── Summary table ─────────────────────────────────────────────────
    tbl = Table(
        box=box.DOUBLE_EDGE,
        header_style="bold white",
        title="[bold cyan] Evaluation Results — Hold-Out Test Set",
    )
    tbl.add_column("Metric",    style="dim white", min_width=22)
    tbl.add_column("Score",     justify="right", style="bold", min_width=12)

    def fmt(s):
        c = "green" if s >= 0.78 else "yellow" if s >= 0.68 else "red"
        return f"[{c}]{s*100:.2f}%[/{c}]"

    tbl.add_row("Accuracy",          fmt(acc))
    tbl.add_row("Precision (macro)", fmt(prec_mac))
    tbl.add_row("Recall (macro)",    fmt(rec_mac))
    tbl.add_row("F1 (macro)",        fmt(f1_mac))
    tbl.add_row("F1 (weighted)",     fmt(f1_wt))
    tbl.add_row("ROC-AUC (OvR)",     fmt(roc))
    console.print(tbl)

    # ── Per-class breakdown ───────────────────────────────────────────
    report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True
    )
    console.print("\n  [bold]Per-Class Breakdown:[/bold]")
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

    # ── Confusion matrix (text art) ───────────────────────────────────
    console.print("\n  [bold]Confusion Matrix (rows=actual, cols=predicted):[/bold]")
    cm = confusion_matrix(y_test, y_pred)
    cm_tbl = Table(box=box.SIMPLE_HEAD)
    cm_tbl.add_column("Actual \\ Pred", style="bold dim")
    for cn in class_names:
        cm_tbl.add_column(cn, justify="center")
    for i, row in enumerate(cm):
        cm_tbl.add_row(class_names[i], *[str(v) for v in row])
    console.print(cm_tbl)

    # ── Save confusion matrix plot ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax
    )
    ax.set_title("Confusion Matrix — AI Traffic Congestion Predictor", pad=14)
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label",      labelpad=10)
    plt.tight_layout()
    fig.savefig(CM_PLOT_PATH, dpi=150)
    plt.close()
    console.print(f"\n  Confusion matrix plot saved → {CM_PLOT_PATH}")

    # ── Save JSON report ──────────────────────────────────────────────
    eval_data = {
        "accuracy":         round(acc,      4),
        "f1_macro":         round(f1_mac,   4),
        "f1_weighted":      round(f1_wt,    4),
        "precision_macro":  round(prec_mac, 4),
        "recall_macro":     round(rec_mac,  4),
        "roc_auc_macro":    round(roc,      4),
        "per_class": {
            cls: {k: round(v, 4) for k, v in report[cls].items()}
            for cls in class_names
        },
        "confusion_matrix": cm.tolist(),
        "class_names":      list(class_names),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(eval_data, f, indent=2)
    console.print(f"  Evaluation report saved → {REPORT_PATH}")

    return eval_data


def main():
    console.print(Panel.fit(
        "[bold cyan]AI TRAFFIC CONGESTION PREDICTOR[/bold cyan]\n"
        "[dim]Model Evaluation — Hold-Out Test Set[/dim]",
        border_style="cyan",
    ))

    predictor, X_test, y_test = load_model_and_data()
    results = evaluate(predictor, X_test, y_test)

    console.print(Panel.fit(
        f"[bold green]Evaluation Complete[/bold green]\n"
        f"Test Accuracy  : [bold]{results['accuracy']*100:.2f}%[/bold]\n"
        f"ROC-AUC (OvR)  : [bold]{results['roc_auc_macro']*100:.2f}%[/bold]\n\n"
        f"Run [bold cyan]python ml/predict.py[/bold cyan] to test individual predictions",
        border_style="green",
        title="[bold green] Done ",
    ))


if __name__ == "__main__":
    main()
