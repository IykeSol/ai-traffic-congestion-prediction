"""
AI Traffic Congestion Predictor
Interactive Terminal Predictor — run: python ml/predict.py

Prompts user to enter road conditions and outputs:
  - Predicted congestion level
  - Confidence scores per class
  - Actionable recommendations
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich import box

from ml.model import TrafficPredictor

console = Console()

RECOMMENDATIONS = {
    "low":      ("No significant delays expected. Proceed normally.",
                 "Consider carpooling or cycling to reduce future congestion."),
    "moderate": ("Expect minor delays. Allow 10-15 extra minutes.",
                 "Use alternative routes if available. Monitor live traffic."),
    "high":     ("Significant delays likely. Depart earlier or later.",
                 "Public transport strongly recommended. Avoid peak corridors."),
    "severe":   ("Critical congestion. Severe delays unavoidable.",
                 "Postpone travel if possible. Emergency response protocols may be active."),
}


def prompt_features() -> dict:
    console.print(Panel.fit(
        "[bold cyan]Enter Road & Traffic Conditions[/bold cyan]\n"
        "[dim]Press Enter to accept defaults shown in brackets[/dim]",
        border_style="cyan",
    ))

    hour = IntPrompt.ask(
        "  Hour of day (0-23)", default=8
    )
    day_of_week = IntPrompt.ask(
        "  Day of week (0=Mon, 6=Sun)", default=0
    )
    month = IntPrompt.ask(
        "  Month (1-12)", default=6
    )

    road_type = Prompt.ask(
        "  Road type",
        choices=["highway", "arterial", "local", "expressway"],
        default="arterial"
    )
    city_zone = Prompt.ask(
        "  City zone",
        choices=["CBD", "residential", "industrial", "suburban", "airport"],
        default="CBD"
    )
    weather = Prompt.ask(
        "  Weather condition",
        choices=["clear", "rain", "fog", "snow", "storm"],
        default="clear"
    )

    temperature_c = FloatPrompt.ask("  Temperature (°C)", default=25.0)
    vehicle_count = IntPrompt.ask("  Vehicle count (vehicles/hr)", default=1800)
    avg_speed     = FloatPrompt.ask("  Average speed (km/h)", default=45.0)
    signal_cycle  = FloatPrompt.ask("  Signal cycle time (seconds)", default=90.0)
    cap_util      = FloatPrompt.ask(
        "  Road capacity utilization (0.0-1.0)", default=0.65
    )

    incident   = Prompt.ask("  Incident reported?",  choices=["yes","no"], default="no")
    school     = Prompt.ask("  School zone?",         choices=["yes","no"], default="no")
    construct  = Prompt.ask("  Construction zone?",   choices=["yes","no"], default="no")
    pub_event  = Prompt.ask("  Public event nearby?", choices=["yes","no"], default="no")

    is_weekend = int(day_of_week >= 5)

    import math
    def sincos(v, mx): return math.sin(2*math.pi*v/mx), math.cos(2*math.pi*v/mx)
    hs, hc = sincos(hour, 24)
    ds, dc = sincos(day_of_week, 7)
    ms, mc = sincos(month, 12)

    return {
        "hour":                    hour,
        "hour_sin":                round(hs, 5),
        "hour_cos":                round(hc, 5),
        "day_of_week":             day_of_week,
        "day_sin":                 round(ds, 5),
        "day_cos":                 round(dc, 5),
        "month":                   month,
        "month_sin":               round(ms, 5),
        "month_cos":               round(mc, 5),
        "is_weekend":              is_weekend,
        "road_type":               road_type,
        "city_zone":               city_zone,
        "weather_condition":       weather,
        "temperature_c":           temperature_c,
        "vehicle_count":           vehicle_count,
        "average_speed_kmh":       avg_speed,
        "signal_cycle_time":       signal_cycle,
        "road_capacity_utilization": cap_util,
        "incident_reported":       int(incident == "yes"),
        "school_zone":             int(school == "yes"),
        "construction_zone":       int(construct == "yes"),
        "public_event_nearby":     int(pub_event == "yes"),
    }


def display_result(result: dict):
    label      = result["prediction"]
    confidence = result["confidence"]
    top_conf   = result["top_confidence"]

    colors = {"low": "green", "moderate": "yellow",
              "high": "red", "severe": "bold red"}
    color  = colors.get(label, "white")

    console.print()
    console.print(Panel.fit(
        f"[bold {color}]Predicted Congestion Level: {label.upper()}[/bold {color}]\n"
        f"Confidence: [bold]{top_conf}%[/bold]",
        border_style=color.replace("bold ", ""),
        title="[bold white] Prediction Result ",
    ))

    # Confidence bar chart
    console.print("\n  [bold]Confidence Scores:[/bold]")
    cls_colors = {"low": "green", "moderate": "yellow",
                  "high": "red",  "severe": "bright_red"}
    for cls, prob in sorted(confidence.items(),
                            key=lambda x: x[1], reverse=True):
        pct  = prob * 100
        bar  = "#" * int(pct / 2.5)
        c    = cls_colors.get(cls, "white")
        marker = " <" if cls == label else ""
        console.print(
            f"    [{c}]{cls:<10}[/{c}] "
            f"[{c}]{bar:<40}[/{c}] "
            f"[bold]{pct:5.1f}%[/bold]{marker}"
        )

    # Recommendations
    rec_main, rec_tip = RECOMMENDATIONS.get(label, ("", ""))
    console.print(
        Panel(
            f"[bold white]Status:[/bold white]  {rec_main}\n"
            f"[bold white]Tip:[/bold white]     {rec_tip}",
            title="[bold cyan] Recommendations",
            border_style="cyan",
        )
    )


def main():
    console.print(Panel.fit(
        "[bold cyan]AI TRAFFIC CONGESTION PREDICTOR[/bold cyan]\n"
        "[dim]Interactive Terminal Prediction Tool[/dim]",
        border_style="cyan",
    ))

    # Load model
    try:
        predictor = TrafficPredictor.load("traffic_predictor")
    except FileNotFoundError:
        console.print("[red]  Model not found. Run: python ml/train.py first.[/red]")
        sys.exit(1)

    while True:
        features = prompt_features()

        console.print("\n  [dim]Running prediction...[/dim]")
        result = predictor.predict_single(features)
        display_result(result)

        again = Prompt.ask(
            "\n  Run another prediction?",
            choices=["yes", "no"], default="yes"
        )
        if again == "no":
            console.print(
                Panel.fit(
                    "[bold green]Session ended. Thank you for using the AI Traffic Predictor.[/bold green]",
                    border_style="green",
                )
            )
            break
        console.print()


if __name__ == "__main__":
    main()
