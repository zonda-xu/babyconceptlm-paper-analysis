#!/usr/bin/env python3
"""Build exposure-indexed STRICT and MULTILINGUAL learning curves."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
from pathlib import Path
from statistics import mean


EXPOSURES_M = [
    *range(1, 11),
    *range(20, 101, 10),
    *range(200, 1001, 100),
]
MODELS = {
    "noDWA-242": "babyconceptLM-242-en-noDWA",
    "DWA-242": "babyconceptLM-242-en",
    "noDWA-583": "babyconceptLM-583-en-noDWA",
    "DWA-583": "babyconceptLM-583-en",
}

MULTILINGUAL_ZERO_SHOT_TASKS = {
    "eng": [
        "blimp_babylm_filtered",
        "hellaswag_en_mubench",
        "multiblimp_eng",
        "winogrande_en_mubench",
        "xstorycloze_en_mubench",
    ],
    "nld": [
        "blimp_nl",
        "hellaswag_nl_mubench",
        "multiblimp_nld",
        "winogrande_nl_mubench",
        "xcomps_nl",
        "xstorycloze_nl_mubench",
    ],
    "zho": [
        "hellaswag_zh_mubench",
        "winogrande_zh_mubench",
        "xcomps_zh",
        "xstorycloze_zh_mubench",
        "zhoblimp",
    ],
}

MULTILINGUAL_LANGUAGE_CODES = {"eng": "en", "nld": "nl", "zho": "zh"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-results-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--multilingual-results-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--multilingual-collator",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--multilingual-model-name",
        default="babyconceptLM-multi",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def average_accuracy(report_path: Path) -> float:
    matches = re.findall(
        r"### AVERAGE ACCURACY\s*\n([+-]?[0-9.]+)",
        report_path.read_text(encoding="utf-8"),
    )
    if not matches:
        raise ValueError(f"No average accuracy in {report_path}")
    return float(matches[-1])


def load_model_rows(results_dir: Path, model_name: str) -> list[dict[str, float]]:
    model_dir = results_dir / model_name
    collated_path = model_dir / "all_full_preds_and_fast_scores_causal.json"
    fast_results = json.loads(collated_path.read_text(encoding="utf-8"))[
        "fast_eval_results"
    ]
    if len(fast_results["reading"]) != len(EXPOSURES_M):
        raise ValueError(f"Unexpected checkpoint count in {collated_path}")

    rows = []
    for index, exposure_m in enumerate(EXPOSURES_M):
        zero_shot = (
            model_dir
            / f"chck_{exposure_m}M"
            / "zero_shot"
            / "causal"
        )
        blimp = average_accuracy(
            zero_shot / "blimp" / "blimp_fast" / "best_temperature_report.txt"
        )
        supplement = average_accuracy(
            zero_shot
            / "blimp"
            / "supplement_fast"
            / "best_temperature_report.txt"
        )
        ewok = average_accuracy(
            zero_shot / "ewok" / "ewok_fast" / "best_temperature_report.txt"
        )
        entity = average_accuracy(
            zero_shot
            / "entity_tracking"
            / "entity_tracking_fast"
            / "best_temperature_report.txt"
        )
        parallel = average_accuracy(
            zero_shot
            / "global_piqa_parallel"
            / "global_piqa_parallel"
            / "best_temperature_report.txt"
        )
        nonparallel = average_accuracy(
            zero_shot
            / "global_piqa_nonparallel"
            / "global_piqa_nonparallel"
            / "best_temperature_report.txt"
        )
        global_piqa = mean([parallel, nonparallel])
        reading = mean(fast_results["reading"][index].values()) * 100
        fast_macro = mean(
            [blimp, supplement, ewok, entity, global_piqa, reading]
        )
        rows.append(
            {
                "exposure_m": exposure_m,
                "blimp": blimp,
                "supplement": supplement,
                "ewok": ewok,
                "entity_tracking": entity,
                "global_piqa": global_piqa,
                "reading": reading,
                "fast_macro": fast_macro,
            }
        )
    return rows


def load_multilingual_collator(collator_path: Path):
    spec = importlib.util.spec_from_file_location(
        "babyconceptlm_multilingual_collator", collator_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load multilingual collator from {collator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_multilingual_rows(
    results_dir: Path,
    collator_path: Path,
    model_name: str,
) -> list[dict[str, float]]:
    collator = load_multilingual_collator(collator_path)
    rows = []
    for exposure_m in EXPOSURES_M:
        scores = collator.load_zeroshot(
            results_dir / f"chck_{exposure_m}M",
            model_name,
            include_server_scored=True,
        )
        flat_scores = {
            task: next(iter(subtasks.values()))
            for task, subtasks in scores.items()
        }
        language_scores = {}
        for language, tasks in MULTILINGUAL_ZERO_SHOT_TASKS.items():
            language_code = MULTILINGUAL_LANGUAGE_CODES[language]
            required = [
                *tasks,
                f"global_piqa_parallel_{language_code}",
                f"global_piqa_nonparallel_{language_code}",
            ]
            missing = [task for task in required if task not in flat_scores]
            if missing:
                raise ValueError(
                    f"Missing {language} tasks at chck_{exposure_m}M: {missing}"
                )
            global_piqa = mean(
                [
                    flat_scores[f"global_piqa_parallel_{language_code}"],
                    flat_scores[f"global_piqa_nonparallel_{language_code}"],
                ]
            )
            language_scores[language] = (
                mean([flat_scores[task] for task in tasks] + [global_piqa]) * 100
            )
        rows.append(
            {
                "exposure_m": exposure_m,
                **language_scores,
                "multilingual_macro": mean(language_scores.values()),
            }
        )
    return rows


def normalized_log_auc(
    rows: list[dict[str, float]], metric: str = "fast_macro"
) -> float:
    x_values = [math.log10(row["exposure_m"]) for row in rows]
    y_values = [row[metric] for row in rows]
    area = sum(
        (y_values[index] + y_values[index + 1])
        / 2
        * (x_values[index + 1] - x_values[index])
        for index in range(len(rows) - 1)
    )
    return area / (x_values[-1] - x_values[0])


def write_csv(
    output_path: Path,
    model_rows: dict[str, list[dict[str, float]]],
    multilingual_rows: list[dict[str, float]],
) -> None:
    metric_names = [
        "blimp",
        "supplement",
        "ewok",
        "entity_tracking",
        "global_piqa",
        "reading",
        "fast_macro",
    ]
    fieldnames = (
        ["exposure_words"]
        + [
            f"strict_{model}_{metric}"
            for model in MODELS
            for metric in metric_names
        ]
        + [
            "multilingual_eng_zero_shot",
            "multilingual_nld_zero_shot",
            "multilingual_zho_zero_shot",
            "multilingual_macro_zero_shot",
        ]
    )
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, exposure_m in enumerate(EXPOSURES_M):
            row: dict[str, float | int] = {"exposure_words": exposure_m * 1_000_000}
            for model in MODELS:
                for metric in metric_names:
                    row[f"strict_{model}_{metric}"] = model_rows[model][index][metric]
            row["multilingual_eng_zero_shot"] = multilingual_rows[index]["eng"]
            row["multilingual_nld_zero_shot"] = multilingual_rows[index]["nld"]
            row["multilingual_zho_zero_shot"] = multilingual_rows[index]["zho"]
            row["multilingual_macro_zero_shot"] = multilingual_rows[index][
                "multilingual_macro"
            ]
            writer.writerow(row)


def write_plot(
    output_dir: Path,
    model_rows: dict[str, list[dict[str, float]]],
    multilingual_rows: list[dict[str, float]],
) -> None:
    import matplotlib.pyplot as plt

    figure, (strict_axis, multilingual_axis) = plt.subplots(
        1, 2, figsize=(9.2, 3.7), sharex=True
    )
    colors = {
        "noDWA-242": "#0072B2",
        "DWA-242": "#56B4E9",
        "noDWA-583": "#D55E00",
        "DWA-583": "#CC79A7",
    }
    linestyles = {
        "noDWA-242": "-",
        "DWA-242": "--",
        "noDWA-583": "-",
        "DWA-583": "--",
    }
    for model, rows in model_rows.items():
        strict_axis.plot(
            [row["exposure_m"] for row in rows],
            [row["fast_macro"] for row in rows],
            marker="o",
            markersize=2.8,
            linewidth=1.8,
            color=colors[model],
            linestyle=linestyles[model],
            label=model,
        )
    strict_axis.legend(fontsize=6.8, frameon=False, ncol=2)
    strict_axis.set_title("(a) English STRICT")
    strict_axis.set_ylabel("STRICT fast-task macro (%)")
    strict_axis.grid(alpha=0.25, linewidth=0.6)

    multilingual_axis.plot(
        [row["exposure_m"] for row in multilingual_rows],
        [row["multilingual_macro"] for row in multilingual_rows],
        marker="o",
        markersize=2.8,
        linewidth=2.1,
        color="#000000",
        label="mean",
    )
    language_styles = {
        "eng": ("#0072B2", "English"),
        "nld": ("#009E73", "Dutch"),
        "zho": ("#E69F00", "Chinese"),
    }
    for language, (color, label) in language_styles.items():
        multilingual_axis.plot(
            [row["exposure_m"] for row in multilingual_rows],
            [row[language] for row in multilingual_rows],
            linewidth=1.3,
            linestyle="--",
            color=color,
            alpha=0.9,
            label=label,
        )
    multilingual_axis.set_title("(b) MULTILINGUAL")
    multilingual_axis.set_ylabel("Fast zero-shot average (%)")
    multilingual_axis.legend(fontsize=6.8, frameon=False, ncol=2)
    multilingual_axis.grid(alpha=0.25, linewidth=0.6)
    for axis in (strict_axis, multilingual_axis):
        axis.set_xscale("log")
        axis.set_xlabel("Word exposure (millions, log scale)")
    figure.tight_layout()
    figure.savefig(output_dir / "exposure_indexed_learning_curves.png", dpi=220)
    figure.savefig(output_dir / "exposure_indexed_learning_curves.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_rows = {
        model: load_model_rows(args.strict_results_dir, directory)
        for model, directory in MODELS.items()
    }
    multilingual_rows = load_multilingual_rows(
        args.multilingual_results_dir,
        args.multilingual_collator,
        args.multilingual_model_name,
    )
    write_csv(
        args.output_dir / "exposure_indexed_learning_curves.csv",
        model_rows,
        multilingual_rows,
    )
    write_plot(args.output_dir, model_rows, multilingual_rows)
    summary = {
        "strict": {
            model: {
                "normalized_log_exposure_auc": normalized_log_auc(rows),
                "score_at_1M": rows[0]["fast_macro"],
                "score_at_10M": rows[EXPOSURES_M.index(10)]["fast_macro"],
                "score_at_100M": rows[EXPOSURES_M.index(100)]["fast_macro"],
                "score_at_200M": rows[EXPOSURES_M.index(200)]["fast_macro"],
                "score_at_500M": rows[EXPOSURES_M.index(500)]["fast_macro"],
                "score_at_1000M": rows[-1]["fast_macro"],
            }
            for model, rows in model_rows.items()
        },
        "multilingual": {
            "normalized_log_exposure_auc": normalized_log_auc(
                multilingual_rows, "multilingual_macro"
            ),
            "score_at_1M": multilingual_rows[0]["multilingual_macro"],
            "score_at_10M": multilingual_rows[EXPOSURES_M.index(10)][
                "multilingual_macro"
            ],
            "score_at_100M": multilingual_rows[EXPOSURES_M.index(100)][
                "multilingual_macro"
            ],
            "score_at_200M": multilingual_rows[EXPOSURES_M.index(200)][
                "multilingual_macro"
            ],
            "score_at_500M": multilingual_rows[EXPOSURES_M.index(500)][
                "multilingual_macro"
            ],
            "score_at_1000M": multilingual_rows[-1]["multilingual_macro"],
        },
    }
    (args.output_dir / "exposure_indexed_learning_curves_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
