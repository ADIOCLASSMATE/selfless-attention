#!/usr/bin/env python3
"""
Summarize lm-eval result JSON files for an evaluation output directory.

Pass a directory such as output_eval. For each immediate child directory, the
script recursively finds results_*.json files, picks the newest timestamped
file, builds a combined analysis JSON, and writes SVG bar charts.
"""

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_TASKS = [
    "lambada_openai",
    "wikitext",
    "hellaswag",
    "copa",
    "piqa",
    "arc_easy",
    "openbookqa",
    "winogrande",
    "boolq",
    "sciq",
    "truthfulqa_mc1",
    "truthfulqa_mc2",
    "gpqa_diamond_zeroshot",
    "super-glue-lm-eval-v1",
    "arc_challenge",
    "paloma_c4_en",
    "paloma_falcon-refinedweb",
    "paloma_wikitext_103",
]

SUPER_GLUE_SUBTASKS = [
    "boolq",
    "cb",
    "copa",
    "multirc",
    "record",
    "sglue_rte",
    "wic",
    "wsc",
]

TRUTHFULQA_TASKS = ["truthfulqa_mc1", "truthfulqa_mc2"]

ACCURACY_TASKS = [
    "hellaswag",
    "copa",
    "piqa",
    "arc_easy",
    "openbookqa",
    "winogrande",
    "boolq",
    "sciq",
    "gpqa_diamond_zeroshot",
    "arc_challenge",
]

PALOMA_TASKS = [
    "paloma_c4_en",
    "paloma_falcon-refinedweb",
    "paloma_wikitext_103",
]

TASK_METRIC_PRIORITY = {
    # LM / PPL tasks. Do not let lambada acc=0 become its primary score.
    "lambada_openai": ["perplexity,none", "acc,none"],
    "wikitext": ["word_perplexity,none", "byte_perplexity,none", "bits_per_byte,none"],
    "paloma_c4_en": ["word_perplexity,none", "byte_perplexity,none", "bits_per_byte,none"],
    "paloma_falcon-refinedweb": ["word_perplexity,none", "byte_perplexity,none", "bits_per_byte,none"],
    "paloma_wikitext_103": ["word_perplexity,none", "byte_perplexity,none", "bits_per_byte,none"],
    # SuperGLUE subtasks with multi-metric conventions.
    "record": ["f1,none", "em,none"],
    "cb": ["acc,none", "f1,none"],
    # Multiple-choice tasks where normalized accuracy is the standard choice.
    "hellaswag": ["acc_norm,none", "acc,none"],
    "piqa": ["acc_norm,none", "acc,none"],
    "arc_easy": ["acc_norm,none", "acc,none"],
    "arc_challenge": ["acc_norm,none", "acc,none"],
    "openbookqa": ["acc_norm,none", "acc,none"],
    "sciq": ["acc_norm,none", "acc,none"],
    "gpqa_diamond_zeroshot": ["acc_norm,none", "acc,none"],
}

DEFAULT_METRIC_PRIORITY = ["acc_norm,none", "acc,none", "f1,none", "em,none"]

LOWER_IS_BETTER_METRICS = {
    "perplexity,none",
    "word_perplexity,none",
    "byte_perplexity,none",
    "bits_per_byte,none",
}

SUMMARY_METRIC_DIRECTIONS = {
    "standalone_accuracy_macro_avg": True,
    "truthfulqa_avg": True,
    "super_glue_avg": True,
    "paloma_word_perplexity_avg": False,
    "paloma_bits_per_byte_avg": False,
    "lm_word_or_token_perplexity_avg": False,
}

RESULTS_RE = re.compile(r"results_(?P<timestamp>.+)\.json$")


def warn(message: str) -> None:
    print(f"警告: {message}", file=sys.stderr)


def mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    return statistics.fmean(values) if values else None


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_tasks(tasks_arg: Optional[str]) -> List[str]:
    if not tasks_arg:
        return DEFAULT_TASKS
    return [task.strip() for task in tasks_arg.split(",") if task.strip()]


def numeric_metrics(task_data: Dict[str, Any]) -> Dict[str, float]:
    return {
        key: float(value)
        for key, value in task_data.items()
        if key != "alias" and is_number(value)
    }


def get_metric_priority(task_name: str) -> List[str]:
    return TASK_METRIC_PRIORITY.get(task_name, DEFAULT_METRIC_PRIORITY)


def get_primary_metric(
    task_name: str,
    task_data: Dict[str, Any],
) -> Tuple[Optional[str], Optional[float]]:
    for metric in get_metric_priority(task_name):
        value = task_data.get(metric)
        if is_number(value):
            return metric, float(value)

    metrics = numeric_metrics(task_data)
    for metric, value in metrics.items():
        if "_stderr" not in metric:
            warn(f"任务 {task_name} 没有匹配到配置的主指标，回退到 {metric}")
            return metric, value

    warn(f"任务 {task_name} 没有找到可用数值指标")
    return None, None


def higher_is_better(metric_name: Optional[str]) -> Optional[bool]:
    if metric_name is None:
        return None
    return metric_name not in LOWER_IS_BETTER_METRICS


def summarize_task(task_name: str, results: Dict[str, Any]) -> Dict[str, Any]:
    if task_name not in results:
        warn(f"任务 {task_name} 未找到")
        return {}

    task_data = results[task_name]
    primary_metric, primary_score = get_primary_metric(task_name, task_data)

    return {
        "alias": task_data.get("alias", task_name),
        "primary_metric": primary_metric,
        "primary_score": primary_score,
        "higher_is_better": higher_is_better(primary_metric),
        "metrics": numeric_metrics(task_data),
    }


def summarize_group(
    group_name: str,
    subtasks: List[str],
    results: Dict[str, Any],
) -> Dict[str, Any]:
    subtask_scores = {}

    for task_name in subtasks:
        if task_name not in results:
            warn(f"{group_name} 子任务 {task_name} 未找到")
            continue

        metric, score = get_primary_metric(task_name, results[task_name])
        if score is None:
            continue

        subtask_scores[task_name] = {
            "primary_metric": metric,
            "primary_score": score,
            "higher_is_better": higher_is_better(metric),
        }

    return {
        "subtasks": subtask_scores,
        "average": mean(
            item["primary_score"]
            for item in subtask_scores.values()
            if item["higher_is_better"]
        ),
    }


def metric_average(
    task_names: List[str],
    metric_name: str,
    results: Dict[str, Any],
) -> Optional[float]:
    return mean(
        results[task_name][metric_name]
        for task_name in task_names
        if task_name in results and is_number(results[task_name].get(metric_name))
    )


def build_summary(results: Dict[str, Any], requested_tasks: List[str]) -> Dict[str, Any]:
    has_super_glue = "super-glue-lm-eval-v1" in requested_tasks
    accuracy_tasks = [
        task_name
        for task_name in ACCURACY_TASKS
        if task_name in requested_tasks
        and (not has_super_glue or task_name not in SUPER_GLUE_SUBTASKS)
    ]
    tqa_tasks = [task_name for task_name in TRUTHFULQA_TASKS if task_name in requested_tasks]
    paloma_tasks = [task_name for task_name in PALOMA_TASKS if task_name in requested_tasks]
    lm_ppl_tasks = [
        task_name
        for task_name in ["lambada_openai", "wikitext"]
        if task_name in requested_tasks
    ]

    accuracy_scores = []
    for task_name in accuracy_tasks:
        if task_name not in results:
            continue
        metric, score = get_primary_metric(task_name, results[task_name])
        if score is not None and higher_is_better(metric):
            accuracy_scores.append(score)

    tqa_scores = []
    for task_name in tqa_tasks:
        if task_name not in results:
            continue
        metric, score = get_primary_metric(task_name, results[task_name])
        if score is not None and higher_is_better(metric):
            tqa_scores.append(score)

    return {
        "standalone_accuracy_macro_avg": mean(accuracy_scores),
        "truthfulqa_avg": mean(tqa_scores),
        "paloma_word_perplexity_avg": metric_average(
            paloma_tasks,
            "word_perplexity,none",
            results,
        ),
        "paloma_bits_per_byte_avg": metric_average(
            paloma_tasks,
            "bits_per_byte,none",
            results,
        ),
        "lm_word_or_token_perplexity_avg": mean(
            score
            for task_name in lm_ppl_tasks
            if task_name in results
            for metric, score in [get_primary_metric(task_name, results[task_name])]
            if score is not None and not higher_is_better(metric)
        ),
    }


def build_single_result(results: Dict[str, Any], requested_tasks: List[str]) -> Dict[str, Any]:
    has_super_glue = "super-glue-lm-eval-v1" in requested_tasks
    expanded_tasks = [
        task_name
        for task_name in requested_tasks
        if task_name != "super-glue-lm-eval-v1"
    ]

    output = {
        "tasks": {},
        "groups": {},
        "summary": build_summary(results, requested_tasks),
    }

    for task_name in dict.fromkeys(expanded_tasks):
        if has_super_glue and task_name in SUPER_GLUE_SUBTASKS:
            continue
        if task_name in results:
            output["tasks"][task_name] = summarize_task(task_name, results)
        else:
            warn(f"任务 {task_name} 未找到")

    if "super-glue-lm-eval-v1" in requested_tasks:
        output["groups"]["super-glue-lm-eval-v1"] = summarize_group(
            "super-glue-lm-eval-v1",
            SUPER_GLUE_SUBTASKS,
            results,
        )
        output["summary"]["super_glue_avg"] = output["groups"]["super-glue-lm-eval-v1"][
            "average"
        ]

    if any(task in requested_tasks for task in TRUTHFULQA_TASKS):
        output["groups"]["truthfulqa"] = summarize_group(
            "truthfulqa",
            TRUTHFULQA_TASKS,
            results,
        )

    if any(task in requested_tasks for task in PALOMA_TASKS):
        requested_paloma_tasks = [
            task_name for task_name in PALOMA_TASKS if task_name in requested_tasks
        ]
        output["groups"]["paloma"] = {
            "tasks": requested_paloma_tasks,
            "word_perplexity_avg": output["summary"]["paloma_word_perplexity_avg"],
            "bits_per_byte_avg": output["summary"]["paloma_bits_per_byte_avg"],
            "higher_is_better": False,
        }

    output["all_tasks"] = {
        task_name: summarize_task(task_name, results)
        for task_name in sorted(results)
    }
    return output


def parse_result_timestamp(path: Path) -> datetime:
    match = RESULTS_RE.match(path.name)
    if match:
        timestamp = match.group("timestamp")
        for timestamp_format in (None, "%Y-%m-%dT%H-%M-%S.%f", "%Y-%m-%dT%H-%M-%S"):
            try:
                if timestamp_format is None:
                    return datetime.fromisoformat(timestamp)
                return datetime.strptime(timestamp, timestamp_format)
            except ValueError:
                continue
        warn(f"无法解析 results 时间戳，改用文件修改时间: {path}")
    return datetime.fromtimestamp(path.stat().st_mtime)


def find_latest_result_files(input_dir: Path, output_dir: Path) -> Dict[str, Path]:
    result_files = {}
    output_dir = output_dir.resolve()

    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.resolve() == output_dir:
            continue

        files = list(child.rglob("results_*.json"))
        if not files:
            continue

        result_files[child.name] = max(
            files,
            key=lambda path: (parse_result_timestamp(path), path.stat().st_mtime),
        )

    if not result_files:
        files = list(input_dir.rglob("results_*.json"))
        if files:
            result_files[input_dir.name] = max(
                files,
                key=lambda path: (parse_result_timestamp(path), path.stat().st_mtime),
            )

    return result_files


def load_results(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "results" not in data:
        raise ValueError(f"JSON 文件中没有找到 'results' 字段: {path}")
    return data["results"]


def collect_comparison(runs: Dict[str, Any]) -> Dict[str, Any]:
    summary_metrics: Dict[str, Any] = {}
    task_scores: Dict[str, Any] = {}

    for run_name, run in runs.items():
        for metric_name, score in run["summary"].items():
            if not is_number(score):
                continue
            summary_metrics.setdefault(
                metric_name,
                {
                    "higher_is_better": SUMMARY_METRIC_DIRECTIONS.get(metric_name),
                    "scores": [],
                },
            )
            summary_metrics[metric_name]["scores"].append(
                {"run": run_name, "score": float(score)}
            )

        for task_name, task in run["all_tasks"].items():
            score = task.get("primary_score")
            if not is_number(score):
                continue
            entry = task_scores.setdefault(
                task_name,
                {
                    "alias": task.get("alias", task_name),
                    "primary_metric": task.get("primary_metric"),
                    "higher_is_better": task.get("higher_is_better"),
                    "scores": [],
                },
            )
            entry["scores"].append({"run": run_name, "score": float(score)})

    for group in list(summary_metrics.values()) + list(task_scores.values()):
        higher = group.get("higher_is_better")
        reverse = True if higher is not False else False
        group["scores"].sort(key=lambda item: item["score"], reverse=reverse)

    return {
        "summary_metrics": summary_metrics,
        "task_scores": task_scores,
    }


def format_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 10:
        return f"{value:.3f}"
    return f"{value:.4f}"


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("_") or "chart"


def bar_color(higher_is_better: Optional[bool]) -> str:
    if higher_is_better is False:
        return "#c2410c"
    if higher_is_better is True:
        return "#2563eb"
    return "#64748b"


def write_bar_svg(
    path: Path,
    title: str,
    values: List[Dict[str, Any]],
    *,
    higher_is_better: Optional[bool],
    metric_name: Optional[str] = None,
) -> None:
    if not values:
        return

    longest_label = max(len(str(item["run"])) for item in values)
    label_width = min(max(240, int(longest_label * 7.2) + 28), 430)
    bar_width = 520
    right_width = 90
    row_height = 28
    top = 72
    bottom = 30
    width = label_width + bar_width + right_width + 40
    height = top + len(values) * row_height + bottom
    max_value = max(abs(item["score"]) for item in values) or 1.0
    color = bar_color(higher_is_better)
    subtitle = ""
    if metric_name:
        direction = "higher is better" if higher_is_better else "lower is better"
        subtitle = f"{metric_name} ({direction})"

    rows = []
    for index, item in enumerate(values):
        y = top + index * row_height
        label = escape(str(item["run"]))
        score = float(item["score"])
        length = max(2.0, abs(score) / max_value * bar_width)
        rows.append(
            f'<text x="18" y="{y + 18}" class="label">{label}</text>'
            f'<rect x="{label_width}" y="{y + 5}" width="{length:.2f}" '
            f'height="16" rx="3" fill="{color}"/>'
            f'<text x="{label_width + length + 8:.2f}" y="{y + 18}" '
            f'class="value">{format_value(score)}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <style>
    .title {{ font: 700 20px Arial, sans-serif; fill: #0f172a; }}
    .subtitle {{ font: 13px Arial, sans-serif; fill: #475569; }}
    .label {{ font: 12px Arial, sans-serif; fill: #0f172a; }}
    .value {{ font: 12px Arial, sans-serif; fill: #334155; }}
    .axis {{ stroke: #cbd5e1; stroke-width: 1; }}
  </style>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="18" y="30" class="title">{escape(title)}</text>
  <text x="18" y="52" class="subtitle">{escape(subtitle)}</text>
  <line x1="{label_width}" y1="{top - 8}" x2="{label_width}" y2="{height - bottom + 2}" class="axis"/>
  {''.join(rows)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def write_index_html(path: Path, chart_paths: Dict[str, List[str]], analysis_json: str) -> None:
    sections = []
    for section_title, paths in chart_paths.items():
        links = "\n".join(
            f'<li><a href="{escape(chart)}">{escape(Path(chart).stem)}</a></li>'
            for chart in paths
        )
        sections.append(f"<h2>{escape(section_title)}</h2><ul>{links}</ul>")

    html = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Evaluation Analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #0f172a; }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    li {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <h1>Evaluation Analysis</h1>
  <p>Analysis JSON: <a href="{escape(analysis_json)}">{escape(analysis_json)}</a></p>
  {''.join(sections)}
</body>
</html>
'''
    path.write_text(html, encoding="utf-8")


def write_visualizations(analysis: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    plots_dir = output_dir / "plots"
    summary_dir = plots_dir / "summary"
    tasks_dir = plots_dir / "tasks"
    summary_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(parents=True, exist_ok=True)

    summary_charts = []
    for metric_name, group in analysis["comparison"]["summary_metrics"].items():
        scores = group["scores"]
        if not scores:
            continue
        chart_path = summary_dir / f"{safe_filename(metric_name)}.svg"
        write_bar_svg(
            chart_path,
            f"Summary: {metric_name}",
            scores,
            higher_is_better=group.get("higher_is_better"),
            metric_name=metric_name,
        )
        summary_charts.append(chart_path.relative_to(output_dir).as_posix())

    task_charts = []
    for task_name, group in analysis["comparison"]["task_scores"].items():
        scores = group["scores"]
        if not scores:
            continue
        metric_name = group.get("primary_metric")
        chart_path = tasks_dir / f"{safe_filename(task_name)}.svg"
        write_bar_svg(
            chart_path,
            f"Task: {task_name}",
            scores,
            higher_is_better=group.get("higher_is_better"),
            metric_name=metric_name,
        )
        task_charts.append(chart_path.relative_to(output_dir).as_posix())

    index_path = output_dir / "index.html"
    analysis_json = "analysis.json"
    write_index_html(
        index_path,
        {
            "Summary Bars": summary_charts,
            "Task Bars": task_charts,
        },
        analysis_json,
    )

    return {
        "index_html": index_path.as_posix(),
        "summary_bar_charts": summary_charts,
        "task_bar_charts": task_charts,
    }


def build_directory_analysis(
    input_dir: Path,
    output_dir: Path,
    requested_tasks: List[str],
) -> Dict[str, Any]:
    latest_files = find_latest_result_files(input_dir, output_dir)
    if not latest_files:
        raise FileNotFoundError(f"没有在目录中找到 results_*.json: {input_dir}")

    runs = {}
    for run_name, result_file in latest_files.items():
        results = load_results(result_file)
        run_output = build_single_result(results, requested_tasks)
        runs[run_name] = {
            "result_file": result_file.as_posix(),
            "result_timestamp": parse_result_timestamp(result_file).isoformat(),
            **run_output,
        }

    analysis = {
        "input_dir": input_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "requested_tasks": requested_tasks,
        "run_count": len(runs),
        "runs": runs,
        "comparison": {},
        "visualizations": {},
    }
    analysis["comparison"] = collect_comparison(runs)
    analysis["visualizations"] = write_visualizations(analysis, output_dir)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="解析 output_eval 目录并生成分析 JSON 与可视化")
    parser.add_argument("input_dir", type=Path, help="评测输出根目录，例如 output_eval")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="分析输出目录，默认写入 <input_dir>/analysis",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="逗号分隔的任务列表；默认使用当前 eval 脚本中的 TASKS",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.is_dir():
        print(f"错误: {input_dir} 不是目录", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or (input_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        analysis = build_directory_analysis(input_dir, output_dir, parse_tasks(args.tasks))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)

    analysis_path = output_dir / "analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"已处理 {analysis['run_count']} 个评测目录")
    print(f"分析 JSON: {analysis_path}")
    print(f"可视化入口: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
