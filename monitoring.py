"""Monitoring checks for the credit default model, shared by the notebook and the CLI.

A batch is a prediction log: the model's input features, its predicted
probability of default, and (once known) the actual outcome.

    python monitoring.py prediction_logs/2006-02.csv --report reports/2006-02.html
    python monitoring.py prediction_logs/2006-01.csv --no-labels

Exits with status 1 when any check fails, so it can gate a pipeline step.
"""

import argparse
import os
import sys
import warnings

os.environ.setdefault("DO_NOT_TRACK", "1")
warnings.filterwarnings("ignore", category=SyntaxWarning)

import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import DatasetMissingValueCount, DriftedColumnsCount, MeanValue, RocAuc, ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset
from evidently.tests import eq, gte, lte

TARGET = "default"
PREDICTION = "prediction"
AMOUNTS = [f"bill_amt{i}" for i in range(1, 7)] + [f"pay_amt{i}" for i in range(1, 7)]
NUMERICAL = ["limit_bal", "age"] + AMOUNTS
CATEGORICAL = ["sex", "education", "marriage"] + [f"pay_{i}" for i in range(1, 7)]
FEATURES = NUMERICAL + CATEGORICAL
REFERENCE_PATH = "prediction_logs/reference.csv"

# Rules 1 to 3 work as soon as predictions are made; 4 and 5 need the actual outcome.
THRESHOLDS = {
    "drifted_feature_share": 0.25,
    "prediction_drift": 0.1,
    "missing_values": 0,
    "roc_auc": 0.75,
    "default_rate": 0.30,
}
LABEL_RULES = ["roc_auc", "default_rate"]


def data_definition(with_labels):
    return DataDefinition(
        numerical_columns=NUMERICAL + [PREDICTION],
        categorical_columns=CATEGORICAL + ([TARGET] if with_labels else []),
        classification=[BinaryClassification(target=TARGET, prediction_probas=PREDICTION)] if with_labels else None,
    )


def build_report(with_labels):
    t = THRESHOLDS
    metrics = [
        DriftedColumnsCount(columns=FEATURES, share_tests=[lte(t["drifted_feature_share"])]),
        ValueDrift(column=PREDICTION, tests=[lte(t["prediction_drift"])]),
        DatasetMissingValueCount(tests=[eq(t["missing_values"])]),
    ]
    if with_labels:
        metrics += [
            RocAuc(tests=[gte(t["roc_auc"])]),
            MeanValue(column=TARGET, tests=[lte(t["default_rate"])]),
        ]
    # Presets add plots for the HTML report. Their own tests only run with
    # Report(include_tests=True), so the five rules above are the only tests.
    metrics.append(DataDriftPreset(columns=FEATURES))
    if with_labels:
        metrics.append(ClassificationPreset())
    return Report(metrics)


def rule_for(params):
    kind = params["type"].split(":")[-1]
    if kind == "DriftedColumnsCount":
        return "drifted_feature_share"
    if kind == "ValueDrift" and params.get("column") == PREDICTION:
        return "prediction_drift"
    if kind == "DatasetMissingValueCount":
        return "missing_values"
    if kind == "RocAuc":
        return "roc_auc"
    if kind == "MeanValue" and params.get("column") == TARGET:
        return "default_rate"
    return None


def summarize(snapshot):
    result = snapshot.dict()
    values, column_drift = {}, {}
    for metric in result["metrics"]:
        params = metric["config"]
        kind = params["type"].split(":")[-1]
        value = metric["value"]
        if kind == "ValueDrift" and params["column"] in FEATURES:
            column_drift[params["column"]] = round(float(value) / params["threshold"], 4)
        rule = rule_for(params)
        if rule is None or rule in values:
            continue
        if rule == "drifted_feature_share":
            values["drifted_features"] = int(value["count"])
            values[rule] = round(float(value["share"]), 4)
        elif rule == "missing_values":
            values[rule] = int(value["count"])
        else:
            values[rule] = round(float(value), 4)

    checks = {}
    for test in result["tests"]:
        rule = rule_for(test["metric_config"]["params"])
        if rule and rule not in checks:
            checks[rule] = "pass" if "SUCCESS" in str(test["status"]) else "fail"
    return {
        "with_labels": any(rule in checks for rule in LABEL_RULES),
        "values": values,
        "checks": checks,
        "alerts": [rule for rule, status in checks.items() if status == "fail"],
        "drifted_columns": [c for c, ratio in column_drift.items() if ratio >= 1],
        "column_drift_ratio": column_drift,
    }


def run_checks(current, reference, timestamp=None, with_labels=None):
    """Compare a batch with the reference period and return (snapshot, summary)."""
    if with_labels is None:
        with_labels = TARGET in current.columns
    if not with_labels:
        current = current.drop(columns=[TARGET], errors="ignore")
        reference = reference.drop(columns=[TARGET], errors="ignore")
    definition = data_definition(with_labels)
    snapshot = build_report(with_labels).run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
        timestamp=timestamp,
    )
    return snapshot, summarize(snapshot)


def main():
    parser = argparse.ArgumentParser(description="Run the monitoring checks on one batch of prediction logs.")
    parser.add_argument("batch", help="CSV with the features, the prediction and optionally the actual outcome")
    parser.add_argument("--reference", default=REFERENCE_PATH)
    parser.add_argument("--report", help="save the Evidently HTML report to this path")
    parser.add_argument("--no-labels", action="store_true", help="skip the checks that need the actual outcome")
    args = parser.parse_args()

    current = pd.read_csv(args.batch)
    reference = pd.read_csv(args.reference)
    snapshot, summary = run_checks(current, reference, with_labels=False if args.no_labels else None)
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        snapshot.save_html(args.report)

    print(f"{args.batch}: {len(current)} rows, labels {'used' if summary['with_labels'] else 'not used'}")
    for rule, status in summary["checks"].items():
        print(f"  {status.upper():4}  {rule:<22} {summary['values'][rule]:<8} threshold {THRESHOLDS[rule]}")
    if summary["drifted_columns"]:
        print(f"  drifted features: {', '.join(summary['drifted_columns'])}")
    print("ALERT" if summary["alerts"] else "OK", "".join(f" {a}" for a in summary["alerts"]))
    return 1 if summary["alerts"] else 0


if __name__ == "__main__":
    sys.exit(main())
