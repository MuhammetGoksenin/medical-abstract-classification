"""Audit the raw Medical Abstracts Text Classification dataset.

This script performs read-only checks on the original CSV files in data/raw
and writes audit outputs to reports/. It does not clean, alter, or remove data.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = RAW_DIR / "medical_tc_train.csv"
TEST_PATH = RAW_DIR / "medical_tc_test.csv"
LABELS_PATH = RAW_DIR / "medical_tc_labels.csv"

LABEL_COLUMN = "condition_label"
TEXT_COLUMN = "medical_abstract"


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load a CSV file and return its column names and rows."""
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        columns = list(reader.fieldnames)
        rows = list(reader)
    return columns, rows


def label_sort_key(label: str) -> tuple[int, int | str]:
    """Sort numeric labels numerically, followed by any non-numeric labels."""
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


def missing_counts(
    columns: list[str], rows: list[dict[str, str]]
) -> dict[str, int]:
    """Count empty CSV fields in each column."""
    return {
        column: sum(row.get(column) in (None, "") for row in rows)
        for column in columns
    }


def add_file_summary(
    report: list[str],
    name: str,
    columns: list[str],
    rows: list[dict[str, str]],
) -> None:
    """Append shape, columns, and missing-value information to the report."""
    report.extend(
        [
            name,
            f"  Shape (rows, columns): ({len(rows):,}, {len(columns)})",
            f"  Columns: {', '.join(columns)}",
            "  Missing values (empty fields):",
        ]
    )
    for column, count in missing_counts(columns, rows).items():
        report.append(f"    {column}: {count:,}")
    report.append("")


def add_label_distribution(
    report: list[str],
    name: str,
    rows: list[dict[str, str]],
    label_names: dict[str, str],
) -> None:
    """Append label counts and percentages for one dataset split."""
    counts = Counter(row[LABEL_COLUMN] for row in rows)
    total = len(rows)
    report.append(name)
    report.append("  Label | Condition | Count | Percentage")
    for label in sorted(counts, key=label_sort_key):
        percentage = (counts[label] / total * 100) if total else 0.0
        condition = label_names.get(label, "<unknown label>")
        report.append(
            f"  {label} | {condition} | {counts[label]:,} | {percentage:.2f}%"
        )
    report.append(f"  Total | all conditions | {total:,} | 100.00%")
    report.append("")


def write_conflicts_csv(
    path: Path,
    conflicting_abstracts: list[str],
    labels_by_abstract: dict[str, set[str]],
    train_text_counts: Counter[str],
    test_text_counts: Counter[str],
) -> None:
    """Save one readable row per abstract assigned to multiple labels."""
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = [
            TEXT_COLUMN,
            "different_labels",
            "label_count",
            "combined_row_count",
            "train_row_count",
            "test_row_count",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for abstract in conflicting_abstracts:
            labels = sorted(labels_by_abstract[abstract], key=label_sort_key)
            train_count = train_text_counts[abstract]
            test_count = test_text_counts[abstract]
            writer.writerow(
                {
                    TEXT_COLUMN: abstract,
                    "different_labels": "|".join(labels),
                    "label_count": len(labels),
                    "combined_row_count": train_count + test_count,
                    "train_row_count": train_count,
                    "test_row_count": test_count,
                }
            )


def write_overlap_csv(
    path: Path,
    overlapping_abstracts: list[str],
    train_labels_by_abstract: dict[str, set[str]],
    test_labels_by_abstract: dict[str, set[str]],
    train_text_counts: Counter[str],
    test_text_counts: Counter[str],
) -> None:
    """Save one row per exact abstract shared by train and test."""
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = [
            TEXT_COLUMN,
            "train_labels",
            "test_labels",
            "train_row_count",
            "test_row_count",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for abstract in overlapping_abstracts:
            writer.writerow(
                {
                    TEXT_COLUMN: abstract,
                    "train_labels": "|".join(
                        sorted(
                            train_labels_by_abstract[abstract], key=label_sort_key
                        )
                    ),
                    "test_labels": "|".join(
                        sorted(test_labels_by_abstract[abstract], key=label_sort_key)
                    ),
                    "train_row_count": train_text_counts[abstract],
                    "test_row_count": test_text_counts[abstract],
                }
            )


def main() -> None:
    """Run the dataset audit and write the report files."""
    train_columns, train_rows = load_csv(TRAIN_PATH)
    test_columns, test_rows = load_csv(TEST_PATH)
    label_columns, label_rows = load_csv(LABELS_PATH)

    required_data_columns = {LABEL_COLUMN, TEXT_COLUMN}
    for path, columns in (
        (TRAIN_PATH, train_columns),
        (TEST_PATH, test_columns),
    ):
        missing_columns = required_data_columns.difference(columns)
        if missing_columns:
            raise ValueError(
                f"{path.name} is missing columns: {sorted(missing_columns)}"
            )

    required_label_columns = {LABEL_COLUMN, "condition_name"}
    missing_label_columns = required_label_columns.difference(label_columns)
    if missing_label_columns:
        raise ValueError(
            f"{LABELS_PATH.name} is missing columns: {sorted(missing_label_columns)}"
        )

    label_names = {
        row[LABEL_COLUMN]: row["condition_name"] for row in label_rows
    }

    # Train and test are combined in memory for audit calculations only.
    combined_rows = train_rows + test_rows
    combined_text_counts = Counter(row[TEXT_COLUMN] for row in combined_rows)
    train_text_counts = Counter(row[TEXT_COLUMN] for row in train_rows)
    test_text_counts = Counter(row[TEXT_COLUMN] for row in test_rows)

    labels_by_abstract: dict[str, set[str]] = defaultdict(set)
    train_labels_by_abstract: dict[str, set[str]] = defaultdict(set)
    test_labels_by_abstract: dict[str, set[str]] = defaultdict(set)
    for row in combined_rows:
        labels_by_abstract[row[TEXT_COLUMN]].add(row[LABEL_COLUMN])
    for row in train_rows:
        train_labels_by_abstract[row[TEXT_COLUMN]].add(row[LABEL_COLUMN])
    for row in test_rows:
        test_labels_by_abstract[row[TEXT_COLUMN]].add(row[LABEL_COLUMN])

    repeated_abstracts = sorted(
        (text for text, count in combined_text_counts.items() if count > 1)
    )
    conflicting_abstracts = sorted(
        text for text, labels in labels_by_abstract.items() if len(labels) > 1
    )
    overlapping_abstracts = sorted(set(train_text_counts) & set(test_text_counts))

    repeated_rows = sum(combined_text_counts[text] for text in repeated_abstracts)
    duplicate_occurrences = len(combined_rows) - len(combined_text_counts)
    overlap_train_rows = sum(train_text_counts[text] for text in overlapping_abstracts)
    overlap_test_rows = sum(test_text_counts[text] for text in overlapping_abstracts)

    report: list[str] = [
        "MEDICAL ABSTRACT DISEASE CLASSIFICATION - DATASET AUDIT",
        "=" * 56,
        "",
        "Scope and definitions",
        "---------------------",
        "- Raw files were read from data/raw/ and were not modified.",
        "- Train and test were combined in memory only for audit calculations.",
        "- All text comparisons are exact (no case, whitespace, or punctuation normalization).",
        "- Missing values are CSV fields that are empty.",
        "- A repeated abstract is an exact text value appearing in more than one combined row.",
        "",
        "FILE SUMMARIES",
        "==============",
        "",
    ]
    add_file_summary(report, TRAIN_PATH.name, train_columns, train_rows)
    add_file_summary(report, TEST_PATH.name, test_columns, test_rows)
    add_file_summary(report, LABELS_PATH.name, label_columns, label_rows)

    report.extend(["LABEL DISTRIBUTIONS", "===================", ""])
    add_label_distribution(report, "Training set", train_rows, label_names)
    add_label_distribution(report, "Test set", test_rows, label_names)
    add_label_distribution(report, "Combined audit set", combined_rows, label_names)

    report.extend(
        [
            "EXACT-TEXT DUPLICATION AND CONFLICTS",
            "====================================",
            "",
            f"Combined rows: {len(combined_rows):,}",
            f"Total unique medical abstracts: {len(combined_text_counts):,}",
            f"Repeated unique abstract texts: {len(repeated_abstracts):,}",
            f"Rows belonging to repeated abstract groups: {repeated_rows:,}",
            f"Duplicate occurrences beyond the first: {duplicate_occurrences:,}",
            "Exact identical abstracts with more than one different label: "
            f"{len(conflicting_abstracts):,}",
            "",
            "TRAIN/TEST EXACT-TEXT OVERLAP",
            "=============================",
            "",
            f"Unique abstract texts appearing in both train and test: {len(overlapping_abstracts):,}",
            f"Training rows whose abstract appears in test: {overlap_train_rows:,}",
            f"Test rows whose abstract appears in train: {overlap_test_rows:,}",
            "",
            "SUPPORTING FILES",
            "================",
            "",
            "- reports/conflicting_abstracts.csv",
            "- reports/train_test_overlap.csv",
            "",
        ]
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "dataset_audit.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")
    write_conflicts_csv(
        REPORTS_DIR / "conflicting_abstracts.csv",
        conflicting_abstracts,
        labels_by_abstract,
        train_text_counts,
        test_text_counts,
    )
    write_overlap_csv(
        REPORTS_DIR / "train_test_overlap.csv",
        overlapping_abstracts,
        train_labels_by_abstract,
        test_labels_by_abstract,
        train_text_counts,
        test_text_counts,
    )

    print(f"Audit complete: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
