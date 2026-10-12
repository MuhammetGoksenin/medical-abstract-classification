"""Create a reproducible single-label medical abstracts dataset.

The original train and test files are combined because their original split is
discarded. Abstract text is compared and written exactly as loaded: no text
normalization or other linguistic preprocessing is performed.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = RAW_DIR / "medical_tc_train.csv"
TEST_PATH = RAW_DIR / "medical_tc_test.csv"
LABELS_PATH = RAW_DIR / "medical_tc_labels.csv"

CLEAN_DATA_PATH = PROCESSED_DIR / "medical_abstracts_clean.csv"
CLEANING_REPORT_PATH = REPORTS_DIR / "cleaning_report.txt"
REMOVED_AMBIGUOUS_PATH = REPORTS_DIR / "removed_ambiguous_abstracts.csv"

LABEL_COLUMN = "condition_label"
TEXT_COLUMN = "medical_abstract"
NAME_COLUMN = "condition_name"
EXPECTED_LABELS = {"1", "2", "3", "4", "5"}
OUTPUT_COLUMNS = ["abstract_id", TEXT_COLUMN, LABEL_COLUMN, NAME_COLUMN]


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load a CSV file without changing any field values."""
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        columns = list(reader.fieldnames)
        rows = list(reader)
    return columns, rows


def require_columns(path: Path, columns: list[str], required: set[str]) -> None:
    """Fail with a useful message when an input schema is incomplete."""
    missing_columns = required.difference(columns)
    if missing_columns:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing_columns)}")


def make_abstract_id(abstract: str) -> str:
    """Return a stable 16-character SHA-256 ID for the exact abstract text."""
    return hashlib.sha256(abstract.encode("utf-8")).hexdigest()[:16]


def label_sort_key(label: str) -> tuple[int, int | str]:
    """Sort numeric labels numerically, followed by non-numeric values."""
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


def missing_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    """Count empty fields in every final output column."""
    return {
        column: sum(row.get(column) in (None, "") for row in rows)
        for column in OUTPUT_COLUMNS
    }


def labels_by_abstract(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    """Collect the unique labels associated with every exact abstract text."""
    grouped_labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped_labels[row[TEXT_COLUMN]].add(row[LABEL_COLUMN])
    return grouped_labels


def add_distribution(
    report: list[str],
    title: str,
    counts: Counter[str],
    total: int,
    label_names: dict[str, str],
) -> None:
    """Append a readable class-count and percentage table to the report."""
    report.extend([title, "  Label | Condition | Count | Percentage"])
    for label in sorted(counts, key=label_sort_key):
        percentage = (counts[label] / total * 100) if total else 0.0
        report.append(
            f"  {label} | {label_names[label]} | {counts[label]:,} | "
            f"{percentage:.2f}%"
        )
    report.extend([f"  Total | all conditions | {total:,} | 100.00%", ""])


def assert_final_dataset(final_rows: list[dict[str, str]]) -> None:
    """Enforce all single-label dataset invariants requested for this phase."""
    final_abstracts = [row[TEXT_COLUMN] for row in final_rows]
    final_grouped_labels = labels_by_abstract(final_rows)
    final_labels = {row[LABEL_COLUMN] for row in final_rows}
    final_ids = [row["abstract_id"] for row in final_rows]

    assert len(final_abstracts) == len(set(final_abstracts)), (
        "Final dataset contains duplicate medical_abstract values."
    )
    assert all(row[TEXT_COLUMN] != "" for row in final_rows), (
        "Final dataset contains missing abstract text."
    )
    assert all(row[LABEL_COLUMN] != "" for row in final_rows), (
        "Final dataset contains missing labels."
    )
    assert all(len(labels) == 1 for labels in final_grouped_labels.values()), (
        "At least one final abstract maps to multiple labels."
    )
    assert final_labels == EXPECTED_LABELS, (
        f"Expected labels {sorted(EXPECTED_LABELS)}, found {sorted(final_labels)}."
    )
    assert all(row[NAME_COLUMN] != "" for row in final_rows), (
        "Final dataset contains a missing condition name."
    )
    assert len(final_ids) == len(set(final_ids)), (
        "A shortened SHA-256 abstract_id collision was detected."
    )


def write_clean_dataset(rows: list[dict[str, str]]) -> None:
    """Write the final single-label dataset."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with CLEAN_DATA_PATH.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_removed_ambiguous(
    ambiguous_abstracts: list[str],
    label_names: dict[str, str],
    combined_rows: list[dict[str, str]],
    train_text_counts: Counter[str],
    test_text_counts: Counter[str],
) -> None:
    """Write one transparent audit row for every excluded abstract group."""
    per_text_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in combined_rows:
        per_text_label_counts[row[TEXT_COLUMN]][row[LABEL_COLUMN]] += 1

    fieldnames = [
        "abstract_id",
        TEXT_COLUMN,
        "condition_labels",
        "condition_names",
        "rows_per_label",
        "original_row_count",
        "train_row_count",
        "test_row_count",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with REMOVED_AMBIGUOUS_PATH.open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for abstract in ambiguous_abstracts:
            label_counts = per_text_label_counts[abstract]
            labels = sorted(label_counts, key=label_sort_key)
            writer.writerow(
                {
                    "abstract_id": make_abstract_id(abstract),
                    TEXT_COLUMN: abstract,
                    "condition_labels": "|".join(labels),
                    "condition_names": "|".join(label_names[label] for label in labels),
                    "rows_per_label": "|".join(
                        f"{label}:{label_counts[label]}" for label in labels
                    ),
                    "original_row_count": sum(label_counts.values()),
                    "train_row_count": train_text_counts[abstract],
                    "test_row_count": test_text_counts[abstract],
                }
            )


def main() -> None:
    """Build and validate the cleaned single-label dataset."""
    train_columns, train_rows = load_csv(TRAIN_PATH)
    test_columns, test_rows = load_csv(TEST_PATH)
    label_columns, label_rows = load_csv(LABELS_PATH)

    require_columns(TRAIN_PATH, train_columns, {LABEL_COLUMN, TEXT_COLUMN})
    require_columns(TEST_PATH, test_columns, {LABEL_COLUMN, TEXT_COLUMN})
    require_columns(LABELS_PATH, label_columns, {LABEL_COLUMN, NAME_COLUMN})

    label_names = {
        row[LABEL_COLUMN]: row[NAME_COLUMN]
        for row in label_rows
        if row[LABEL_COLUMN] != ""
    }
    assert set(label_names) == EXPECTED_LABELS, (
        f"Label file must define exactly {sorted(EXPECTED_LABELS)}."
    )
    assert all(label_names.values()), "Label file contains a missing condition name."

    # The original split is intentionally discarded for dataset preparation.
    combined_rows = train_rows + test_rows
    grouped_labels = labels_by_abstract(combined_rows)

    ambiguous_texts = {
        abstract for abstract, labels in grouped_labels.items() if len(labels) > 1
    }
    single_label_texts = {
        abstract for abstract, labels in grouped_labels.items() if len(labels) == 1
    }

    # Create exactly one row per unambiguous exact abstract text.
    final_rows = []
    for abstract in single_label_texts:
        label = next(iter(grouped_labels[abstract]))
        if label not in label_names:
            raise ValueError(f"Unknown condition label in source data: {label!r}")
        final_rows.append(
            {
                "abstract_id": make_abstract_id(abstract),
                TEXT_COLUMN: abstract,
                LABEL_COLUMN: label,
                NAME_COLUMN: label_names[label],
            }
        )

    # Hash ordering makes the output stable even if source row order changes.
    final_rows.sort(key=lambda row: (row["abstract_id"], row[TEXT_COLUMN]))
    ambiguous_abstracts = sorted(
        ambiguous_texts, key=lambda text: (make_abstract_id(text), text)
    )

    # Required checks run before any cleaned outputs are written.
    assert_final_dataset(final_rows)

    original_text_counts = Counter(row[TEXT_COLUMN] for row in combined_rows)
    train_text_counts = Counter(row[TEXT_COLUMN] for row in train_rows)
    test_text_counts = Counter(row[TEXT_COLUMN] for row in test_rows)
    ambiguous_original_rows = sum(
        original_text_counts[abstract] for abstract in ambiguous_texts
    )

    before_counts = Counter(row[LABEL_COLUMN] for row in combined_rows)
    after_counts = Counter(row[LABEL_COLUMN] for row in final_rows)
    final_missing = missing_counts(final_rows)
    final_grouped_labels = labels_by_abstract(final_rows)
    final_duplicate_texts = len(final_rows) - len(final_grouped_labels)
    final_conflicts = sum(
        len(labels) > 1 for labels in final_grouped_labels.values()
    )

    report = [
        "MEDICAL ABSTRACT DISEASE CLASSIFICATION - CLEANING REPORT",
        "=" * 59,
        "",
        "Method",
        "------",
        "- Original train and test rows were combined; the original split was discarded.",
        "- Abstracts were grouped using exact medical_abstract text.",
        "- No linguistic text preprocessing or normalization was performed.",
        "- Every abstract group associated with multiple labels was removed entirely.",
        "- One row was retained for each unique, single-label abstract.",
        "- abstract_id is the first 16 hexadecimal characters of the exact text's SHA-256 hash.",
        "",
        "Summary",
        "-------",
        f"Original total rows: {len(combined_rows):,}",
        f"Original unique abstracts: {len(grouped_labels):,}",
        f"Ambiguous/conflicting abstract groups removed: {len(ambiguous_texts):,}",
        f"Original rows belonging to ambiguous groups: {ambiguous_original_rows:,}",
        f"Final clean dataset size: {len(final_rows):,}",
        "",
        "CLASS DISTRIBUTIONS",
        "===================",
        "",
    ]
    add_distribution(
        report,
        "Before cleaning (original combined rows)",
        before_counts,
        len(combined_rows),
        label_names,
    )
    add_distribution(
        report,
        "After cleaning (unique, unambiguous abstracts)",
        after_counts,
        len(final_rows),
        label_names,
    )
    report.extend(
        [
            "FINAL DATA QUALITY CHECKS",
            "=========================",
            "",
            f"Missing values in final data (total): {sum(final_missing.values()):,}",
            *(f"  {column}: {count:,}" for column, count in final_missing.items()),
            f"Duplicate abstract texts in final data: {final_duplicate_texts:,}",
            f"Conflicting-label abstracts remaining in final data: {final_conflicts:,}",
            f"Labels present: {', '.join(sorted(after_counts, key=label_sort_key))}",
            "All required assertions passed: yes",
            "",
        ]
    )

    write_clean_dataset(final_rows)
    write_removed_ambiguous(
        ambiguous_abstracts,
        label_names,
        combined_rows,
        train_text_counts,
        test_text_counts,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CLEANING_REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print(f"Clean dataset written: {CLEAN_DATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Final clean dataset size: {len(final_rows):,}")
    print(f"Ambiguous groups removed: {len(ambiguous_texts):,}")
    print("All required assertions passed.")


if __name__ == "__main__":
    main()
