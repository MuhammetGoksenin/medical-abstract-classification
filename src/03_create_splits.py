"""Create reproducible stratified train, validation, and test splits.

This script reads only the finalized clean dataset. It performs no linguistic
preprocessing and never writes to the clean master file or data/raw/.
"""

from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

MASTER_PATH = PROCESSED_DIR / "medical_abstracts_clean.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VALIDATION_PATH = PROCESSED_DIR / "validation.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
REPORT_PATH = REPORTS_DIR / "split_report.txt"

OUTPUT_COLUMNS = [
    "abstract_id",
    "medical_abstract",
    "condition_label",
    "condition_name",
]
EXPECTED_LABELS = {"1", "2", "3", "4", "5"}
EXPECTED_TOTAL_ROWS = 8_298
RANDOM_SEED = 42


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a file without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_master(path: Path) -> list[dict[str, str]]:
    """Load the finalized clean dataset while preserving all field values."""
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        if list(reader.fieldnames) != OUTPUT_COLUMNS:
            raise ValueError(
                f"Expected columns {OUTPUT_COLUMNS}, found {reader.fieldnames}."
            )
        return list(reader)


def label_sort_key(label: str) -> tuple[int, int | str]:
    """Sort numeric labels numerically, followed by non-numeric values."""
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


def ceil_fraction(value: int, numerator: int, denominator: int) -> int:
    """Return ceil(value * numerator / denominator) using integer arithmetic."""
    return (value * numerator + denominator - 1) // denominator


def allocate_stratified_counts(
    class_counts: Counter[str], target_size: int
) -> dict[str, int]:
    """Allocate an exact target size proportionally using largest remainders."""
    total = sum(class_counts.values())
    allocations = {
        label: count * target_size // total
        for label, count in class_counts.items()
    }
    remaining = target_size - sum(allocations.values())

    # Integer remainders avoid platform-dependent floating-point tie behavior.
    remainder_order = sorted(
        class_counts,
        key=lambda label: (
            -(class_counts[label] * target_size % total),
            label_sort_key(label),
        ),
    )
    for label in remainder_order[:remaining]:
        allocations[label] += 1

    assert sum(allocations.values()) == target_size
    assert all(0 <= allocations[label] <= class_counts[label] for label in class_counts)
    return allocations


def stratified_split(
    rows: list[dict[str, str]],
    second_numerator: int,
    second_denominator: int,
    random_seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split rows reproducibly while preserving approximate class proportions."""
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped_rows[row["condition_label"]].append(row)

    class_counts = Counter(
        {label: len(label_rows) for label, label_rows in grouped_rows.items()}
    )
    second_size = ceil_fraction(
        len(rows), second_numerator, second_denominator
    )
    second_counts = allocate_stratified_counts(class_counts, second_size)

    first_rows: list[dict[str, str]] = []
    second_rows: list[dict[str, str]] = []
    generator = random.Random(random_seed)

    for label in sorted(grouped_rows, key=label_sort_key):
        label_rows = list(grouped_rows[label])
        generator.shuffle(label_rows)
        second_count = second_counts[label]
        second_rows.extend(label_rows[:second_count])
        first_rows.extend(label_rows[second_count:])

    # Avoid grouping the saved output by class while keeping order reproducible.
    generator.shuffle(first_rows)
    generator.shuffle(second_rows)

    assert len(first_rows) + len(second_rows) == len(rows)
    assert len(second_rows) == second_size
    return first_rows, second_rows


def missing_value_count(rows: list[dict[str, str]]) -> int:
    """Count empty fields across the four required columns."""
    return sum(
        row.get(column) in (None, "")
        for row in rows
        for column in OUTPUT_COLUMNS
    )


def verify_splits(
    master_rows: list[dict[str, str]],
    splits: dict[str, list[dict[str, str]]],
) -> dict[tuple[str, str], dict[str, int]]:
    """Run all requested split integrity assertions and return overlap counts."""
    assert len(master_rows) == EXPECTED_TOTAL_ROWS, (
        f"Expected {EXPECTED_TOTAL_ROWS:,} clean rows, found {len(master_rows):,}."
    )
    assert sum(len(rows) for rows in splits.values()) == EXPECTED_TOTAL_ROWS, (
        "Split row counts do not sum to the clean dataset size."
    )

    for split_name, rows in splits.items():
        ids = [row["abstract_id"] for row in rows]
        abstracts = [row["medical_abstract"] for row in rows]
        labels = {row["condition_label"] for row in rows}

        assert missing_value_count(rows) == 0, (
            f"{split_name} contains missing values."
        )
        assert len(ids) == len(set(ids)), (
            f"{split_name} contains duplicate abstract_id values."
        )
        assert len(abstracts) == len(set(abstracts)), (
            f"{split_name} contains duplicate medical_abstract values."
        )
        assert labels == EXPECTED_LABELS, (
            f"{split_name} does not contain all five expected labels."
        )

    overlap_results: dict[tuple[str, str], dict[str, int]] = {}
    for (left_name, left_rows), (right_name, right_rows) in combinations(
        splits.items(), 2
    ):
        left_ids = {row["abstract_id"] for row in left_rows}
        right_ids = {row["abstract_id"] for row in right_rows}
        left_texts = {row["medical_abstract"] for row in left_rows}
        right_texts = {row["medical_abstract"] for row in right_rows}
        id_overlap = len(left_ids & right_ids)
        text_overlap = len(left_texts & right_texts)
        overlap_results[(left_name, right_name)] = {
            "abstract_id": id_overlap,
            "medical_abstract": text_overlap,
        }
        assert id_overlap == 0, (
            f"abstract_id overlap found between {left_name} and {right_name}."
        )
        assert text_overlap == 0, (
            f"Exact abstract overlap found between {left_name} and {right_name}."
        )

    master_ids = {row["abstract_id"] for row in master_rows}
    split_ids = {
        row["abstract_id"] for rows in splits.values() for row in rows
    }
    assert split_ids == master_ids, "Split rows do not exactly cover the clean master."
    return overlap_results


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    """Write one split with the original four-column schema."""
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def append_distribution(
    report: list[str], split_name: str, rows: list[dict[str, str]]
) -> None:
    """Append class counts and within-split percentages to the report."""
    counts = Counter(row["condition_label"] for row in rows)
    condition_names = {
        row["condition_label"]: row["condition_name"] for row in rows
    }
    report.extend(
        [
            split_name,
            "  Label | Condition | Count | Percentage",
        ]
    )
    for label in sorted(counts, key=label_sort_key):
        percentage = counts[label] / len(rows) * 100
        report.append(
            f"  {label} | {condition_names[label]} | {counts[label]:,} | "
            f"{percentage:.2f}%"
        )
    report.extend(
        [f"  Total | all conditions | {len(rows):,} | 100.00%", ""]
    )


def main() -> None:
    """Create, validate, save, and report the three dataset splits."""
    master_hash_before = sha256_file(MASTER_PATH)
    master_rows = load_master(MASTER_PATH)

    # Stage 1: 70% train and 30% temporary, stratified by condition_label.
    train_rows, temporary_rows = stratified_split(
        master_rows,
        second_numerator=3,
        second_denominator=10,
        random_seed=RANDOM_SEED,
    )

    # Stage 2: divide temporary rows equally into validation and test.
    validation_rows, test_rows = stratified_split(
        temporary_rows,
        second_numerator=1,
        second_denominator=2,
        random_seed=RANDOM_SEED,
    )

    splits = {
        "Train": train_rows,
        "Validation": validation_rows,
        "Test": test_rows,
    }
    overlaps = verify_splits(master_rows, splits)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_split(TRAIN_PATH, train_rows)
    write_split(VALIDATION_PATH, validation_rows)
    write_split(TEST_PATH, test_rows)

    master_hash_after = sha256_file(MASTER_PATH)
    assert master_hash_before == master_hash_after, (
        "The clean master dataset changed during split creation."
    )

    report = [
        "MEDICAL ABSTRACT DISEASE CLASSIFICATION - SPLIT REPORT",
        "=" * 56,
        "",
        f"Random seed: {RANDOM_SEED}",
        "Split method: stratified 70%/30%, then stratified 50%/50% of temporary data",
        f"Total clean rows: {len(master_rows):,}",
        f"Train size: {len(train_rows):,} ({len(train_rows) / len(master_rows) * 100:.2f}%)",
        f"Validation size: {len(validation_rows):,} ({len(validation_rows) / len(master_rows) * 100:.2f}%)",
        f"Test size: {len(test_rows):,} ({len(test_rows) / len(master_rows) * 100:.2f}%)",
        "",
        "CLASS DISTRIBUTIONS",
        "===================",
        "",
    ]
    for split_name, rows in splits.items():
        append_distribution(report, split_name, rows)

    report.extend(["PAIRWISE OVERLAP CHECKS", "=======================", ""])
    for (left_name, right_name), counts in overlaps.items():
        report.extend(
            [
                f"{left_name} vs {right_name}:",
                f"  abstract_id overlap: {counts['abstract_id']:,}",
                f"  exact medical_abstract overlap: {counts['medical_abstract']:,}",
            ]
        )

    report.extend(
        [
            "",
            "VERIFICATION",
            "============",
            "",
            f"Clean master SHA-256 before: {master_hash_before}",
            f"Clean master SHA-256 after:  {master_hash_after}",
            "Clean master unchanged: yes",
            "Missing values across all splits: 0",
            "Duplicate abstracts within splits: 0",
            "All five labels present in every split: yes",
            f"Rows across all splits: {sum(len(rows) for rows in splits.values()):,}",
            "All required assertions passed: yes",
            "",
        ]
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(report)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()
