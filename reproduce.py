"""Recompute the archived analysis without changing the reference files."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import runpy
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent


def run_script(path):
    previous = Path.cwd()
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        os.chdir(previous)


def compare(actual, expected, path="root"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), f"Different keys: {path}"
        for key in expected:
            if key != "environment":
                compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), f"Different length: {path}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, f"{path}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9), f"Numerical mismatch: {path}: {actual} != {expected}"
    else:
        assert actual == expected, f"Different value: {path}"


def check_csv_exports():
    inputs = json.loads((ROOT / "analysis/expert_inputs.json").read_text())
    with (ROOT / "data/ahp_comparisons.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 777
    for row in rows:
        e = int(row["expert_id"].split("_")[-1]) - 1
        j, k = int(row["criterion_left"][1:]) - 1, int(row["criterion_right"][1:]) - 1
        assert float(row["comparison_ratio"]) == inputs["AHP"][e][j][k]
    with (ROOT / "data/r1_scores.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1036
    for row in rows:
        e = int(row["expert_id"].split("_")[-1]) - 1
        a = ["Client", "Contractor", "Consultant", "Shared"].index(row["alternative"])
        c = int(row["criterion"][1:]) - 1
        assert float(row["score"]) == inputs["R1_scores"][e][a][c]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    manifest = ROOT / "SHA256SUMS.txt"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, f"Changed archived file: {name}"
    check_csv_exports()
    generated = ["audit_results.json", "part1_results.json", "part2_results.json", "aggregate_scores.csv", "all_predictions.csv"]
    previous = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="ahp-topsis-") as directory:
            work = Path(directory) / "analysis"
            shutil.copytree(ROOT / "analysis", work)
            run_script(work / "verify_workbook.py")
            run_script(work / "audit_analysis.py")
            for name in generated[:3]:
                compare(json.loads((work / name).read_text()), json.loads((ROOT / "analysis" / name).read_text()), name)
            for name in generated[3:]:
                with (work / name).open(newline="") as f:
                    actual = list(csv.reader(f))
                with (ROOT / "analysis" / name).open(newline="") as f:
                    expected = list(csv.reader(f))
                assert len(actual) == len(expected)
                for ar, er in zip(actual, expected):
                    assert len(ar) == len(er)
                    for a, e in zip(ar, er):
                        try:
                            numeric = float(e)
                        except ValueError:
                            assert a == e
                        else:
                            compare(float(a), numeric, name)
            if not args.no_figures:
                os.environ.setdefault("MPLCONFIGDIR", str(Path(directory) / "matplotlib"))
                run_script(work / "make_figures.py")
            destination = ROOT / "reproduced"
            destination.mkdir(exist_ok=True)
            for name in generated:
                shutil.copy2(work / name, destination / name)
            if not args.no_figures:
                shutil.copytree(work / "figures", destination / "figures", dirs_exist_ok=True)
            report = {"status": "PASS", "reference_comparison": {"rtol": 1e-7, "atol": 1e-9}, "workbook_comparisons": 777, "workbook_R1_scores": 1036, "figures_regenerated": not args.no_figures}
            (destination / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        os.chdir(previous)
    print("PASS: workbook, CSV exports and all numerical reference outputs verified.")
    print(f"Fresh results: {ROOT / 'reproduced'}")


if __name__ == "__main__":
    main()
