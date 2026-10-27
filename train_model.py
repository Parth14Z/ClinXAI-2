"""Train the ClinXAI XGBoost model and the logistic-regression baseline.

Steps
  1. Obtain the authentic Pima dataset (OpenML #37) and verify its MD5.
  2. Validate and summarise it; write provenance.
  3. Stratified 80/20 split (seed 42); save both splits with patient ids.
  4. Tune XGBoost with 5-fold stratified CV on the training split only.
  5. Fit the baseline; record training-split CV scores for both models.
  6. Save pipelines + metadata.  The test split is NOT used here.

Usage:  python train_model.py
"""
from __future__ import annotations

import json
import sys

from src import config
from src.data import (DatasetError, dataset_summary, ensure_dataset, file_md5, load_dataset,
                      save_split, split_dataset, write_provenance)
from src.model import (Timer, build_baseline_pipeline, build_xgb_pipeline, cross_validate_pipeline,
                       save_pipeline, tune_xgb, write_metadata)
from src.preprocessing import preprocessing_summary


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    config.ensure_directories()
    try:
        path = ensure_dataset()
    except DatasetError as exc:
        print(f"[stop] {exc}", file=sys.stderr)
        return 1
    provenance = write_provenance(path)
    print(f"[data] {provenance['source']}  md5={provenance['md5_actual']}")

    df = load_dataset(path)
    df.to_csv(config.FULL_CSV, index=False)
    summary = dataset_summary(df)
    (config.METRICS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), "utf-8")
    print(f"[data] {summary['n_rows']} rows, classes {summary['class_counts']}, "
          f"zero placeholders {summary['zero_placeholders']}")

    split = split_dataset(df)
    save_split(split)
    print(f"[split] train={len(split.X_train)} (pos rate {split.y_train.mean():.3f})  "
          f"test={len(split.X_test)} (pos rate {split.y_test.mean():.3f})")

    with Timer() as t:
        search = tune_xgb(split.X_train, split.y_train)
    best = search.best_estimator_
    print(f"[xgb] best CV ROC-AUC {search.best_score_:.4f} with {search.best_params_} ({t.seconds}s)")

    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    cv_xgb = cross_validate_pipeline(build_xgb_pipeline(**best_params), split.X_train, split.y_train)
    baseline = build_baseline_pipeline()
    cv_base = cross_validate_pipeline(baseline, split.X_train, split.y_train)
    baseline.fit(split.X_train, split.y_train)
    for name, cv in (("xgb", cv_xgb), ("logreg", cv_base)):
        print(f"[cv:{name}] " + "  ".join(f"{m}={v['mean']:.3f}±{v['std']:.3f}" for m, v in cv.items()))

    cv_table = search.cv_results_
    grid_rows = [
        {**{k.replace("model__", ""): v for k, v in p.items()},
         "mean_cv_roc_auc": float(m), "std_cv_roc_auc": float(s), "rank": int(r)}
        for p, m, s, r in zip(cv_table["params"], cv_table["mean_test_score"],
                              cv_table["std_test_score"], cv_table["rank_test_score"])
    ]
    (config.METRICS_DIR / "xgb_grid_search_cv.json").write_text(
        json.dumps(sorted(grid_rows, key=lambda r: r["rank"]), indent=2), "utf-8")

    pre = best.named_steps["preprocess"]
    (config.METRICS_DIR / "preprocessing_summary.json").write_text(
        json.dumps(preprocessing_summary(pre, split.X_train, split.X_test), indent=2), "utf-8")

    save_pipeline(best, config.XGB_MODEL_PATH)
    save_pipeline(baseline, config.BASELINE_MODEL_PATH)
    write_metadata(search, cv_xgb, cv_base, len(split.X_train), len(split.X_test),
                   file_md5(path), train_seconds=t.seconds)
    print(f"[save] {config.XGB_MODEL_PATH.name}, {config.BASELINE_MODEL_PATH.name}, "
          f"{config.MODEL_METADATA_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
