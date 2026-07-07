"""
DROP-IN REPLACEMENT for the _evaluate_uae and related methods in model_training.py.

Instructions:
  1. In model_training.py, ADD at the top of imports:
       from uae_validation import run_uae_external_validation, UAEValidationReport

  2. REPLACE the entire _evaluate_uae method with the one below.

  3. REPLACE the _save_best_model_summary method with the one below (minor update
     to handle UAEValidationReport instead of raw dict).

  4. In _load_task_data, update the "uci" task config to also load and return
     cv_fold_indices — already done in the existing code, no change needed.

  5. In train_all(), update the call to _evaluate_uae to pass additional args:
       see updated train_all() snippet below.

  6. In TaskResult dataclass, change uae_metrics type annotation:
       from:  uae_metrics: Optional[Dict[str, Any]] = None
       to:    uae_metrics: Optional["UAEValidationReport"] = None
"""

# ─── Add this import to the top of model_training.py ───────────────────────
# from uae_validation import run_uae_external_validation, UAEValidationReport

# ─── Replace _evaluate_uae in model_training.py ─────────────────────────────

from pathlib import Path

from uae_validation import UAEValidationReport


def _evaluate_uae(self, uci_result) -> "UAEValidationReport":
    """
    Rigorous UAE external validation using feature alignment rather than
    zero-fill imputation.

    Runs two complementary validation tracks:
      Track A (PRIMARY): Reduced-feature UCI model using only UCI-UAE
                         aligned features. Report this in the paper.
      Track B (SUPPLEMENTARY): Full model with training-median imputation
                                for missing features. Caveated.

    See uae_validation.py for full documentation.
    """
    from uae_validation import run_uae_external_validation

    self.logger.info("[UAE] Running rigorous external validation (2-track) …")

    task_cfg_uae = self.cfg.get_task("uae")
    task_cfg_uci = self.cfg.get_task("uci")
    splits_dir = Path(task_cfg_uae.get("splits_dir", "data/splits"))
    uae_path = splits_dir / task_cfg_uae.get("full_file", "uae_full.csv")

    if not uae_path.exists():
        self.logger.warning(
            "[UAE] File not found at %s — skipping UAE validation.", uae_path
        )
        return None

    uae_df = pd.read_csv(uae_path)
    target_col = task_cfg_uae["target_col"]

    # Reload UCI train/test and CV fold indices
    train_df, test_df, cv_fold_indices = self._load_task_data("uci", task_cfg_uci)
    train_target = task_cfg_uci["target_col"]
    X_train = train_df.drop(columns=[train_target])
    y_train = train_df[train_target].values.astype(int)
    X_test = test_df.drop(columns=[train_target])
    y_test = test_df[train_target].values.astype(int)

    # Get the best UCI model and its union features
    best_name = uci_result.best_model_name
    best_result = uci_result.model_results[best_name]
    union_features = best_result.union_features
    best_params = best_result.hyperparameters

    # Load the saved final model for Track B
    uci_artifacts_dir = Path(task_cfg_uci["artifacts_dir"])
    model_path = uci_artifacts_dir / best_name / "calibrated_model.joblib"
    if not model_path.exists():
        model_path = uci_artifacts_dir / best_name / "final_model.joblib"
    if not model_path.exists():
        self.logger.warning(
            "[UAE] Model artifact not found at %s — Track B unavailable.", model_path
        )
        full_model = None
    else:
        full_model = joblib.load(model_path)

    uae_artifacts_dir = Path(task_cfg_uci["artifacts_dir"]) / "uae_validation"

    from uae_validation import run_uae_external_validation
    report = run_uae_external_validation(
        best_model_name=best_name,
        best_model_params=best_params,
        full_trained_model=full_model,
        union_features=union_features,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cv_fold_indices=cv_fold_indices,
        uae_df=uae_df,
        target_col=target_col,
        random_seed=self.cfg.random_seed,
        artifacts_dir=uae_artifacts_dir,
    )
    return report


# ─── Replace _save_best_model_summary in model_training.py ──────────────────

def _save_best_model_summary(self, results):
    """Updated to handle UAEValidationReport in uae_metrics."""
    from model_utils import save_json_artifact
    from uae_validation import UAEValidationReport

    summary = {
        "pipeline_stage": "model_training",
        "random_seed": self.cfg.random_seed,
        "tasks": {},
    }

    for task_key, task_result in results.items():
        task_summary = {
            "best_model": task_result.best_model_name,
            "primary_metric": task_result.primary_metric,
            "best_cv_score": round(task_result.best_cv_score, 6),
            "test_metrics": task_result.test_metrics_best,
            "all_model_cv_scores": {
                name: round(r.primary_cv_score, 6)
                for name, r in task_result.model_results.items()
            },
        }

        if task_result.uae_metrics is not None:
            uae = task_result.uae_metrics
            if isinstance(uae, UAEValidationReport):
                task_summary["uae_external_validation"] = uae.as_dict()
                # Surface the primary result (Track A) prominently
                if uae.track_a_valid:
                    task_summary["uae_primary_result"] = {
                        "note": (
                            "Track A: reduced-feature UCI model (%d features). "
                            "This is the valid, reportable external validation result."
                            % uae.track_a_n_features
                        ),
                        "metrics": uae.track_a_uae_metrics,
                    }
            else:
                task_summary["uae_external_validation"] = uae

        summary["tasks"][task_key] = task_summary

        task_artifacts_dir = Path(self.cfg.get_task(task_key)["artifacts_dir"])
        save_json_artifact(task_summary, task_artifacts_dir / "best_model.json")
        self.logger.info(
            "[%s] best_model.json saved: best=%s, cv_%s=%.4f",
            task_key.upper(), task_result.best_model_name,
            task_result.primary_metric, task_result.best_cv_score,
        )

    save_json_artifact(summary, Path("artifacts/models/best_model_summary.json"))
    self.logger.info("Global best_model_summary.json saved.")