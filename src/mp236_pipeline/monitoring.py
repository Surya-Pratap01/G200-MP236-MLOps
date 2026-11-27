from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error

from mp236_pipeline.preprocessing import clean_dataframe

BASE_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = BASE_DIR / "params.yaml"
RAW_TRAIN_PATH = BASE_DIR / "data" / "raw" / "train.csv"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
REPORT_PATH = BASE_DIR / "reports" / "stage4_monitoring_results.csv"

REQUIRED_PSI_FEATURES = [
    "battery_start_pct",
    "session_minutes",
    "charger_kw",
    "temp_c",
    "vehicle_age_yr",
    "queue_before",
]

MODEL_INPUT_FEATURES = [
    "battery_start_pct",
    "session_minutes",
    "charger_kw",
    "temp_c",
    "vehicle_age_yr",
    "queue_before",
    "vehicle",
    "station_type",
]

STATION_GROUPS = ["Mall", "Highway", "Residential"]


def load_params(params_path: Path = PARAMS_PATH) -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
    eps: float = 1e-6,
) -> dict:
    """Compute 10-bin PSI using train-derived quantile bin edges.

    Missing values are explicitly counted and excluded from numeric binning
    (documented in the returned metadata rather than silently dropped).
    """
    exp_missing = int(expected.isna().sum())
    act_missing = int(actual.isna().sum())

    exp_valid = pd.to_numeric(expected.dropna(), errors="coerce").dropna().to_numpy()
    act_valid = pd.to_numeric(actual.dropna(), errors="coerce").dropna().to_numpy()

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(exp_valid, quantiles))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    exp_counts, _ = np.histogram(exp_valid, bins=edges)
    act_counts, _ = np.histogram(act_valid, bins=edges)

    exp_share = np.clip(exp_counts / len(exp_valid), eps, None)
    act_share = np.clip(act_counts / len(act_valid), eps, None)

    psi_value = float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))

    return {
        "psi": round(psi_value, 6),
        "bins_used": int(len(edges) - 1),
        "train_non_null": int(len(exp_valid)),
        "train_missing": exp_missing,
        "later_non_null": int(len(act_valid)),
        "later_missing": act_missing,
        "missing_handling": "excluded_after_explicit_count",
    }


def run_monitoring() -> pd.DataFrame:
    params = load_params()
    target_col = params["target_column"]
    later_path = BASE_DIR / params["data"]["later_path"]

    # 1. Load & clean training reference + load later.csv
    raw_train_df = pd.read_csv(RAW_TRAIN_PATH)
    train_df = clean_dataframe(raw_train_df)
    later_df = pd.read_csv(later_path)

    # 2. Full numeric-column PSI audit (train-cleaned vs later)
    numeric_cols = [
        col
        for col in train_df.select_dtypes(include=[np.number]).columns
        if col in later_df.columns
    ]
    # Ensure all 6 required PSI features are included even if nullable dtype
    for col in REQUIRED_PSI_FEATURES:
        if col not in numeric_cols and col in train_df.columns and col in later_df.columns:
            numeric_cols.append(col)

    rows = []
    feature_psi_map = {}

    for col in numeric_cols:
        stats = compute_psi(train_df[col], later_df[col], bins=10, eps=1e-6)
        role = "prediction_feature" if col in REQUIRED_PSI_FEATURES else "target_audit"
        if col in REQUIRED_PSI_FEATURES:
            feature_psi_map[col] = stats["psi"]
        rows.append(
            {
                "section": "psi_audit",
                "metric": col,
                "value": stats["psi"],
                "role": role,
                "row_count": stats["later_non_null"],
                "train_missing": stats["train_missing"],
                "later_missing": stats["later_missing"],
                "notes": f"bins={stats['bins_used']}; missing={stats['missing_handling']}",
            }
        )

    most_drifted_feature = max(feature_psi_map, key=feature_psi_map.get)

    # 3. Evaluate committed V1 model on all 300 later.csv rows by station_type
    v1_model = joblib.load(MODEL_PATH)
    X_later = later_df[MODEL_INPUT_FEATURES]
    y_later = later_df[target_col]
    y_pred = v1_model.predict(X_later)

    eval_df = later_df[["station_type", target_col]].copy()
    eval_df["prediction"] = y_pred

    overall_mae = round(float(mean_absolute_error(y_later, y_pred)), 6)
    group_maes = {}
    group_counts = {}

    for group in STATION_GROUPS:
        subset = eval_df[eval_df["station_type"] == group]
        group_counts[group] = int(len(subset))
        group_maes[group] = round(
            float(mean_absolute_error(subset[target_col], subset["prediction"])), 6
        )
        rows.append(
            {
                "section": "v1_group_mae",
                "metric": group,
                "value": group_maes[group],
                "role": "station_type",
                "row_count": group_counts[group],
                "train_missing": 0,
                "later_missing": 0,
                "notes": "V1 model.joblib on later.csv",
            }
        )

    worst_served_group = max(group_maes, key=group_maes.get)

    # 4. Summary rows for easy downstream consumption by findings.yaml
    rows.extend(
        [
            {
                "section": "summary",
                "metric": "most_drifted_feature",
                "value": most_drifted_feature,
                "role": "prediction_feature",
                "row_count": len(later_df),
                "train_missing": 0,
                "later_missing": 0,
                "notes": f"max_psi={feature_psi_map[most_drifted_feature]}",
            },
            {
                "section": "summary",
                "metric": "worst_served_group",
                "value": worst_served_group,
                "role": "station_type",
                "row_count": group_counts[worst_served_group],
                "train_missing": 0,
                "later_missing": 0,
                "notes": f"max_mae={group_maes[worst_served_group]}",
            },
            {
                "section": "summary",
                "metric": "v1_overall_mae_later",
                "value": overall_mae,
                "role": "all_later_rows",
                "row_count": len(later_df),
                "train_missing": 0,
                "later_missing": 0,
                "notes": "All 300 rows of later.csv",
            },
        ]
    )

    report_df = pd.DataFrame(rows)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_PATH, index=False)

    # 5. Console report
    print("\n========== STAGE 4 MONITORING REPORT (KUNAL) ==========")
    print(f"Train rows (cleaned): {len(train_df)}")
    print(f"Later rows:           {len(later_df)}")
    print("\n--- PSI Audit (10-bin, train-derived edges, eps=1e-6) ---")
    psi_df = report_df[report_df["section"] == "psi_audit"][
        ["metric", "role", "value", "train_missing", "later_missing"]
    ]
    print(psi_df.to_string(index=False))
    print(f"\nMost drifted prediction feature: {most_drifted_feature} (PSI = {feature_psi_map[most_drifted_feature]})")

    print("\n--- V1 Station-wise MAE on later.csv ---")
    for group in STATION_GROUPS:
        print(f"  {group:12s} | rows={group_counts[group]:3d} | MAE={group_maes[group]:.6f}")
    print(f"  {'Overall':12s} | rows={len(later_df):3d} | MAE={overall_mae:.6f}")
    print(f"\nWorst served group: {worst_served_group} (MAE = {group_maes[worst_served_group]:.6f})")
    print(f"\nSaved report to: {REPORT_PATH}")

    return report_df


if __name__ == "__main__":
    run_monitoring()