from pathlib import Path
import pandas as pd


# Dataset paths
DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "train.csv"
)

LATER_PATH = DATA_PATH.parent / "later.csv"
HOLDOUT_PATH = DATA_PATH.parent / "holdout_features.csv"


def explore_dataset():

    # Load datasets
    df = pd.read_csv(DATA_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)
    later = pd.read_csv(LATER_PATH)

    # Basic dataset information
    print("\n========== DATASET OVERVIEW ==========")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nNumerical summary:")
    print(df.describe().T)

    # Problem 1: Data leakage
    print("\n========== DATA LEAKAGE INVESTIGATION ==========")

    print("\nTraining columns:")
    print(df.columns.tolist())

    print("\nHoldout columns:")
    print(holdout.columns.tolist())

    print("\nColumns missing from holdout:")
    print(sorted(set(df.columns) - set(holdout.columns)))

    print("\nCorrelation with energy_kwh:")
    print(
        df.corr(numeric_only=True)["energy_kwh"]
        .sort_values(ascending=False)
    )

    difference = (
        df["kwh_billed_in_app"] - df["energy_kwh"]
    ).abs()

    print("\nBilling and actual energy comparison:")
    print(
        "Correlation:",
        df["kwh_billed_in_app"].corr(df["energy_kwh"])
    )
    print(
        "Exactly equal records:",
        (df["kwh_billed_in_app"] == df["energy_kwh"]).sum()
    )
    print("Mean absolute difference:", difference.mean())
    print("Maximum absolute difference:", difference.max())

    # Problem 2: Unit mix-up
    print("\n========== UNIT MIX-UP INVESTIGATION ==========")

    print("\nSession duration by station type:")

    grouped = df.groupby("station_type")["session_minutes"].agg(
        ["count", "min", "mean", "median", "max"]
    )

    print(grouped)

    highway = df.loc[
        df["station_type"] == "Highway",
        "session_minutes"
    ]

    print("\nChecking Highway session values:")
    print("All values divisible by 60:", (highway % 60 == 0).all())

    print("\nHighway values after dividing by 60:")
    print((highway / 60).describe())

    # Problem 3: Sentinel value investigation
    print("\n========== SENTINEL VALUE INVESTIGATION ==========")

    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:
        print(f"\nColumn: {column}")

        counts = df[column].value_counts(dropna=False)

        print("Most frequent values:")
        print(counts.head(10))

        print("\nSmallest values:")
        print(counts.sort_index().head(5))

        print("\nLargest values:")
        print(counts.sort_index().tail(5))

    # Problem 4: Duplicate records
    print("\n========== DUPLICATE ROW INVESTIGATION ==========")

    print("Exact duplicate rows:", df.duplicated().sum())
    print("Duplicate record IDs:", df["record_id"].duplicated().sum())

    # Problem 5: Stuck sensor investigation
    print("\n========== STUCK SENSOR INVESTIGATION ==========")

    sorted_df = df.copy()
    sorted_df["date"] = pd.to_datetime(sorted_df["date"])
    sorted_df = sorted_df.sort_values("date").reset_index(drop=True)

    numeric_columns = sorted_df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        values = sorted_df[column]
        dates = sorted_df["date"]

        run_start = 0
        runs = []

        for i in range(1, len(sorted_df) + 1):

            if (
                i == len(sorted_df)
                or values.iloc[i] != values.iloc[run_start]
            ):

                run_length = i - run_start

                if run_length >= 5:
                    runs.append({
                        "value": values.iloc[run_start],
                        "count": run_length,
                        "start_date": dates.iloc[run_start],
                        "end_date": dates.iloc[i - 1]
                    })

                run_start = i

        if runs:
            print(f"\nColumn: {column}")

            for run in sorted(
                runs,
                key=lambda x: x["count"],
                reverse=True
            )[:5]:
                print(run)

    # Problem 6: Target typo investigation
    print("\n========== TARGET TYPO INVESTIGATION ==========")

    target = df["energy_kwh"].sort_values().reset_index(drop=True)

    print("\nTarget summary:")
    print(target.describe())

    print("\nLargest target values:")
    print(
        target.tail(20)
        .sort_values(ascending=False)
        .to_string(index=False)
    )

    # Find large gaps between consecutive sorted target values
    gaps = target.diff().dropna().sort_values(ascending=False)

    print("\nLargest gaps in sorted target values:")

    for index in gaps.head(10).index:
        print({
            "lower_value": target.iloc[index - 1],
            "upper_value": target.iloc[index],
            "gap": gaps.loc[index]
        })

    print("\nTarget values above 100:", (target > 100).sum())

    # Problem 7: Spelling variants investigation
    print("\n========== SPELLING VARIANTS INVESTIGATION ==========")

    for column in ["vehicle", "station_type"]:

        print(f"\n{column} - TRAIN values:")
        print(df[column].value_counts(dropna=False))

        print(f"\n{column} - LATER values:")
        print(later[column].value_counts(dropna=False))

        # Normalize case and remove extra spaces
        cleaned_values = (
            df[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        print(f"\n{column} - Cleaned TRAIN values:")
        print(cleaned_values.value_counts(dropna=False))

        print(f"\nNumber of distinct cleaned {column} levels:")
        print(cleaned_values.nunique())

        print(f"\nUnique values in LATER {column}:")
        print(
            later[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .unique()
        )


if __name__ == "__main__":
    explore_dataset()