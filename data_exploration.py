import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


RANDOM_STATE = 1
COMMISSION_RATE = 0.18


def load_train_data(path: str = "./data/train_set.csv") -> pd.DataFrame:
    """Load training data from disk."""
    return pd.read_csv(path)


def split_with_rank(df: pd.DataFrame):
    """Prepare train/valid splits using all features except ID and target."""
    y = np.log1p(df["n_clicks"])
    x_features = [c for c in df.columns if c not in ["n_clicks", "hotel_id"]]
    x = df[x_features]

    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.25, random_state=RANDOM_STATE
    )
    return x_train, x_valid, y_train, y_valid


def train_baseline_model(x_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """Train baseline click model with all ranking features."""
    model = XGBRegressor(random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    return model


def evaluate_regression(
    model: XGBRegressor, x_valid: pd.DataFrame, y_valid: pd.Series, title: str
) -> np.ndarray:
    """Evaluate regression fit and return log-scale predictions."""
    y_pred = model.predict(x_valid)
    print(f"{title} R2 score: {r2_score(y_valid, y_pred):.4f}")
    print(f"{title} mean(pred): {np.mean(y_pred):.4f}, mean(true): {np.mean(y_valid):.4f}")

    plt.figure()
    plt.scatter(y_valid, y_pred, alpha=0.01)
    plt.plot([y_valid.min(), y_valid.max()], [y_valid.min(), y_valid.max()], color="red")
    plt.title(f"{title}: prediction vs. truth")
    plt.xlabel("True log1p(n_clicks)")
    plt.ylabel("Predicted log1p(n_clicks)")
    plt.show()

    return y_pred


def estimate_rank_effect(model: XGBRegressor, x_valid: pd.DataFrame) -> None:
    """Estimate click uplift when avg_rank improves from 10 to 5."""
    x_low_rank = x_valid.copy()
    x_high_rank = x_valid.copy()
    x_low_rank["avg_rank"] = 10
    x_high_rank["avg_rank"] = 5

    y_low = model.predict(x_low_rank)
    y_high = model.predict(x_high_rank)

    effect = np.expm1(y_high) - np.expm1(y_low)
    print(f"ATE for rank 10 -> 5: {effect.mean():.4f}")

    plt.figure()
    plt.hist(effect, bins=80)
    plt.title("Estimated click effect: avg_rank 10 -> 5")
    plt.xlabel("Click uplift")
    plt.ylabel("Count")
    plt.show()


def estimate_grouped_rank_effect(model: XGBRegressor, x_valid: pd.DataFrame) -> None:
    """Compare average treatment effect across rank intervals."""
    rank_pairs = [(20, 15), (15, 10), (10, 5), (5, 1)]
    results = []

    for r_low, r_high in rank_pairs:
        x_low_rank = x_valid.copy()
        x_high_rank = x_valid.copy()
        x_low_rank["avg_rank"] = r_low
        x_high_rank["avg_rank"] = r_high

        y_low = np.expm1(model.predict(x_low_rank))
        y_high = np.expm1(model.predict(x_high_rank))
        effect = (y_high - y_low).mean()
        results.append({"from_rank": r_low, "to_rank": r_high, "ATE": effect})

    results_df = pd.DataFrame(results)
    print(results_df)

    plt.figure()
    plt.plot(results_df["to_rank"], results_df["ATE"], marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("Better rank")
    plt.ylabel("ATE (click increase)")
    plt.title("Rank improvement effect by interval")
    plt.show()


def estimate_position_curve(model: XGBRegressor, x_valid: pd.DataFrame) -> pd.Series:
    """Estimate expected clicks for rank 1..30 under counterfactual ranking."""
    x_base = x_valid.copy()
    ranks = range(1, 31)
    position_effects = []

    for rank in ranks:
        x_tmp = x_base.copy()
        x_tmp["avg_rank"] = rank
        y_pred_log = model.predict(x_tmp)
        position_effects.append(np.expm1(y_pred_log).mean())

    plt.figure()
    plt.plot(ranks, position_effects, marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("Rank")
    plt.ylabel("Expected clicks")
    plt.title("Position effect curve")
    plt.show()

    return pd.Series(position_effects, index=ranks)


def compare_observed_vs_causal(df: pd.DataFrame, causal_curve: pd.Series) -> pd.DataFrame:
    """Compare observed click means with model-based causal position estimates."""
    observed_df = df.copy()
    observed_df["rank_int"] = observed_df["avg_rank"].round().astype(int)
    observed = observed_df.groupby("rank_int")["n_clicks"].mean().loc[1:30]

    plt.figure()
    plt.plot(observed.index, observed.values, label="Observed", marker="o")
    plt.plot(causal_curve.index, causal_curve.values, label="Causal estimate", marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("Rank")
    plt.ylabel("Expected clicks")
    plt.title("Observed vs. model-based position effect")
    plt.legend()
    plt.show()

    comparison = pd.DataFrame([observed, causal_curve]).T
    comparison.columns = ["observed", "causal"]
    print(comparison)
    return comparison


def estimate_revenue(
    model: XGBRegressor, x_valid: pd.DataFrame, y_valid: pd.Series, commission_rate: float
) -> pd.DataFrame:
    """Estimate revenue using de-logged click predictions."""
    df_valid = pd.concat([x_valid.reset_index(drop=True), y_valid.reset_index(drop=True)], axis=1)

    y_pred_log = model.predict(x_valid)
    pred_clicks = np.expm1(y_pred_log)
    pred_clicks = np.clip(pred_clicks, 0, None)

    df_valid["pred_clicks"] = pred_clicks
    df_valid["expected_revenue"] = df_valid["pred_clicks"] * df_valid["avg_price"] * commission_rate
    print(df_valid["expected_revenue"].describe())
    return df_valid.sort_values("expected_revenue", ascending=False)


def train_model_without_rank(df: pd.DataFrame):
    """Train a model without avg_rank to inspect rank dependence."""
    y = np.log1p(df["n_clicks"])
    drop_cols = ["n_clicks", "hotel_id", "avg_rank"]
    x_features = [c for c in df.columns if c not in drop_cols]
    x = df[x_features]

    x_train, _, y_train, _ = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.25, random_state=RANDOM_STATE
    )

    model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    return model, x_valid, y_valid


def inspect_content_rank_relationship(df: pd.DataFrame) -> None:
    """Check correlation between content score and ranking position."""
    print(
        "content_score stats:",
        min(df["content_score"]),
        max(df["content_score"]),
        np.mean(df["content_score"]),
        np.median(df["content_score"]),
    )

    plt.figure()
    plt.scatter(df["content_score"], df["avg_rank"], alpha=0.01)
    plt.xlabel("content_score")
    plt.ylabel("avg_rank")
    plt.title("content_score vs. avg_rank")
    plt.show()

    print(df[["content_score", "avg_rank"]].corr())


def main() -> None:
    """Run end-to-end exploration and modeling workflow."""
    df = load_train_data()
    print(df.describe())
    print(df.columns.values)

    plt.figure()
    plt.hist(np.log(df["avg_price"]), bins=100)
    plt.title("Log(avg_price) distribution")
    plt.show()

    plt.figure()
    plt.scatter(np.log1p(df["avg_rank"]), np.log1p(df["n_clicks"]), alpha=0.01)
    plt.xlabel("log1p(avg_rank)")
    plt.ylabel("log1p(n_clicks)")
    plt.title("Rank vs clicks (log scale)")
    plt.show()

    x_train, x_valid, y_train, y_valid = split_with_rank(df)
    model = train_baseline_model(x_train, y_train)
    evaluate_regression(model, x_valid, y_valid, title="Baseline model")

    estimate_rank_effect(model, x_valid)
    estimate_grouped_rank_effect(model, x_valid)
    causal_curve = estimate_position_curve(model, x_valid)
    compare_observed_vs_causal(df, causal_curve)
    estimate_revenue(model, x_valid, y_valid, commission_rate=COMMISSION_RATE)

    no_rank_model, x_valid2, y_valid2 = train_model_without_rank(df)
    evaluate_regression(no_rank_model, x_valid2, y_valid2, title="No-rank model")
    inspect_content_rank_relationship(df)


if __name__ == "__main__":
    main()
