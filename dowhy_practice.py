#%%
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dowhy import CausalModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


RANDOM_STATE = 1
SAMPLE_SIZE = 20000
COMMON_CAUSES = [
    "distance_to_center",
    "avg_rating",
    "stars",
    "n_reviews",
    "avg_price",
    "city_id",
]


class RF1D(RandomForestRegressor):
    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y).ravel()
        return super().fit(X, y, sample_weight=sample_weight)

#%%
# 1) Load and prepare data
start_time = datetime.now()
df = pd.read_csv("./data/train_set.csv").dropna().reset_index(drop=True)
df_ci = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)

print(df_ci.shape)
print(df_ci[COMMON_CAUSES + ["content_score", "n_clicks"]].head())

#%%
# 2) Build causal graph and identify estimand
model = CausalModel(
    data=df_ci,
    treatment="content_score",
    outcome="n_clicks",
    common_causes=COMMON_CAUSES,
)

identified = model.identify_effect()
print(identified)

#%%
# 3) Estimate ATE with DML + linear final model
estimate_ate = model.estimate_effect(
    identified,
    method_name="backdoor.econml.dml.DML",
    method_params={
        "init_params": {
            "model_y": RF1D(
                n_estimators=20, random_state=RANDOM_STATE, min_samples_leaf=20
            ),
            "model_t": RF1D(
                n_estimators=20, random_state=RANDOM_STATE, min_samples_leaf=20
            ),
            "model_final": LinearRegression(),
            "discrete_treatment": False,
            "cv": 3,
            "random_state": RANDOM_STATE,
        },
        "fit_params": {"inference": "bootstrap"},
    },
)

print("ATE:", estimate_ate.value)
print("Time taken:", datetime.now() - start_time)
print(estimate_ate.estimator.estimator.summary())

#%%
# 4) Refutation checks
refute_placebo = model.refute_estimate(
    identified,
    estimate_ate,
    method_name="placebo_treatment_refuter",
    show_progress_bar=True,
    placebo_type="permute",
)
print(refute_placebo)

refute_random_common_cause = model.refute_estimate(
    identified,
    estimate_ate,
    method_name="random_common_cause",
    show_progress_bar=True,
)
print(refute_random_common_cause)

#%%
# 5) Estimate CATE with XGBoost as final model
estimate_xgb = model.estimate_effect(
    identified,
    method_name="backdoor.econml.dml.DML",
    method_params={
        "init_params": {
            "model_y": RF1D(
                n_estimators=20, random_state=RANDOM_STATE, min_samples_leaf=20
            ),
            "model_t": RF1D(
                n_estimators=20, random_state=RANDOM_STATE, min_samples_leaf=20
            ),
            "model_final": XGBRegressor(
                n_estimators=50,
                max_depth=2,
                min_child_weight=10,
                learning_rate=0.05,
                reg_lambda=10,
                random_state=RANDOM_STATE,
            ),
            "discrete_treatment": False,
            "cv": 3,
            "random_state": RANDOM_STATE,
        },
        "fit_params": {"inference": "bootstrap"},
    },
    effect_modifiers=COMMON_CAUSES,
)

print(estimate_xgb)
refute_xgb = model.refute_estimate(identified, estimate_xgb, method_name="random_common_cause")
print(refute_xgb)

#%%
# 6) Analyze high-effect segment from XGB CATE
df_ci["cate_effect_xgb"] = estimate_xgb.estimator.effect(df_ci[COMMON_CAUSES])
print(df_ci["cate_effect_xgb"].describe())

high_effect_xgb = df_ci[(df_ci["cate_effect_xgb"] > 0.01) & (df_ci["cate_effect_xgb"] < 50.0)]
comparison_xgb = pd.concat(
    [
        df_ci[COMMON_CAUSES].mean().rename("overall_mean"),
        high_effect_xgb[COMMON_CAUSES].mean().rename("high_effect_group_mean"),
    ],
    axis=1,
)
print("--- High-effect group vs overall mean (XGB CATE) ---")
print(comparison_xgb)
print(f"Selected hotels: {len(high_effect_xgb)}")

#%%
# 7) Estimate CATE with linear final model
estimate_linear = model.estimate_effect(
    identified_estimand=identified,
    method_name="backdoor.econml.dml.DML",
    target_units="ate",
    effect_modifiers=COMMON_CAUSES,
    method_params={
        "init_params": {
            "model_y": RF1D(
                n_estimators=20, random_state=RANDOM_STATE, min_samples_leaf=20
            ),
            "model_t": RF1D(
                n_estimators=20, random_state=RANDOM_STATE, min_samples_leaf=20
            ),
            "model_final": LinearRegression(),
            "discrete_treatment": False,
            "cv": 3,
        },
        "fit_params": {"inference": "auto"},
    },
)

print(estimate_linear.estimator.estimator.summary())

#%%
# 8) Analyze high-effect segment from linear CATE
df_ci["cate_effect_linear"] = estimate_linear.estimator.effect(df_ci[COMMON_CAUSES])
print(df_ci["cate_effect_linear"].describe())

high_effect_linear = df_ci[
    (df_ci["cate_effect_linear"] > 1.0) & (df_ci["cate_effect_linear"] < 50.0)
]

comparison_linear = pd.concat(
    [
        df_ci[COMMON_CAUSES].mean().rename("overall_mean"),
        high_effect_linear[COMMON_CAUSES].mean().rename("high_effect_group_mean"),
    ],
    axis=1,
)
print("--- High-effect group vs overall mean (Linear CATE) ---")
print(comparison_linear)
print(f"Selected hotels: {len(high_effect_linear)}")

total_lift = df_ci["cate_effect_linear"].sum()
target_lift = high_effect_linear["cate_effect_linear"].sum()
share_hotels = len(high_effect_linear) / len(df_ci) * 100 if len(df_ci) > 0 else 0
share_lift = target_lift / total_lift * 100 if total_lift != 0 else 0
print(f"Total expected click lift (all hotels): {total_lift:.0f}")
print(f"Total expected click lift (targeted group): {target_lift:.0f}")
print(f"Top {share_hotels:.1f}% hotels explain {share_lift:.1f}% of estimated lift")

#%%
# 9) Visualize where the effect is concentrated
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_ci[df_ci["cate_effect_linear"] > 0],
    x="n_reviews",
    y="stars",
    hue="cate_effect_linear",
    size="cate_effect_linear",
    palette="YlGnBu",
    sizes=(20, 200),
)
plt.title("Strategic Target Map: Where Content Score Matters Most")
plt.xlabel("Number of Reviews")
plt.ylabel("Hotel Stars")
plt.show()

#%%
# 10) Cumulative impact curve
df_sorted = df_ci.sort_values("cate_effect_linear", ascending=False).reset_index(drop=True)
total_effect = df_sorted["cate_effect_linear"].sum()

if total_effect != 0:
    df_sorted["cumulative_effect"] = df_sorted["cate_effect_linear"].cumsum() / total_effect
    df_sorted["percentile"] = df_sorted.index / len(df_sorted)

    plt.figure(figsize=(8, 5))
    plt.plot(df_sorted["percentile"], df_sorted["cumulative_effect"], color="darkorange", lw=3)
    plt.fill_between(
        df_sorted["percentile"],
        df_sorted["cumulative_effect"],
        color="orange",
        alpha=0.2,
    )
    plt.title("Efficiency of Targeted Content Optimization")
    plt.xlabel("Percentage of Hotels (sorted by estimated effect)")
    plt.ylabel("Cumulative proportion of total click lift")
    plt.grid(alpha=0.3)
    plt.show()
else:
    print("Cumulative impact plot skipped because total effect is zero.")

#%%
# 11) Effect trend by hotel stars
plt.figure(figsize=(8, 5))
sns.lineplot(data=df_ci, x="stars", y="cate_effect_linear", marker="o", color="teal")
plt.title("How Hotel Stars Relate to Content Score Effect")
plt.ylabel("Estimated Causal Effect (Clicks)")
plt.xlabel("Hotel Stars")
plt.show()
