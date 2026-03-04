#%%
from dowhy import CausalModel
from econml.dml import DML
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
#%%
starttime = datetime.now()
df = pd.read_csv('./data/train_set.csv')
df = df.dropna().reset_index(drop=True)
X_cols = [
    'distance_to_center', 'avg_rating', 'stars', 
    'n_reviews', 'avg_price', 'city_id'
]

#%%
df_ci = df.copy().sample(n=20000, random_state=1).reset_index(drop=True)

model = CausalModel(
    data=df_ci,
    treatment='content_score',
    outcome='n_clicks',
    common_causes=[
        'distance_to_center',
        'avg_rating',
        'stars',
        'n_reviews',
        'avg_price',
        'city_id'
        ]
)

identified = model.identify_effect()

#%%
estimate = model.estimate_effect(
    identified,
    method_name="backdoor.econml.dml.DML",
    method_params={
        'init_params': {
            'model_y': RandomForestRegressor(n_estimators=20, random_state=1, min_samples_leaf=20),
            'model_t': RandomForestRegressor(n_estimators=20, random_state=1, min_samples_leaf=20),
            'model_final': LinearRegression(),
            'discrete_treatment': False,
            'cv': 3,
            'random_state': 1
        },
        'fit_params': {'inference':'bootstrap'}
        }
)

print('ATE:', estimate.value)

print('Time taken:', datetime.now() - starttime)
# %%
print(estimate.interpret())
print(estimate)
refute_placebo = model.refute_estimate(
    identified,
    estimate,
    method_name="placebo_treatment_refuter",
    show_progress_bar=True,
    placebo_type="permute"
)
print(refute_placebo)
#%%
refute_random_common_cause = model.refute_estimate(
    identified,
    estimate,
    show_progress_bar=True,
    method_name="random_common_cause"
)
print(refute_random_common_cause)
#%%
# 95% 신뢰 구간 출력
print(estimate.estimator.estimator.summary())
# %%
estimate_xgb = model.estimate_effect(
    identified,
    method_name="backdoor.econml.dml.DML",
    method_params={
        'init_params': {
            'model_y': RandomForestRegressor(n_estimators=20, random_state=1, min_samples_leaf=20),
            'model_t': RandomForestRegressor(n_estimators=20, random_state=1, min_samples_leaf=20),
            'model_final': XGBRegressor(
                n_estimators=50, 
                max_depth=2,          # 트리를 얕게 해서 과적합 방지
                min_child_weight=10,  # 최소 데이터 개수 보장
                learning_rate=0.05,
                reg_lambda=10         # 규제 강화
            ),
            'discrete_treatment': False,
            'cv': 3,
            'random_state': 1
        },
        'fit_params': {'inference':'bootstrap'}
        },
    effect_modifiers=X_cols
)
#%%
print(estimate_xgb)

refute_xgb = model.refute_estimate(identified, estimate_xgb,
    method_name="random_common_cause")
print(refute_xgb)

# %%
df_ci['cate_effect'] = estimate_xgb.estimator.effect(df_ci[X_cols])
# 1. 현실적인 고효율 그룹 추출 (효과가 1.0 이상 50.0 이하)
# 600 같은 숫자는 노이즈로 보고 제외한 뒤 분석합니다.
realistic_high_effect = df_ci[(df_ci['cate_effect'] > 0.01) & (df_ci['cate_effect'] < 50.0)]

# 2. 이들의 특징을 전체 평균과 비교
comparison = pd.concat([
    df_ci[X_cols].mean().rename('전체 평균'),
    realistic_high_effect[X_cols].mean().rename('고효율 그룹 평균')
], axis=1)

print("--- 현실적 고효율 그룹 vs 전체 평균 비교 ---")
print(comparison)
print(f"\n대상 호텔 수: {len(realistic_high_effect)}개")
# %%
print(df_ci['cate_effect_xgb'].describe())
df_ci.loc[df_ci['cate_effect_xgb'] > 0.01, 'cate_effect_xgb'].value_counts()
# %%
estimate_linear_cate = model.estimate_effect(
    identified_estimand=identified,
    method_name="backdoor.econml.dml.DML",
    target_units="ate",
    effect_modifiers=X_cols, # X 변수 지정 필수
    method_params={
        "init_params": {
            'model_y': RandomForestRegressor(n_estimators=20, random_state=1, min_samples_leaf=20),
            'model_t': RandomForestRegressor(n_estimators=20, random_state=1, min_samples_leaf=20),
            'model_final': LinearRegression(), # 다시 선형으로!
            'discrete_treatment': False,
            'cv': 3
        },
        "fit_params": {
            'inference': 'auto'
        }
    }
)

# 결과 확인
# %%
# print(estimate_linear_cate)
print(estimate_linear_cate.estimator.estimator.summary())
#%%
df_ci['cate_effect'] = estimate_linear_cate.estimator.effect(df_ci[X_cols])
# 1. 현실적인 고효율 그룹 추출 (효과가 1.0 이상 50.0 이하)
# 600 같은 숫자는 노이즈로 보고 제외한 뒤 분석합니다.
realistic_high_effect = df_ci[(df_ci['cate_effect'] > 1) & (df_ci['cate_effect'] < 50.0)]

# 2. 이들의 특징을 전체 평균과 비교
comparison = pd.concat([
    df_ci[X_cols].mean().rename('전체 평균'),
    realistic_high_effect[X_cols].mean().rename('고효율 그룹 평균')
], axis=1)

print("--- 현실적 고효율 그룹 vs 전체 평균 비교 ---")
print(comparison)
print(f"\n대상 호텔 수: {len(realistic_high_effect)}개")
# %%
print(df_ci['cate_effect'].describe())
df_ci.loc[df_ci['cate_effect'] > 0.01, 'cate_effect'].value_counts()
# %%
# 모든 호텔에 대해 예측된 효과를 합산 (컨텐츠 점수를 1점씩 올린다고 가정할 때)
total_lift = df_ci['cate_effect'].sum()
print(f"전체 호텔 컨텐츠 점수 1점 개선 시 총 클릭 상승 기대치: {total_lift:.0f}회")

# 상위 4382개 호텔만 집중 관리했을 때의 상승분
target_lift = realistic_high_effect['cate_effect'].sum()
print(f"타겟 그룹(4382개) 집중 관리 시 총 클릭 상승 기대치: {target_lift:.0f}회")
print(f"효율성: 상위 {len(realistic_high_effect)/len(df_ci)*100:.1f}% 호텔이 전체 효과의 {target_lift/total_lift*100:.1f}%를 점유함")# %%

# %%
plt.figure(figsize=(10, 6))
# 효과가 0보다 큰 데이터만 시각화하면 더 명확합니다.
sns.scatterplot(data=df_ci[df_ci['cate_effect'] > 0], 
                x='n_reviews', y='stars', 
                hue='cate_effect', size='cate_effect',
                palette='YlGnBu', sizes=(20, 200))
plt.title("Strategic Target Map: Where Content Score Matters Most")
plt.xlabel("Number of Reviews")
plt.ylabel("Hotel Stars")
plt.show()
# %%
# 효과 순으로 정렬하여 누적 합계 계산
df_sorted = df_ci.sort_values('cate_effect', ascending=False).reset_index()
df_sorted['cumulative_effect'] = df_sorted['cate_effect'].cumsum() / df_sorted['cate_effect'].sum()
df_sorted['percentile'] = df_sorted.index / len(df_sorted)

plt.figure(figsize=(8, 5))
plt.plot(df_sorted['percentile'], df_sorted['cumulative_effect'], color='darkorange', lw=3)
plt.fill_between(df_sorted['percentile'], df_sorted['cumulative_effect'], color='orange', alpha=0.2)
plt.axvline(x=0.219, color='red', linestyle='--', label='Top 21.9% Targets')
plt.axhline(y=0.634, color='blue', linestyle='--', label='63.4% Total Impact')
plt.title("Efficiency of Targeted Content Optimization")
plt.xlabel("Percentage of Hotels (Sorted by Effect)")
plt.ylabel("Cumulative Proportion of Total Click Lift")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# %%
# 성급(stars)에 따른 평균 효과 시각화
plt.figure(figsize=(8, 5))
sns.lineplot(data=df_ci, x='stars', y='cate_effect', marker='o', color='teal')
plt.title("How Hotel Stars Amplify Content Score Effect")
plt.ylabel("Estimated Causal Effect (Clicks)")
plt.xlabel("Hotel Stars")
plt.show()
# %%
