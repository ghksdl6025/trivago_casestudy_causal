#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

df = pd.read_csv('./data/train_set.csv')
# print(df.describe())

dft = df.dropna().reset_index(drop=True)
print(df.describe())
#%%
# print(df.describe())
print(df.columns.values)
# print(df.head)

df['avg_price'].describe()

plt.hist(np.log(df['avg_price']), bins=100)
plt.show()
#%%
plt.scatter(np.log1p(df['avg_rank']), np.log1p(df['n_clicks']), alpha=0.01)
plt.xlabel('avg_rank')
plt.ylabel('n_click')
plt.show()

#%%
y = np.log1p(df['n_clicks'])
x_features = df.columns.values
x_features = x_features[x_features != 'n_clicks']
x_features = x_features[x_features != 'hotel_id']
x = df[x_features]
x_train,x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1)
x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train, test_size=0.25, random_state=1)

#%%
# Build n_clicks prediction baseline model
# The model is trained with all features

model = XGBRegressor()
model.fit(x_train, y_train)

# %%
y_pred = model.predict(x_valid)
print(len(y_valid))
print(len(y_pred))
plt.scatter(y_valid, y_pred, alpha=0.01)
plt.plot([y_valid.min(), y_valid.max()],
         [y_valid.min(), y_valid.max()],
         color='red')

plt.show()
print(np.mean(y_pred), np.mean(y_valid))
# %%
plt.hist(y_pred, bins=100, alpha=0.5, label='predicted')
plt.title('Prediction distribution')
# %%

print('R2 score:', r2_score(y_valid, y_pred))

# %%
# Do causal inference for avg_rank and n_clicks
x_low_rank = x_valid.copy()
x_high_rank = x_valid.copy()
x_low_rank['avg_rank'] = 10
x_high_rank['avg_rank'] = 5

y_low = model.predict(x_low_rank)
y_high = model.predict(x_high_rank)

effect = np.expm1(y_high) - np.expm1(y_low)
plt.hist(effect)
plt.show()
ATE = effect.mean()
print(ATE)
# %%
# Group the average rank and compare
# Here the groups are [(20, 15], (15, 10], (10, 5], (5, 0]])
rank_pairs = [(20, 15), (15, 10), (10, 5), (5, 1)]
results= []

for r_low, r_high in rank_pairs:
    x_low_rank = x_valid.copy()
    x_high_rank = x_valid.copy()
    x_low_rank['avg_rank'] = r_low
    x_high_rank['avg_rank'] = r_high

    y_low = np.expm1(model.predict(x_low_rank))
    y_high = np.expm1(model.predict(x_high_rank))

    effect = (y_high - y_low).mean()
    results.append({
        'from_rank': r_low,
        'to_rank': r_high,
        'ATE': effect
    })
results_df = pd.DataFrame(results)
print(results_df)
plt.plot(results_df['to_rank'], results_df['ATE'], marker='o')
plt.gca().invert_xaxis()
plt.xlabel('Better Rank')
plt.ylabel('ATE (clink increase)')
plt.show()
# %%
# When rank = 1~30, expected clicks
x_base = x_valid.copy()
ranks = range(1,31)
position_effects = []

for r in ranks:
    x_tmp = x_base.copy()
    x_tmp['avg_rank'] = r
    y_pred = np.expm1(model.predict(x_tmp))
    position_effects.append(y_pred.mean())

plt.figure()
plt.plot(ranks, position_effects, marker='o')
plt.gca().invert_xaxis()
plt.xlabel('Rank')
plt.ylabel('Expected Clicks')
plt.title('Posistion Effect Curve')
plt.show()

# %%
# Compare with observed clicks
df['rank_int']= df['avg_rank'].round().astype(int)
observed = (
    df.groupby('rank_int')['n_clicks'].mean().loc[1:30])
print(observed)
causal = pd.Series(position_effects, index=ranks)

plt.figure()
plt.plot(observed.index, observed.values, label='Observed', marker='o')
plt.plot(causal.index, causal.values, label='Causal Estimate', marker='o')
plt.gca().invert_xaxis()
plt.xlabel('Rank')
plt.ylabel('Expected Clicks')
plt.title('Observed vs Causal Position Effect')
plt.legend()
plt.show()

dft = pd.DataFrame([observed, causal]).T
dft.columns = ['observed', 'causal']
print(dft)
# %%
# Expected Revenue with the assumed fixed commision rate, 18%
commission_rate = 0.18

df_valid = pd.concat([x_valid.reset_index(drop=True), y_valid.reset_index(drop=True)], axis=1)
y_pred = model.predict(x_valid)
y_pred_safe = np.clip(y_pred, 0, y_train.max())
df_valid['pred_clicks'] = y_pred_safe
df_valid['expected_revenue'] = (
    df_valid['pred_clicks'] 
    * df_valid['avg_price'] 
    * commission_rate
)
print(df_valid['expected_revenue'].describe())
df_valid.sort_values('expected_revenue', ascending=False)

# %%
# Build n_cliks prediction model wiout avg_rank
y = np.log1p(df['n_clicks'])
x_features = df.columns.values
drop_cols = ['n_clicks', 'hotel_id', 'avg_rank']
x_features = [c for c in df.columns if c not in drop_cols]
x = df[x_features]
x_train2, x_test2, y_train2, y_test2 = train_test_split(x,y, test_size=0.2, random_state=1)
x_train2, x_valid2, y_train2, y_valid2 = train_test_split(x_train2,y_train2, test_size=0.25, random_state=1)

# %%
no_rank_model = XGBRegressor(n_estimators=100,  learning_rate=0.05)
no_rank_model.fit(x_train2, y_train2)

# %%
y_pred = no_rank_model.predict(x_valid2)
print(len(y_valid2))
print(len(y_pred))
plt.scatter(y_valid2, y_pred, alpha=0.01)
plt.plot([y_valid2.min(), y_valid2.max()],
         [y_valid2.min(), y_valid2.max()],
         color='red')

plt.show()
print(np.mean(y_pred), np.mean(y_valid))
# %%
# plt.hist(y_pred, bins=100, alpha=0.5, label='predicted')
# plt.title('Prediction distribution')
# %%
print('R2 score:', r2_score(y_valid, y_pred))

# %%
# Need to check the correlation between the content score and the avarage rank
print(min(df['content_score']), max(df['content_score']), np.mean(df['content_score']), np.median(df['content_score']))
plt.scatter(df['content_score'], df['avg_rank'], alpha=0.01)
plt.xlabel('content_score')
plt.ylabel('avg_rank')
plt.show()
print(df[['content_score', 'avg_rank']].corr())

# %%
