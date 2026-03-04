# Trivago Case Study

## Project Overview 📌

This repo is a small case study built on a Trivago-style hotel dataset. The main goal is to understand what drives hotel clicks, look at how ranking and content quality affect performance, and try a few different approaches ranging from basic modeling to causal inference.

## Dataset 📊

The data includes hotel-level features such as `content_score`, `n_images`, `distance_to_center`, `avg_rating`, `stars`, `n_reviews`, `avg_rank`, `avg_price`, and `avg_saving_percent`. In the training set, the target is `n_clicks`.

The dataset was taken from [`carlocontaldi/trivago_case_study`](https://github.com/carlocontaldi/trivago_case_study).

Files in the `data/` directory:

- `train_set.csv`: training data with features and `n_clicks`
- `test_set.csv`: test data without `n_clicks`
- `sample_submission.csv`: sample output format

## Project Structure 🌳

```text
trivago_case_study-master/
|-- data/
|   |-- sample_submission.csv
|   |-- test_set.csv
|   `-- train_set.csv
|-- data_exploration.py
|-- dowhy_practice.py
|-- trivago_gsp.py
`-- README.md
```

## Python Scripts 📜

### `data_exploration.py`

This file is mainly for EDA and baseline modeling. It checks basic distributions, trains an `XGBRegressor` for click prediction, and runs simple what-if simulations for changes in `avg_rank`.

### `dowhy_practice.py`

This file is focused on causal inference. It uses `DoWhy` and `econml` to estimate the effect of `content_score` on `n_clicks`, run a few refutation tests, and look at which hotel segments may benefit more from content improvements.

### `trivago_gsp.py`

This file contains small auction simulations. It compares a simple GSP setup with a PPS-style ranking example to see how advertiser ordering and expected revenue can change under different rules.
