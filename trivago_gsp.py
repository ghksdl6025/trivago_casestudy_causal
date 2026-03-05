from typing import List, Mapping

import pandas as pd


def simulate_gsp_auction(bids_data: List[Mapping[str, float]]) -> pd.DataFrame:
    """
    Simulate a generalized second-price (GSP) auction.

    Expected input format:
    [{"advertiser": "A", "bid": 10.0, "ctr": 0.05}, ...]
    """
    df = pd.DataFrame(bids_data)
    df = df.sort_values(by="bid", ascending=False).reset_index(drop=True)

    # Price per click is next bidder's bid plus minimum increment.
    df["price_per_click"] = df["bid"].shift(-1).fillna(0) + 0.01
    df["expected_revenue"] = df["price_per_click"] * df["ctr"]
    return df


def simulate_pps_auction(data: List[Mapping[str, float]]) -> pd.DataFrame:
    """
    Simulate a simplified PPS-style ranking based on expected revenue.

    expected_revenue = commission * price * cvr * (1 - cancel_rate)
    """
    df = pd.DataFrame(data)
    df["expected_revenue"] = (
        df["commission"] * df["price"] * df["cvr"] * (1 - df["cancel_rate"])
    )
    return df.sort_values(by="expected_revenue", ascending=False).reset_index(drop=True)


def main() -> None:
    gsp_data = [
        {"advertiser": "A", "bid": 5.5, "ctr": 0.12},
        {"advertiser": "B", "bid": 7.2, "ctr": 0.10},
        {"advertiser": "C", "bid": 4.8, "ctr": 0.15},
        {"advertiser": "D", "bid": 6.1, "ctr": 0.08},
    ]

    gsp_result = simulate_gsp_auction(gsp_data)
    print("--- Auction Result (GSP) ---")
    print(gsp_result[["advertiser", "bid", "price_per_click", "expected_revenue"]])
    print(f"Total Expected Revenue: {gsp_result['expected_revenue'].sum():.2f}")

    pps_data = [
        {"advertiser": "A", "commission": 0.10, "price": 200, "cvr": 0.05, "cancel_rate": 0.20},
        {"advertiser": "B", "commission": 0.12, "price": 210, "cvr": 0.04, "cancel_rate": 0.10},
        {"advertiser": "C", "commission": 0.18, "price": 400, "cvr": 0.02, "cancel_rate": 0.05},
        {"advertiser": "D", "commission": 0.20, "price": 100, "cvr": 0.06, "cancel_rate": 0.01},
    ]

    pps_result = simulate_pps_auction(pps_data)
    print("--- Auction Result (PPS) ---")
    print(pps_result[["advertiser", "expected_revenue"]])
    print(f"Total Expected Revenue: {pps_result['expected_revenue'].sum():.2f}")


if __name__ == "__main__":
    main()
