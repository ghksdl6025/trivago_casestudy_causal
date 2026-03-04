import pandas as pd

def simulate_gsp_auction(bids_data):
    """
    bids data:  Dictionary in the list [{'advertiser': 'A', 'bid': 10, 'ctr':0.05}]
    """
    # 1. Convert bids data to DataFrame and sort by bid amount in ascending order
    df = pd.DataFrame(bids_data)
    df = df.sort_values(by='bid', ascending=False).reset_index(drop=True)
    
    # 2. GSP Auction Logic (Price per click)
    # The price of ith person = the bid of the (i+1)th person + minimum increment (0.01)
    # The last ranking is reserve price or 0 if no reserve price is set
    df['price_per_click'] = df['bid'].shift(-1).fillna(0) + 0.01

    # 3. Calculate expected revenue
    # Revenue = Price per click * Click Through Rate (CTR)
    df['expected_revenue'] = df['price_per_click'] * df['ctr']

    return df

def simulate_pps_auction(data):
    """
    Simulate a Pay-Per-Share (PPS) auction.
    In a PPS auction, advertisers pay based on the number of shares they receive.
    The price per share is determined by the second-highest bid.
    """
    df = pd.DataFrame(data)

    # 1. Calculate expected revenue (Commision * Price * CVR * (1-Cancel_Rate))
    df['expected_revenue'] = (
        df['commission'] * df['price'] * df['cvr'] * (1 - df['cancel_rate'])
    )
    df = df.sort_values(by='expected_revenue', ascending=False).reset_index(drop=True)

    return df

data = [
    {'advertiser': 'A', 'bid': 5.5, 'ctr': 0.12},
    {'advertiser': 'B', 'bid': 7.2, 'ctr': 0.10},
    {'advertiser': 'C', 'bid': 4.8, 'ctr': 0.15},
    {'advertiser': 'D', 'bid': 6.1, 'ctr': 0.08}
]

result = simulate_gsp_auction(data)
print('--- Auction Result (GSP) ---')
print(result[['advertiser', 'bid', 'price_per_click', 'expected_revenue']])

total_revenue = result['expected_revenue'].sum()
print(f'Total Expected Revenue: {total_revenue:.2f}')

pps_data = [
    {'advertiser': 'A', 'commission': 0.1, 'price': 200, 'cvr': 0.05, 'cancel_rate': 0.2},
    {'advertiser': 'B', 'commission': 0.12, 'price': 210, 'cvr': 0.04, 'cancel_rate': 0.1},
    {'advertiser': 'C', 'commission': 0.18, 'price': 400, 'cvr': 0.02, 'cancel_rate': 0.05},
    {'advertiser': 'D', 'commission': 0.20, 'price': 100, 'cvr': 0.06, 'cancel_rate': 0.01}
]
pps_result = simulate_pps_auction(pps_data)
print('--- Auction Result (PPS) ---')
print(pps_result[['advertiser', 'expected_revenue']])
total_revenue = pps_result['expected_revenue'].sum()
print(f'Total Expected Revenue: {total_revenue:.2f}')
