import pandas as pd
import yfinance as yf
import time
import requests
import argparse
from collections import defaultdict
import backtest
    
# === Slack Integration ===
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/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"  # Replace with your actual Slack webhook
SEND_TO_SLACK = True  # Will be overridden via argparse if --slack-off is used

def send_to_slack(message):
    if not SEND_TO_SLACK:
        print(f"[Slack OFF] {message}")
        return
    payload = {"text": message}
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload)
        if response.status_code != 200:
            print(f"Slack error: {response.text}")
    except Exception as e:
        print(f"Slack exception: {e}")

# === Configuration for each pair (multiple configs per pair supported) ===
CONFIGS = [
    # {
    #     'pair': 'EURUSD', # 10/10
    #     'ema_fast': 20,
    #     'ema_slow': 100,
    #     'pip_size': 0.0001,
    #     'sl_tp': '10/10'
    # },
    {
        'pair': 'EURUSD', # 20/10(TSL)
        'ema_fast': 25,
        'ema_slow': 200,
        'pip_size': 0.0001,
        'sl_tp': '20/10(TSL)'
    },
    # {
    #     'pair': 'USDJPY', # 10/10
    #     'ema_fast': 20,
    #     'ema_slow': 100,
    #     'pip_size': 0.01,
    #     'sl_tp': '10/10'
    # },
    {
        'pair': 'USDJPY', # 30/10(TSL)
        'ema_fast': 25,
        'ema_slow': 200,
        'pip_size': 0.01,
        'sl_tp': '30/10(TSL)'
    },
    # {
    #     'pair': 'GBPJPY', # 20/30
    #     'ema_fast': 20,
    #     'ema_slow': 100,
    #     'pip_size': 0.01,
    #     'sl_tp': '20/30'
    # },
    {
        'pair': 'GBPJPY', # 30/5(TSL)
        'ema_fast': 100,
        'ema_slow': 200,
        'pip_size': 0.01,
        'sl_tp': '30/5(TSL)'
    },
    {
        'pair': 'EURJPY', # 20/5(TSL)
        'ema_fast': 100,
        'ema_slow': 200,
        'pip_size': 0.01,
        'sl_tp': '20/5(TSL)'
    }
]

# === Fetch Data ===
def fetch_data(pair):
    df = yf.download(f"{pair}=X", period='2d', interval='5m', progress=False)
    df.index = pd.to_datetime(df.index)
    df = df.drop(columns=('Volume'))
    df.columns = df.columns.droplevel(1)
    df = df.reset_index()
    df.rename(columns={'index': 'Datetime'}, inplace=True)
    print(f"Fetched {len(df)} rows for {pair}.")
    return df

# === Strategy ===
# def run_strategy(df, config):
#     df = df.copy()
#     df['EMA_FAST'] = df['Close'].ewm(span=config['ema_fast'], adjust=False).mean()
#     df['EMA_SLOW'] = df['Close'].ewm(span=config['ema_slow'], adjust=False).mean()
#     df.dropna(subset=['EMA_FAST', 'EMA_SLOW'], inplace=True)
#     df['TRADE'] = None

#     ema_fast = df['EMA_FAST']
#     ema_slow = df['EMA_SLOW']
#     open_ = df['Open']
#     close = df['Close']

#     crossover_up = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
#     crossover_down = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))
#     candle_bullish = open_ < ema_fast
#     candle_bearish = open_ > ema_fast

#     df.loc[crossover_up & candle_bullish & (close > ema_fast), 'TRADE'] = 'BUY'
#     df.loc[crossover_down & candle_bearish & (close < ema_fast), 'TRADE'] = 'SELL'

#     print(df.tail(5)[['Datetime', 'Open', 'Close', 'EMA_FAST', 'EMA_SLOW', 'TRADE']])
#     return df

# === Main Loop ===
def main():
    global SEND_TO_SLACK

    parser = argparse.ArgumentParser()
    parser.add_argument('--slack-off', action='store_true', help='Disable Slack alerts')
    args = parser.parse_args()
    SEND_TO_SLACK = not args.slack_off
    print(f"Slack notifications are {'enabled' if SEND_TO_SLACK else 'disabled'}.")

    print("Starting signal bot...\n")

    while True:
        print(f"Checking for signals at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}...\n")

        # Organize configs by pair
        configs_by_pair = defaultdict(list)
        for cfg in CONFIGS:
            configs_by_pair[cfg['pair']].append(cfg)

        for pair, configs in configs_by_pair.items():
            try:
                df = fetch_data(pair)
                for config in configs:
                    df_result = backtest.ema_no_crossover_double_minor(df, config)
                    # print(df_result['TRADE'].notnull())
                    print(df_result.tail(5))
                    latest_signal = df_result.iloc[-1]['TRADE']
                    if latest_signal in ['BUY', 'SELL']:
                        dt = df_result.iloc[-1]['Datetime'].strftime('%Y-%m-%d %H:%M')
                        msg = (f"[{pair} | EMA {config['ema_fast']}/{config['ema_slow']}] "
                               f"{latest_signal} signal at {dt} — SL/TP: {config['sl_tp']} ")
                        send_to_slack(msg)
                        print(msg)
                    else:
                        print(f"[{pair} | EMA {config['ema_fast']}/{config['ema_slow']} | SL/TP: {config['sl_tp']}] No signal.")
            except Exception as e:
                print(f"Error processing {pair}: {e}")

        print("\nSleeping for 5 minutes...\n" + "-"*40)
        time.sleep(300)

if __name__ == "__main__":
    main()
