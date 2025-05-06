import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from itertools import product
import pickle

# === Load Data ===
def load_data(pickle_file):
    with open(pickle_file, "rb") as f:
        df = pickle.load(f)
    return df

def load_data_csv(csv_file):
    df = pd.read_csv(csv_file, sep=r'\s+', engine='python')  # separador por espacios
    df['Datetime'] = pd.to_datetime(df['Time'] + ' ' + df['Time2'])  # Combinar fecha y hora
    df.drop(columns=['Time', 'Time2'], inplace=True)  # Eliminamos la antigua columna de hora
    df = df[['Datetime', 'Open', 'High', 'Low', 'Close']]  # Eliminar Volumen y Spread si no es necesario
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(float)
    print(df.info())
    return df


# === EMA Crossover Strategy ===
def ema_crossover_single(df, config):
    df = df.copy()
    df['EMA_FAST'] = ta.ema(df['Close'], length=config['ema_fast'])
    df['EMA_SLOW'] = ta.ema(df['Close'], length=config['ema_slow'])

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    crossover_up = (df['EMA_FAST'].shift(1) > df['EMA_SLOW'].shift(1)) & (df['EMA_FAST'].shift(2) < df['EMA_SLOW'].shift(2))
    crossover_down = (df['EMA_FAST'].shift(1) < df['EMA_SLOW'].shift(1)) & (df['EMA_FAST'].shift(2) > df['EMA_SLOW'].shift(2))

    open_bullish = (df['Low'].shift(1) < df['EMA_FAST'].shift(1)) & (df['Low'].shift(1) < df['EMA_SLOW'].shift(1))
    open_bearish = (df['High'].shift(1) > df['EMA_FAST'].shift(1)) & (df['High'].shift(1) > df['EMA_SLOW'].shift(1))

    close_bullish = (df['Close'].shift(1) > df['EMA_FAST'].shift(1)) & (df['Close'].shift(1) > df['EMA_SLOW'].shift(1))
    close_bearish = (df['Close'].shift(1) < df['EMA_FAST'].shift(1)) & (df['Close'].shift(1) < df['EMA_SLOW'].shift(1))
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.loc[crossover_up & open_bullish & close_bullish, 'TRADE'] = 'BUY'
    df.loc[crossover_down & open_bearish & close_bearish, 'TRADE'] = 'SELL'
    return df

def ema_no_crossover_single(df, config):
    df = df.copy()
    df['EMA_FAST'] = ta.ema(df['Close'], length=config['ema_fast'])
    df['EMA_SLOW'] = ta.ema(df['Close'], length=config['ema_slow'])

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    open_bullish = (df['Low'].shift(1) < df['EMA_FAST'].shift(1)) & (df['Low'].shift(1) < df['EMA_SLOW'].shift(1))
    open_bearish = (df['High'].shift(1) > df['EMA_FAST'].shift(1)) & (df['High'].shift(1) > df['EMA_SLOW'].shift(1))

    close_bullish = (df['Close'].shift(1) > df['EMA_FAST'].shift(1)) & (df['Close'].shift(1) > df['EMA_SLOW'].shift(1))
    close_bearish = (df['Close'].shift(1) < df['EMA_FAST'].shift(1)) & (df['Close'].shift(1) < df['EMA_SLOW'].shift(1))
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.loc[open_bullish & close_bullish, 'TRADE'] = 'BUY'
    df.loc[open_bearish & close_bearish, 'TRADE'] = 'SELL'
    return df

def ema_crossover_double_minor(df, config):
    # Crossover strategy using two EMAs and a double candle same direction filter
    df = df.copy()
    df['EMA_FAST'] = ta.ema(df['Close'], length=config['ema_fast'])
    df['EMA_SLOW'] = ta.ema(df['Close'], length=config['ema_slow'])

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    crossover_up = (df['EMA_FAST'].shift(2) > df['EMA_SLOW'].shift(2)) & (df['EMA_FAST'].shift(3) < df['EMA_SLOW'].shift(3))
    crossover_down = (df['EMA_FAST'].shift(2) < df['EMA_SLOW'].shift(2)) & (df['EMA_FAST'].shift(3) > df['EMA_SLOW'].shift(3))

    open_bullish = (df['Low'].shift(2) < df['EMA_FAST'].shift(2)) & (df['Low'].shift(2) < df['EMA_SLOW'].shift(2))
    open_bearish = (df['High'].shift(2) > df['EMA_FAST'].shift(2)) & (df['High'].shift(2) > df['EMA_SLOW'].shift(2))

    close_bullish = (df['Close'].shift(2) > df['EMA_FAST'].shift(2)) & (df['Close'].shift(2) > df['EMA_SLOW'].shift(2))
    close_bearish = (df['Close'].shift(2) < df['EMA_FAST'].shift(2)) & (df['Close'].shift(2) < df['EMA_SLOW'].shift(2))
    
    double_bullish = (df['Open'].shift(2) < df['Close'].shift(2)) & (df['Open'].shift(1) < df['Close'].shift(1))
    double_bearish = (df['Open'].shift(2) > df['Close'].shift(2)) & (df['Open'].shift(1) > df['Close'].shift(1))

    double_bullish_minor = ((df['Close'].shift(2) - df['Open'].shift(2)) < (df['Close'].shift(1) - df['Open'].shift(1)))
    double_bearish_minor = ((df['Open'].shift(2) - df['Close'].shift(2)) < (df['Open'].shift(1) - df['Close'].shift(1)))

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.loc[crossover_up & open_bullish & close_bullish & double_bullish & double_bullish_minor, 'TRADE'] = 'BUY'
    df.loc[crossover_down & open_bearish & close_bearish & double_bearish & double_bearish_minor, 'TRADE'] = 'SELL'
    return df

def ema_no_crossover_double_minor(df, config):
    # Crossover strategy using two EMAs and a double candle same direction filter
    df = df.copy()
    df['EMA_FAST'] = ta.ema(df['Close'], length=config['ema_fast'])
    df['EMA_SLOW'] = ta.ema(df['Close'], length=config['ema_slow'])

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    open_bullish = (df['Low'].shift(2) < df['EMA_FAST'].shift(2)) & (df['Low'].shift(2) < df['EMA_SLOW'].shift(2))
    open_bearish = (df['High'].shift(2) > df['EMA_FAST'].shift(2)) & (df['High'].shift(2) > df['EMA_SLOW'].shift(2))

    close_bullish = (df['Close'].shift(2) > df['EMA_FAST'].shift(2)) & (df['Close'].shift(2) > df['EMA_SLOW'].shift(2))
    close_bearish = (df['Close'].shift(2) < df['EMA_FAST'].shift(2)) & (df['Close'].shift(2) < df['EMA_SLOW'].shift(2))
    
    double_bullish = (df['Open'].shift(2) < df['Close'].shift(2)) & (df['Open'].shift(1) < df['Close'].shift(1))
    double_bearish = (df['Open'].shift(2) > df['Close'].shift(2)) & (df['Open'].shift(1) > df['Close'].shift(1))

    double_bullish_minor = ((df['Close'].shift(2) - df['Open'].shift(2)) < (df['Close'].shift(1) - df['Open'].shift(1)))
    double_bearish_minor = ((df['Open'].shift(2) - df['Close'].shift(2)) < (df['Open'].shift(1) - df['Close'].shift(1)))

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.loc[open_bullish & close_bullish & double_bullish & double_bullish_minor, 'TRADE'] = 'BUY'
    df.loc[open_bearish & close_bearish & double_bearish & double_bearish_minor, 'TRADE'] = 'SELL'
    return df

def ema_no_crossover_double_minor_with_ema_trend(df, config):
    # Crossover strategy using two EMAs trending in the same direction and a double candle same direction filter, second candle smaller than first
    df = df.copy()
    df['EMA_FAST'] = ta.ema(df['Close'], length=config['ema_fast'])
    df['EMA_SLOW'] = ta.ema(df['Close'], length=config['ema_slow'])

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    ema_bullish = (df['EMA_FAST'] > df['EMA_FAST'].shift(1)) &  (df['EMA_FAST'].shift(1) > df['EMA_FAST'].shift(2)) & (df['EMA_SLOW'] > df['EMA_SLOW'].shift(1)) & (df['EMA_SLOW'].shift(1) > df['EMA_SLOW'].shift(2))
    ema_bearish = (df['EMA_FAST'] < df['EMA_FAST'].shift(1)) &  (df['EMA_FAST'].shift(1) < df['EMA_FAST'].shift(2)) & (df['EMA_SLOW'] < df['EMA_SLOW'].shift(1)) & (df['EMA_SLOW'].shift(1) < df['EMA_SLOW'].shift(2))

    open_bullish = (df['Low'].shift(2) < df['EMA_FAST'].shift(2)) & (df['Low'].shift(2) < df['EMA_SLOW'].shift(2))
    open_bearish = (df['High'].shift(2) > df['EMA_FAST'].shift(2)) & (df['High'].shift(2) > df['EMA_SLOW'].shift(2))

    close_bullish = (df['Close'].shift(2) > df['EMA_FAST'].shift(2)) & (df['Close'].shift(2) > df['EMA_SLOW'].shift(2))
    close_bearish = (df['Close'].shift(2) < df['EMA_FAST'].shift(2)) & (df['Close'].shift(2) < df['EMA_SLOW'].shift(2))
    
    double_bullish = (df['Open'].shift(2) < df['Close'].shift(2)) & (df['Open'].shift(1) < df['Close'].shift(1))
    double_bearish = (df['Open'].shift(2) > df['Close'].shift(2)) & (df['Open'].shift(1) > df['Close'].shift(1))

    double_bullish_minor = ((df['Close'].shift(2) - df['Open'].shift(2)) < (df['Close'].shift(1) - df['Open'].shift(1)))
    double_bearish_minor = ((df['Open'].shift(2) - df['Close'].shift(2)) < (df['Open'].shift(1) - df['Close'].shift(1)))

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.loc[ema_bullish & open_bullish & close_bullish & double_bullish & double_bullish_minor, 'TRADE'] = 'BUY'
    df.loc[ema_bearish & open_bearish & close_bearish & double_bearish & double_bearish_minor, 'TRADE'] = 'SELL'
    return df

def ema_crossover_double_minor_with_ema_trend(df, config):
    # Crossover strategy using two EMAs trending in the same direction and a double candle same direction filter, second candle smaller than first
    df = df.copy()
    df['EMA_FAST'] = ta.ema(df['Close'], length=config['ema_fast'])
    df['EMA_SLOW'] = ta.ema(df['Close'], length=config['ema_slow'])

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    crossover_up = (df['EMA_FAST'].shift(2) > df['EMA_SLOW'].shift(2)) & (df['EMA_FAST'].shift(3) < df['EMA_SLOW'].shift(3))
    crossover_down = (df['EMA_FAST'].shift(2) < df['EMA_SLOW'].shift(2)) & (df['EMA_FAST'].shift(3) > df['EMA_SLOW'].shift(3))

    ema_bullish = (df['EMA_FAST'] > df['EMA_FAST'].shift(1)) &  (df['EMA_FAST'].shift(1) > df['EMA_FAST'].shift(2)) & (df['EMA_SLOW'] > df['EMA_SLOW'].shift(1)) & (df['EMA_SLOW'].shift(1) > df['EMA_SLOW'].shift(2))
    ema_bearish = (df['EMA_FAST'] < df['EMA_FAST'].shift(1)) &  (df['EMA_FAST'].shift(1) < df['EMA_FAST'].shift(2)) & (df['EMA_SLOW'] < df['EMA_SLOW'].shift(1)) & (df['EMA_SLOW'].shift(1) < df['EMA_SLOW'].shift(2))

    open_bullish = (df['Low'].shift(2) < df['EMA_FAST'].shift(2)) & (df['Low'].shift(2) < df['EMA_SLOW'].shift(2))
    open_bearish = (df['High'].shift(2) > df['EMA_FAST'].shift(2)) & (df['High'].shift(2) > df['EMA_SLOW'].shift(2))

    close_bullish = (df['Close'].shift(2) > df['EMA_FAST'].shift(2)) & (df['Close'].shift(2) > df['EMA_SLOW'].shift(2))
    close_bearish = (df['Close'].shift(2) < df['EMA_FAST'].shift(2)) & (df['Close'].shift(2) < df['EMA_SLOW'].shift(2))
    
    double_bullish = (df['Open'].shift(2) < df['Close'].shift(2)) & (df['Open'].shift(1) < df['Close'].shift(1))
    double_bearish = (df['Open'].shift(2) > df['Close'].shift(2)) & (df['Open'].shift(1) > df['Close'].shift(1))

    double_bullish_minor = ((df['Close'].shift(2) - df['Open'].shift(2)) < (df['Close'].shift(1) - df['Open'].shift(1)))
    double_bearish_minor = ((df['Open'].shift(2) - df['Close'].shift(2)) < (df['Open'].shift(1) - df['Close'].shift(1)))

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.loc[crossover_up & ema_bullish & open_bullish & close_bullish & double_bullish & double_bullish_minor, 'TRADE'] = 'BUY'
    df.loc[crossover_down & ema_bearish & open_bearish & close_bearish & double_bearish & double_bearish_minor, 'TRADE'] = 'SELL'
    return df

# === Execute Strategy ===
def execute_strategy(df, config):
    df = df.copy()
    trades = df.index[df['TRADE'].notnull()].tolist()

    for trade in trades:
        trade_type = df.at[trade, 'TRADE']
        open_price = df.at[trade, 'Open']
        sl = config['sl_pips'] * config['pip_size']
        tp = config['tp_pips'] * config['pip_size']
        # print(f"Trade: {trade}, Type: {trade_type}, Open Price: {open_price}, SL: {sl}, TP: {tp}")
        trade_open = True
        while trade_open:
            if trade_type == 'BUY':
                for i in range(trade, len(df)):
                    if df.at[i, 'High'] > open_price + tp:
                        # print(f"Trade {trade} hit TP")
                        trade_open = False
                        df.at[trade, 'PIPS'] = tp
                        df.at[trade, 'TRADE_RESULT'] = 'TP'
                        break
                    elif df.at[i, 'Low'] < open_price - sl:
                        # print(f"Trade {trade} hit SL")
                        trade_open = False
                        df.at[trade, 'PIPS'] = sl * -1
                        df.at[trade, 'TRADE_RESULT'] = 'SL'
                        break
                    elif i == len(df) - 1:
                        trade_open = False
            elif trade_type == 'SELL':
                for i in range(trade, len(df)):
                    if df.at[i, 'Low'] < open_price - tp:
                        # print(f"Trade {trade} hit TP")
                        trade_open = False
                        df.at[trade, 'PIPS'] = tp
                        df.at[trade, 'TRADE_RESULT'] = 'TP'
                        break
                    elif df.at[i, 'High'] > open_price + sl:
                        # print(f"Trade {trade} hit SL")
                        trade_open = False
                        df.at[trade, 'PIPS'] = sl * -1
                        df.at[trade, 'TRADE_RESULT'] = 'SL'
                        break
                    elif i == len(df) - 1:
                        trade_open = False
    
    return df

def execute_strategy_with_tl(df, config):
    df = df.copy()
    trades = df.index[df['TRADE'].notnull()].tolist()

    for trade in trades:
        trade_type = df.at[trade, 'TRADE']
        open_price = df.at[trade, 'Open']
        # print(f"Trade: {trade}, Type: {trade_type}, Open Price: {open_price}, SL: {sl}, TP: {tp}")
        trade_open = True
        while trade_open:
            if trade_type == 'BUY':
                sl = open_price - config['sl_pips'] * config['pip_size']
                for i in range(trade, len(df)):
                    if df.at[i, 'Low'] < sl:
                        trade_open = False
                        df.at[trade, 'PIPS'] = sl - open_price
                        df.at[trade, 'CLOSE_PRICE'] = sl
                        if df.at[trade, 'PIPS'] > 0:
                            df.at[trade, 'TRADE_RESULT'] = 'TP'
                        else:
                            df.at[trade, 'TRADE_RESULT'] = 'SL'
                        break
                    elif i == len(df) - 1:
                        trade_open = False
                    else:
                        new_sl = df.at[i, 'Low'] - config['pip_size'] * config['tsl_pips']
                        if new_sl > open_price:
                            sl = new_sl

            elif trade_type == 'SELL':
                sl = open_price + config['sl_pips'] * config['pip_size']
                for i in range(trade, len(df)):
                    if df.at[i, 'High'] > sl:
                        trade_open = False
                        df.at[trade, 'PIPS'] = open_price - sl
                        df.at[trade, 'CLOSE_PRICE'] = sl
                        if df.at[trade, 'PIPS'] > 0:
                            df.at[trade, 'TRADE_RESULT'] = 'TP'
                        else:
                            df.at[trade, 'TRADE_RESULT'] = 'SL'
                        break
                    elif i == len(df) - 1:
                        trade_open = False
                    else:
                        new_sl = df.at[i, 'High'] + config['pip_size'] * config['tsl_pips']
                        if new_sl < open_price:
                            sl = new_sl
    
    return df

# === Summary Printer ===
def print_strategy_summary(df, config):
    trades = df[df['PIPS'].notnull()].copy()
    wins = trades[trades['PIPS'] > 0]
    total_pips = trades['PIPS'].sum() / config['pip_size']
    win_rate = len(wins) / len(trades) * 100 if len(trades) else 0

    # PnL in dollars assuming 1% risk per trade on 100$ starting balance
    balance = 100.0
    risk_per_trade = 0.01 * balance
    for pnl in trades['PIPS']:
        reward_risk = pnl / (config['sl_pips'] * config['pip_size'])
        balance += reward_risk * risk_per_trade
    total_return_pct = ((balance - 100.0) / 100.0) * 100

    print("\n=== Strategy Test ===")
    print(f"Parameters: {config}")
    print(f"Trades: {len(trades)}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Total pips: {total_pips:.2f}")
    print(f"Ending Balance: ${balance:.2f} ({total_return_pct:.2f}%)")

    summary = {
        **config,
        'trades': len(trades),
        'win_rate': round(win_rate, 2),
        'total_pips': round(total_pips, 2),
        'ending_balance': round(balance, 2),
        'return_pct': round(total_return_pct, 2),
    }
    
    return summary

# === Grid Search for Parameters ===
def run_grid_search(df, param_grid):
    # Generate all combinations of parameters
    param_combinations = list(product(*param_grid.values()))
    all_results = []

    for params in param_combinations:
        config = dict(zip(param_grid.keys(), params))

        # Skip test case if fast EMA >= slow EMA
        if config['ema_fast'] >= config['ema_slow']:
            continue

        # Skip test case if sl_pips > tp_pips
        # if config['sl_pips'] > config['tp_pips']:
        #     continue

        df_trades = ema_crossover_double_minor_with_ema_trend(df, config)
        # df_execute = execute_strategy(df_trades, config)
        df_execute = execute_strategy_with_tl(df_trades, config)
        # print(df_execute[df_execute['TRADE_RESULT'].notnull()].tail(20))
        df_summary = print_strategy_summary(df_execute, config)
        all_results.append(df_summary)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("test_case_summary.csv", index=False)
    print("\nTest case summary saved to test_case_summary.csv")


# === Main Execution ===
if __name__ == '__main__':
    # df = load_data("data/EURUSD-60d5m.pkl")  # Replace with your pickle path

    # EURUSD
    df = load_data_csv("data/EURUSD_M5.csv")  # Replace with your CSV file path
    # df = load_data_csv("data/EURUSD_M30.csv")  # Replace with your CSV file path
    # df = load_data_csv("data/EURUSD_H1.csv")  # Replace with your CSV file path

    # df = load_data_csv("data/EURGBP_M5.csv")  # Replace with your CSV file path

    # df = load_data_csv("data/USDJPY_M5.csv")  # Replace with your CSV file path

    # df = load_data_csv("data/EURJPY_M5.csv")  # Replace with your CSV file path

    # df = load_data_csv("data/GBPJPY_M5.csv")  # Replace with your CSV file path

    param_grid = {
        'ema_fast': [10, 20, 25, 30, 50, 100],
        'ema_slow': [20, 25, 50, 60, 100, 200],
        # 'ema_fast': [50],
        # 'ema_slow': [200],
        # 'sl_pips': [10],
        'sl_pips': [5, 10, 15, 20, 30, 40, 50],
        # 'sl_pips': [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'tsl_pips': [5, 10, 15, 20, 30, 40, 50],
        # 'tp_pips': [10, 15, 20, 30, 40, 50],
        'pip_size': [0.0001]
    }

    run_grid_search(df, param_grid)

    # DEFAULT_CONFIG = [
    # {
    #     'ema_fast': 20,
    #     'ema_slow': 100,
    #     'sl_pips': 10,
    #     'tp_pips': 10,
    #     'pip_size': 0.0001,
    #     'trailing_stop': False,
    #     'breakeven': False
    # }
    # ]
    
    # for config in DEFAULT_CONFIG:
    #     df_trades = ema_crossover_double(df, config)
    #     df_execute = execute_strategy(df_trades, config)
    #     print_strategy_summary(df_execute, config)
        # print(df_execute[df_execute['TRADE_RESULT'] == 'TP'].count())
        # print(df_execute[df_execute['PIPS'].notnull()].tail())
        # print(df_execute[df_execute['TRADE_RESULT'] == 'SL'].count())
        # print(df_execute['PIPS'].sum() / 0.0001)

    # print(df_trades.tail(20))
    # print(df_trades[df_trades['TRADE'] == 'BUY'].tail())
    # print(df_trades[df_trades['TRADE'] == 'SELL'].tail())

    # print(df_execute.tail(20))


    # run_grid_search(df)