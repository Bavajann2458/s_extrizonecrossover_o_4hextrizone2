import pandas as pd
import numpy as np
import time
import requests
from tabulate import tabulate

# --- 1. CONFIGURATION & CREDENTIALS ---
TELEGRAM_BOT_TOKEN = "8626580414:AAGg7gzVP-twcFg4e0LFec8vgsP6wfCOC5g"
TELEGRAM_CHAT_ID = "8307615821"

TIMEFRAMES = ['5m', '15m', '30m', '1h', '2h']  # Lower timeframes for precision entries
TOP_N_PAIRS = 50  # Top high-liquidity coins for speed

# Keeps track of sent alerts in this session to prevent spamming the same setup every sweep
ALERTS_SENT = set()

def send_telegram_alert(title, message_body, alert_type="TRADE"):
    """Sends an automated push notification to your Telegram bot."""
    icons = {"INFO": "ℹ️", "SUCCESS": "🟢", "TRADE": "🎯", "ALERT": "🚨"}
    icon = icons.get(alert_type, "📢")
    
    formatted_message = (
        f"{icon} *{title}* {icon}\n"
        f"-----------------------------------\n"
        f"{message_body}\n"
        f"-----------------------------------\n"
        f"🤖 *Bot Status:* 4H Stoch Cross (>85/<15) + LTF Cross (>94/<6) + OB"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": formatted_message, "parse_mode": "Markdown"}
    
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"[TELEGRAM ERROR] {e}")

def fetch_top_usdt_pairs():
    """Fetches top USDT pairs directly via unblocked HTTP request."""
    try:
        url = "https://data-api.binance.vision/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        usdt_tickers = [
            t for t in data 
            if t['symbol'].endswith('USDT') and float(t.get('quoteVolume', 0)) > 0
        ]
        sorted_tickers = sorted(usdt_tickers, key=lambda x: float(x['quoteVolume']), reverse=True)
        
        sorted_pairs = []
        for t in sorted_tickers[:TOP_N_PAIRS]:
            raw = t['symbol']
            formatted = raw[:-4] + '/' + raw[-4:]
            sorted_pairs.append(formatted)
            
        return sorted_pairs
    except Exception as e:
        print(f"[ERROR] Direct ticker fetch failed: {e}")
        return []

def fetch_ohlcv_direct(symbol, interval, limit=60):
    """Fetches candlestick data directly via unblocked HTTP request with strict timeout."""
    try:
        raw_symbol = symbol.replace('/', '')
        url = "https://data-api.binance.vision/api/v3/klines"
        params = {'symbol': raw_symbol, 'interval': interval, 'limit': limit}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if not isinstance(data, list):
            return []
            
        formatted = []
        for k in data:
            formatted.append([
                int(k[0]),          # Timestamp
                float(k[1]),        # Open
                float(k[2]),        # High
                float(k[3]),        # Low
                float(k[4]),        # Close
                float(k[5])         # Volume
            ])
        return formatted
    except Exception:
        return []

def calculate_stoch_rsi(df, period=14, smooth_k=3, smooth_d=3):
    """Calculates Stochastic RSI (%K and %D lines) from raw OHLCV data."""
    close = df['close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    min_rsi = rsi.rolling(window=period).min()
    max_rsi = rsi.rolling(window=period).max()
    
    stoch_rsi = 100 * (rsi - min_rsi) / (max_rsi - min_rsi)
    df['stoch_k'] = stoch_rsi.rolling(window=smooth_k).mean()
    df['stoch_d'] = df['stoch_k'].rolling(window=smooth_d).mean()
    return df

def check_4h_macro_crossover(symbol, is_long):
    """
    Checks 4-hour timeframe Stochastic RSI fresh crossover condition (4h A):
    - For Longs: Fresh crossover upwards (%K crosses above %D) within below 15 zone.
    - For Shorts: Fresh crossover downwards (%K crosses below %D) within above 85 zone.
    """
    try:
        ohlcv_4h = fetch_ohlcv_direct(symbol, '4h', limit=50)
        if len(ohlcv_4h) < 30:
            return False
            
        df_4h = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_4h = calculate_stoch_rsi(df_4h)
        
        curr = df_4h.iloc[-1]
        prev = df_4h.iloc[-2]
        prev2 = df_4h.iloc[-3] if len(df_4h) > 2 else prev
        
        k, d = curr['stoch_k'], curr['stoch_d']
        prev_k, prev_d = prev['stoch_k'], prev['stoch_d']
        prev2_k, prev2_d = prev2['stoch_k'], prev2['stoch_d']
        
        if pd.isna(k) or pd.isna(d):
            return False
            
        if is_long:
            # 4H Long: Fresh cross upwards within below 15 zone
            is_cross_up = (prev_k <= prev_d and k > d) or (prev2_k <= prev2_d and prev_k > prev_d)
            in_zone = (k < 15 or prev_k < 15 or prev2_k < 15) and (d < 15)
            return is_cross_up and in_zone
        else:
            # 4H Short: Fresh cross downwards within above 85 zone
            is_cross_down = (prev_k >= prev_d and k < d) or (prev2_k >= prev2_d and prev_k < prev_d)
            in_zone = (k > 85 or prev_k > 85 or prev2_k > 85) and (d > 85)
            return is_cross_down and in_zone
    except Exception:
        return False

def check_precision_order_block(df, is_long):
    """
    Identifies a true institutional Order Block with exact starting and ending prices:
    - Bullish OB: Last bearish (red) candle before an impulsive bullish expansion.
    - Bearish OB: Last bullish (green) candle before an impulsive bearish expansion.
    Returns: (IsValid, OB_Start_Low, OB_End_High, OB_Type)
    """
    if len(df) < 30:
        return False, 0.0, 0.0, ""
        
    curr_price = df.iloc[-1]['close']
    
    # Scan recent candles for institutional displacement
    for i in range(len(df) - 4, max(5, len(df) - 25), -1):
        prev_row = df.iloc[i]     # Potential Order Block candle
        next_row = df.iloc[i+1]   # Impulsive expansion candle
        
        if is_long:
            # Bullish OB: Red candle before green expansion
            if prev_row['close'] < prev_row['open'] and next_row['close'] > next_row['open']:
                body_prev = abs(prev_row['close'] - prev_row['open'])
                body_next = abs(next_row['close'] - next_row['open'])
                
                if body_next >= (body_prev * 0.9):
                    ob_low = prev_row['low']
                    ob_high = prev_row['high']
                    # Verify price is currently testing or respecting this zone
                    if ob_low <= curr_price <= (ob_high * 1.02) and curr_price >= (ob_low * 0.98):
                        return True, ob_low, ob_high, "Bullish Order Block"
        else:
            # Bearish OB: Green candle before red expansion
            if prev_row['close'] > prev_row['open'] and next_row['close'] < next_row['open']:
                body_prev = abs(prev_row['close'] - prev_row['open'])
                body_next = abs(next_row['close'] - next_row['open'])
                
                if body_next >= (body_prev * 0.9):
                    ob_low = prev_row['low']
                    ob_high = prev_row['high']
                    # Verify price is currently testing this zone from below
                    if (ob_low * 0.98) <= curr_price <= ob_high:
                        return True, ob_low, ob_high, "Bearish Order Block"
                        
    return False, 0.0, 0.0, ""

def scan_markets():
    print("="*70)
    print(" WOWBOT: 4H Stoch Cross (>85/<15) + LTF Cross (>94/<6) + OB Scanner ")
    print("="*70)
    
    pairs = fetch_top_usdt_pairs()
    if not pairs:
        print("[WARNING] No pairs fetched. Retrying in 10s...")
        time.sleep(10)
        return

    print(f"\n[SCANNING] Sweeping {len(pairs)} Binance coins across timeframes: {TIMEFRAMES}\n")
    
    scan_results = []
    for idx, symbol in enumerate(pairs):
        print(f"[{idx+1}/{len(pairs)}] Scanning {symbol}...")
        
        for tf in TIMEFRAMES:
            try:
                ohlcv = fetch_ohlcv_direct(symbol, tf, limit=50)
                if len(ohlcv) < 35:
                    continue
                    
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df = calculate_stoch_rsi(df)
                
                curr = df.iloc[-1]
                prev = df.iloc[-2]
                prev2 = df.iloc[-3] if len(df) > 2 else prev
                
                k, d = curr['stoch_k'], curr['stoch_d']
                prev_k, prev_d = prev['stoch_k'], prev['stoch_d']
                prev2_k, prev2_d = prev2['stoch_k'], prev2['stoch_d']
                price = curr['close']
                
                if pd.isna(k) or pd.isna(d):
                    continue
                
                # --- LOWER TIMEFRAME SETUP CHECK ---
                # Long: Fresh cross upwards within below 6 zone
                is_long_crossover = (
                    ((prev_k <= prev_d and k > d) or (prev2_k <= prev2_d and prev_k > prev_d)) and
                    (k < 6 or prev_k < 6 or prev2_k < 6) and (d < 6)
                )
                
                # Short: Fresh cross downwards within above 94 zone
                is_short_crossover = (
                    ((prev_k >= prev_d and k < d) or (prev2_k >= prev2_d and prev2_k < prev_d)) and
                    (k > 94 or prev_k > 94 or prev2_k > 94) and (d > 94)
                )
                
                if is_long_crossover or is_short_crossover:
                    is_long = is_long_crossover
                    
                    # 1. Precision Order Block Validation on Lower Timeframe
                    has_ob, ob_low, ob_high, ob_type = check_precision_order_block(df, is_long)
                    if not has_ob:
                        continue
                        
                    # 2. Synchronize with 4H Macro Crossover Setup (4h A)
                    macro_synced = check_4h_macro_crossover(symbol, is_long)
                    if not macro_synced:
                        continue

                    ob_size = abs(ob_high - ob_low)

                    # 3. Risk Management: Calculate Stop Loss and Take Profit targets based on OB
                    if is_long:
                        stop_loss = ob_low * 0.995  # 0.5% buffer below OB start low
                        risk = price - stop_loss
                        tp1 = price + (risk * 2.0)   # 1:2 RRR
                        tp2 = price + (risk * 3.5)   # 1:3.5 RRR
                        direction = f"🟢 LONG"
                        macro_desc = "4H Cross Up within < 15 Zone"
                        ltf_desc = "LTF Cross Up within < 6 Zone + Bullish OB"
                    else:
                        stop_loss = ob_high * 1.005 # 0.5% buffer above OB end high
                        risk = stop_loss - price
                        tp1 = price - (risk * 2.0)   # 1:2 RRR
                        tp2 = price - (risk * 3.5)   # 1:3.5 RRR
                        direction = f"🔴 SHORT"
                        macro_desc = "4H Cross Down within > 85 Zone"
                        ltf_desc = "LTF Cross Down within > 94 Zone + Bearish OB"

                    alert_key = f"{symbol}_{tf}"
                    p_str = lambda p: f"{p:,.4f}" if p < 1 else f"{p:,.2f}"
                    
                    record = {
                        'Coin': symbol,
                        'TF': tf,
                        'Setup': direction,
                        'OB Type': ob_type,
                        'OB Start': p_str(ob_low),
                        'OB End': p_str(ob_high),
                        'OB Size': p_str(ob_size),
                        'Entry': p_str(price),
                        'Stop Loss': p_str(stop_loss),
                        'TP 1 (1:2)': p_str(tp1),
                        'TP 2 (1:3.5)': p_str(tp2)
                    }
                    scan_results.append(record)
                    
                    if alert_key not in ALERTS_SENT:
                        send_telegram_alert(
                            title="🎯 PERFECT STOCH CROSS + OB SETUP",
                            message_body=(
                                f"Pair: `{symbol}`\n"
                                f"Timeframe: `{tf}`\n"
                                f"Direction: *{direction}*\n"
                                f"4H Macro Sync: *{macro_desc} (PASSED)*\n"
                                f"LTF Setup: *{ltf_desc}*\n"
                                f"-----------------------------------\n"
                                f"🧱 *OB Type:* `{ob_type}`\n"
                                f"📍 *OB Start (Low):* `{p_str(ob_low)}`\n"
                                f"📍 *OB End (High):* `{p_str(ob_high)}`\n"
                                f"📏 *OB Size:* `{p_str(ob_size)}`\n"
                                f"-----------------------------------\n"
                                f"📥 *Entry Price:* `{p_str(price)}`\n"
                                f"🛑 *Stop Loss:* `{p_str(stop_loss)}`\n"
                                f"🎯 *Take Profit 1:* `{p_str(tp1)}`\n"
                                f"🎯 *Take Profit 2:* `{p_str(tp2)}`"
                            ),
                            alert_type="TRADE"
                        )
                        ALERTS_SENT.add(alert_key)

            except Exception:
                continue
            time.sleep(0.01)

    print("\n" + "="*115)
    print(f"       SCAN COMPLETE — {time.strftime('%Y-%m-%d %H:%M:%S')}       ")
    print("="*115)
    if scan_results:
        print(tabulate(scan_results, headers="keys", tablefmt="grid", showindex=False))
    else:
        print("No matching synchronized assets found in this cycle. Re-scanning in 2 mins...")
    print("="*115)

if __name__ == "__main__":
    while True:
        try:
            scan_markets()
        except Exception as err:
            print(f"[RECOVERY ERROR] {err}. Restarting loop in 10s...")
            time.sleep(10)
            
        print("[INFO] Waiting 2 minutes before next automated background sweep...")
        time.sleep(120)
