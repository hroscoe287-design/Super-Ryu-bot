import os
import time
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)

# ============================================================
# CONFIG
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

DEFAULT_ASSET = os.getenv("DEFAULT_ASSET", "EURUSD_otc")
DEFAULT_TIMEFRAME = int(os.getenv("DEFAULT_TIMEFRAME", "60"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "75"))

DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"

MAX_CANDLES = 300
SIGNAL_COOLDOWN = 30

# ============================================================
# STATE
# ============================================================

state = {
    "running": True,
    "feed_status": "DEMO" if DEMO_MODE else "DISCONNECTED",
    "asset": DEFAULT_ASSET,
    "timeframe": DEFAULT_TIMEFRAME,
    "last_price": None,
    "last_signal": "WAIT",
    "confidence": 0,
    "signal_time": None,
    "entry_deadline": None,
    "expiry": None,
    "last_feed": None,
    "candles": [],
    "signals": [],
    "wins": 0,
    "losses": 0,
    "last_alert_key": None,
}

lock = threading.Lock()


# ============================================================
# HELPERS
# ============================================================

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def clean_asset(asset):
    return str(asset or DEFAULT_ASSET).strip()[:40]


def clean_timeframe(value):
    try:
        value = int(value)
        return max(1, min(value, 86400))
    except Exception:
        return DEFAULT_TIMEFRAME


def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "Telegram credentials not configured"

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    try:
        response = requests.post(
            url,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
            },
            timeout=10,
        )

        if response.ok:
            return True, "sent"

        return False, response.text[:500]

    except Exception as exc:
        return False, str(exc)


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):
    if len(df) < 55:
        return df

    df = df.copy()

    # EMA
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # Bollinger Bands: 20 / 2
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()

    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + (2 * bb_std)
    df["bb_lower"] = bb_mid - (2 * bb_std)

    # ATR for SuperTrend
    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - previous_close).abs()
    tr3 = (df["low"] - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = true_range.rolling(10).mean()

    # SuperTrend
    multiplier = 3.0

    hl2 = (df["high"] + df["low"]) / 2

    upper_band = hl2 + multiplier * df["atr"]
    lower_band = hl2 - multiplier * df["atr"]

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    direction = pd.Series(
        index=df.index,
        dtype="int64"
    )

    direction.iloc[0] = 1

    for i in range(1, len(df)):
        if (
            upper_band.iloc[i] < final_upper.iloc[i - 1]
            or df["close"].iloc[i - 1] > final_upper.iloc[i - 1]
        ):
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if (
            lower_band.iloc[i] > final_lower.iloc[i - 1]
            or df["close"].iloc[i - 1] < final_lower.iloc[i - 1]
        ):
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if direction.iloc[i - 1] == -1:
            if df["close"].iloc[i] > final_upper.iloc[i]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1
        else:
            if df["close"].iloc[i] < final_lower.iloc[i]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = 1

    df["supertrend"] = direction

    return df


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(df):
    if len(df) < 55:
        return {
            "signal": "WAIT",
            "confidence": 0,
            "reason": "Not enough candles",
        }

    df = calculate_indicators(df)

    row = df.iloc[-1]

    required = [
        "ema9",
        "ema20",
        "ema50",
        "bb_upper",
        "bb_lower",
        "bb_mid",
        "supertrend",
    ]

    if any(pd.isna(row[x]) for x in required):
        return {
            "signal": "WAIT",
            "confidence": 0,
            "reason": "Indicators not ready",
        }

    price = float(row["close"])

    call_score = 0
    put_score = 0

    reasons_call = []
    reasons_put = []

    # ---------------- EMA TREND ----------------

    if row["ema9"] > row["ema20"]:
        call_score += 20
        reasons_call.append("EMA9 > EMA20")
    else:
        put_score += 20
        reasons_put.append("EMA9 < EMA20")

    if row["ema20"] > row["ema50"]:
        call_score += 20
        reasons_call.append("EMA20 > EMA50")
    else:
        put_score += 20
        reasons_put.append("EMA20 < EMA50")

    # ---------------- BOLLINGER ----------------

    if price > row["bb_mid"]:
        call_score += 15
        reasons_call.append("Price above BB mid")
    else:
        put_score += 15
        reasons_put.append("Price below BB mid")

    # Avoid treating a band touch alone as a guaranteed reversal.
    if price >= row["bb_upper"]:
        put_score += 10
        reasons_put.append("Upper BB pressure")

    if price <= row["bb_lower"]:
        call_score += 10
        reasons_call.append("Lower BB pressure")

    # ---------------- SUPERTREND ----------------

    if row["supertrend"] == 1:
        call_score += 25
        reasons_call.append("SuperTrend UP")
    else:
        put_score += 25
        reasons_put.append("SuperTrend DOWN")

    # ---------------- MOMENTUM ----------------

    if len(df) >= 3:
        c1 = float(df["close"].iloc[-1])
        c2 = float(df["close"].iloc[-2])
        c3 = float(df["close"].iloc[-3])

        if c1 > c2 > c3:
            call_score += 10
            reasons_call.append("Bullish momentum")

        elif c1 < c2 < c3:
            put_score += 10
            reasons_put.append("Bearish momentum")

    best_score = max(call_score, put_score)

    # Convert score into confidence.
    confidence = min(99, int(best_score))

    if call_score > put_score:
        direction = "CALL"
        reasons = reasons_call
        opposite = put_score
    elif put_score > call_score:
        direction = "PUT"
        reasons = reasons_put
        opposite = call_score
    else:
        direction = "WAIT"
        reasons = []
        opposite = best_score

    # Require meaningful confirmation.
    if confidence < MIN_CONFIDENCE:
        direction = "WAIT"

    if abs(best_score - opposite) < 10:
        direction = "WAIT"

    return {
        "signal": direction,
        "confidence": confidence,
        "price": price,
        "reason": ", ".join(reasons),
        "ema9": float(row["ema9"]),
        "ema20": float(row["ema20"]),
        "ema50": float(row["ema50"]),
        "bb_upper": float(row["bb_upper"]),
        "bb_mid": float(row["bb_mid"]),
        "bb_lower": float(row["bb_lower"]),
        "supertrend": (
            "UP" if row["supertrend"] == 1 else "DOWN"
        ),
    }


# ============================================================
# DATA HANDLING
# ============================================================

def normalize_candle(item):
    if not isinstance(item, dict):
        raise ValueError("Candle must be an object")

    close = float(
        item.get("close", item.get("price"))
    )

    open_price = float(
        item.get("open", close)
    )

    high = float(
        item.get("high", max(open_price, close))
    )

    low = float(
        item.get("low", min(open_price, close))
    )

    timestamp = item.get(
        "timestamp",
        time.time()
    )

    return {
        "timestamp": timestamp,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
    }


def add_candles(asset, candles):
    normalized = []

    for candle in candles:
        normalized.append(
            normalize_candle(candle)
        )

    with lock:
        state["asset"] = clean_asset(asset)

        state["candles"].extend(normalized)

        state["candles"] = (
            state["candles"][-MAX_CANDLES:]
        )

        if normalized:
            state["last_price"] = normalized[-1]["close"]
            state["last_feed"] = now_iso()
            state["feed_status"] = "LIVE"

    process_signal()


# ============================================================
# SIGNAL PROCESSING
# ============================================================

def process_signal():
    with lock:
        if not state["running"]:
            return

        candles = list(state["candles"])
        asset = state["asset"]
        timeframe = state["timeframe"]

    if len(candles) < 55:
        return

    df = pd.DataFrame(candles)

    result = generate_signal(df)

    signal = result["signal"]
    confidence = result["confidence"]

    with lock:
        state["last_signal"] = signal
        state["confidence"] = confidence

    if signal not in ("CALL", "PUT"):
        return

    if confidence < MIN_CONFIDENCE:
        return

    signal_time = now_iso()

    # Prevent repeated alerts for the exact same candle/direction.
    last_candle = candles[-1].get("timestamp")
    alert_key = f"{asset}:{last_candle}:{signal}"

    with lock:
        if state["last_alert_key"] == alert_key:
            return

        state["last_alert_key"] = alert_key

    entry_deadline = time.time() + 15
    expiry = time.time() + timeframe

    with lock:
        state["signal_time"] = signal_time
        state["entry_deadline"] = entry_deadline
        state["expiry"] = expiry

        signal_record = {
            "asset": asset,
            "signal": signal,
            "confidence": confidence,
            "price": result.get("price"),
            "time": signal_time,
            "entry_deadline": entry_deadline,
            "expiry": expiry,
            "supertrend": result.get("supertrend"),
        }

        state["signals"].insert(
            0,
            signal_record
        )

        state["signals"] = state["signals"][:100]

    message = (
        "🔥 RYU SIGNAL BOT\n\n"
        f"Asset: {asset}\n"
        f"Signal: {signal}\n"
        f"Confidence: {confidence}%\n"
        f"Entry Price: {result.get('price')}\n"
        f"Timeframe: {timeframe}s\n\n"
        f"EMA: "
        f"{result.get('ema9', 0):.5f} / "
        f"{result.get('ema20', 0):.5f} / "
        f"{result.get('ema50', 0):.5f}\n"
        f"Bollinger: 20 / 2\n"
        f"SuperTrend: {result.get('supertrend')}\n\n"
        "⚠️ Signal only — not a guarantee."
    )

    send_telegram(message)


# ============================================================
# DEMO FEED
# ============================================================

def demo_loop():
    price = 1.10000

    while True:
        time.sleep(2)

        with lock:
            if not DEMO_MODE:
                continue

            running = state["running"]
            asset = state["asset"]

        if not running:
            continue

        # Synthetic demonstration price.
        movement = np.random.normal(0, 0.00025)
        price += movement

        candle = {
            "timestamp": time.time(),
            "open": price - movement,
            "high": price + abs(movement) * 0.5,
            "low": price - abs(movement) * 0.5,
            "close": price,
        }

        add_candles(asset, [candle])


threading.Thread(
    target=demo_loop,
    daemon=True
).start()


# ============================================================
# WEB DASHBOARD
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport"
      content="width=device-width, initial-scale=1">

<title>RYU Signal Bot</title>

<style>
body {
    margin: 0;
    background: #07110b;
    color: #eaffee;
    font-family: Arial, sans-serif;
}

header {
    padding: 18px;
    background: #0d1d12;
    border-bottom: 1px solid #23452d;
}

h1 {
    margin: 0;
    font-size: 24px;
}

.container {
    padding: 15px;
    max-width: 900px;
    margin: auto;
}

.card {
    background: #0c1a11;
    border: 1px solid #254b30;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 15px;
}

.signal {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    padding: 20px;
}

.call {
    color: #42ff78;
}

.put {
    color: #ff5757;
}

.wait {
    color: #e6e6e6;
}

.grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(140px, 1fr));
    gap: 10px;
}

.stat {
    background: #122318;
    padding: 14px;
    border-radius: 10px;
}

.label {
    color: #91a99a;
    font-size: 12px;
}

.value {
    font-size: 20px;
    margin-top: 5px;
}

button, select, input {
    width: 100%;
    padding: 12px;
    margin-top: 8px;
    border-radius: 8px;
    border: 1px solid #31583b;
    background: #102017;
    color: white;
    box-sizing: border-box;
}

button {
    cursor: pointer;
    font-weight: bold;
}

.history {
    max-height: 350px;
    overflow-y: auto;
}

.row {
    padding: 10px 0;
    border-bottom: 1px solid #203b28;
}
</style>
</head>

<body>

<header>
    <h1>🔥 RYU SIGNAL BOT</h1>
    <div>EMA + Bollinger 20/2 + SuperTrend</div>
</header>

<div class="container">

<div class="card">
    <div id="signal" class="signal wait">
        WAIT
    </div>

    <div class="grid">
        <div class="stat">
            <div class="label">ASSET</div>
            <div id="asset" class="value">--</div>
        </div>

        <div class="stat">
            <div class="label">PRICE</div>
            <div id="price" class="value">--</div>
        </div>

        <div class="stat">
            <div class="label">CONFIDENCE</div>
            <div id="confidence" class="value">0%</div>
        </div>

        <div class="stat">
            <div class="label">FEED</div>
            <div id="feed" class="value">--</div>
        </div>
    </div>
</div>

<div class="card">
    <h3>Entry Timer</h3>
    <div id="timer" class="value">--</div>
</div>

<div class="card">
    <h3>Indicators</h3>

    <div class="grid">
        <div class="stat">
            <div class="label">EMA 9</div>
            <div id="ema9" class="value">--</div>
        </div>

        <div class="stat">
            <div class="label">EMA 20</div>
            <div id="ema20" class="value">--</div>
        </div>

        <div class="stat">
            <div class="label">EMA 50</div>
            <div id="ema50" class="value">--</div>
        </div>

        <div class="stat">
            <div class="label">SUPERTREND</div>
            <div id="supertrend" class="value">--</div>
        </div>
    </div>

    <p>
        Bollinger Bands:
        <b>Period 20 / Deviation 2</b>
    </p>
</div>

<div class="card">
    <h3>Controls</h3>

    <label>Asset</label>
    <input id="assetInput"
           value="EURUSD_otc">

    <label>Timeframe (seconds)</label>
    <input id="timeframeInput"
           type="number"
           value="60">

    <label>Minimum Confidence</label>
    <input id="confidenceInput"
           type="number"
           value="75">

    <button onclick="saveSettings()">
        SAVE SETTINGS
    </button>

    <button onclick="toggleBot()">
        START / STOP
    </button>
</div>

<div class="card">
    <h3>Signal History</h3>
    <div id="history" class="history"></div>
</div>

</div>

<script>

async function update() {

    try {

        const response =
            await fetch('/api/signal');

        const data =
            await response.json();

        const signal =
            document.getElementById('signal');

        signal.innerText =
            data.signal || 'WAIT';

        signal.className =
            'signal ' +
            (data.signal === 'CALL'
                ? 'call'
                : data.signal === 'PUT'
                ? 'put'
                : 'wait');

        document.getElementById('asset')
            .innerText = data.asset || '--';

        document.getElementById('price')
            .innerText =
            data.price == null
                ? '--'
                : Number(data.price).toFixed(6);

        document.getElementById('confidence')
            .innerText =
            (data.confidence || 0) + '%';

        document.getElementById('feed')
            .innerText =
            data.feed_status || '--';

        document.getElementById('ema9')
            .innerText =
            data.ema9 == null
                ? '--'
                : Number(data.ema9).toFixed(6);

        document.getElementById('ema20')
            .innerText =
            data.ema20 == null
                ? '--'
                : Number(data.ema20).toFixed(6);

        document.getElementById('ema50')
            .innerText =
            data.ema50 == null
                ? '--'
                : Number(data.ema50).toFixed(6);

        document.getElementById('supertrend')
            .innerText =
            data.supertrend || '--';

        if (data.entry_deadline) {

            const seconds =
                Math.max(
                    0,
                    Math.ceil(
                        data.entry_deadline -
                        Date.now() / 1000
                    )
                );

            document.getElementById('timer')
                .innerText =
                seconds + ' seconds';

        } else {
            document.getElementById('timer')
                .innerText = '--';
        }

        renderHistory(data.history || []);

    } catch (e) {
        document.getElementById('feed')
            .innerText = 'DISCONNECTED';
    }
}


function renderHistory(history) {

    const box =
        document.getElementById('history');

    box.innerHTML = '';

    history.forEach(item => {

        const row =
            document.createElement('div');

        row.className = 'row';

        row.innerText =
            item.asset +
            ' — ' +
            item.signal +
            ' — ' +
            item.confidence +
            '% — ' +
            item.price;

        box.appendChild(row);
    });
}


async function saveSettings() {

    await fetch('/api/settings', {
        method: 'POST',
        headers: {
            'Content-Type':
                'application/json'
        },
        body: JSON.stringify({
            asset:
                document.getElementById(
                    'assetInput'
                ).value,

            timeframe:
                document.getElementById(
                    'timeframeInput'
                ).value,

            min_confidence:
                document.getElementById(
                    'confidenceInput'
                ).value
        })
    });

    update();
}


async function toggleBot() {

    await fetch('/api/start', {
        method: 'POST'
    });

    update();
}


setInterval(update, 2000);

update();

</script>

</body>
</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/api/status")
def api_status():
    with lock:
        return jsonify({
            "running": state["running"],
            "feed_status": state["feed_status"],
            "asset": state["asset"],
            "timeframe": state["timeframe"],
            "price": state["last_price"],
            "last_feed": state["last_feed"],
            "candles": len(state["candles"]),
            "signals": len(state["signals"]),
        })


@app.route("/api/signal")
def api_signal():
    with lock:
        candles = list(state["candles"])

        latest = {}

        if len(candles) >= 55:
            df = pd.DataFrame(candles)
            latest = generate_signal(df)

        return jsonify({
            "running": state["running"],
            "feed_status": state["feed_status"],
            "asset": state["asset"],
            "timeframe": state["timeframe"],
            "price": state["last_price"],
            "signal": state["last_signal"],
            "confidence": state["confidence"],
            "signal_time": state["signal_time"],
            "entry_deadline": state["entry_deadline"],
            "expiry": state["expiry"],
            "ema9": latest.get("ema9"),
            "ema20": latest.get("ema20"),
            "ema50": latest.get("ema50"),
            "supertrend": latest.get("supertrend"),
            "history": state["signals"][:25],
        })


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():

    if request.method == "POST":

        data = request.get_json(
            silent=True
        ) or {}

        with lock:

            if "asset" in data:
                state["asset"] = clean_asset(
                    data["asset"]
                )

            if "timeframe" in data:
                state["timeframe"] = clean_timeframe(
                    data["timeframe"]
                )

            if "min_confidence" in data:
                global MIN_CONFIDENCE

                try:
                    MIN_CONFIDENCE = max(
                        1,
                        min(
                            99,
                            float(
                                data["min_confidence"]
                            )
                        )
                    )
                except Exception:
                    pass

    with lock:
        return jsonify({
            "asset": state["asset"],
            "timeframe": state["timeframe"],
            "min_confidence": MIN_CONFIDENCE,
            "demo_mode": DEMO_MODE,
        })


@app.route("/api/start", methods=["POST"])
def api_start():

    with lock:
        state["running"] = True

    return jsonify({
        "ok": True,
        "running": True,
    })


@app.route("/api/stop", methods=["POST"])
def api_stop():

    with lock:
        state["running"] = False
        state["last_signal"] = "WAIT"

    return jsonify({
        "ok": True,
        "running": False,
    })


@app.route("/api/history")
def api_history():

    with lock:
        return jsonify({
            "signals": state["signals"],
            "wins": state["wins"],
            "losses": state["losses"],
        })


# ============================================================
# LIVE FEED BRIDGE
# ============================================================

@app.route("/api/feed", methods=["POST"])
def api_feed():

    data = request.get_json(
        silent=True
    )

    if not data:
        return jsonify({
            "ok": False,
            "error": "JSON body required"
        }), 400

    # Optional shared secret.
    expected_token = os.getenv(
        "RYU_FEED_TOKEN",
        ""
    ).strip()

    if expected_token:

        supplied_token = (
            request.headers.get(
                "X-RYU-FEED-TOKEN",
                ""
            ).strip()
        )

        if supplied_token != expected_token:
            return jsonify({
                "ok": False,
                "error": "Invalid feed token"
            }), 401

    asset = clean_asset(
        data.get(
            "asset",
            DEFAULT_ASSET
        )
    )

    candles = data.get("candles")

    # Accept either:
    #
    # {
    #   "asset": "EURUSD_otc",
    #   "candles": [...]
    # }
    #
    # or a single candle:
    #
    # {
    #   "asset": "EURUSD_otc",
    #   "close": 1.12345
    # }

    if candles is None:
        candles = [data]

    if not isinstance(candles, list):
        return jsonify({
            "ok": False,
            "error": "candles must be a list"
        }), 400

    try:
        add_candles(
            asset,
            candles
        )

        with lock:
            count = len(state["candles"])
            price = state["last_price"]

        return jsonify({
            "ok": True,
            "asset": asset,
            "candles": count,
            "price": price,
            "feed_status": "LIVE",
        })

    except Exception as exc:

        return jsonify({
            "ok": False,
            "error": str(exc)
        }), 400


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "RYU Signal Bot",
        "time": now_iso(),
    })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv("PORT", "5000")
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
