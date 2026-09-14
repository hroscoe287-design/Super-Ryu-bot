Build a complete, mobile-friendly Telegram Trading Signal Bot + Web Dashboard called RYU SIGNAL BOT.

The project must be a working Python Flask application that can be deployed directly to Render.

Core dashboard

Create a dark, professional trading dashboard with:

- RYU SIGNAL BOT branding
- Mobile-responsive layout
- Current asset/pair
- Current price
- CALL / PUT / WAIT signal
- Signal confidence percentage
- Signal timestamp
- Real-world entry countdown
- Expiry/timeframe
- Indicator status
- Live/Disconnected feed status

Trading indicators

The signal engine must use ALL of these together before generating a CALL or PUT:

1. Moving Average
   
   - EMA 9
   - EMA 20
   - EMA 50

2. Bollinger Bands
   
   - Period: 20
   - Standard deviation: 2

3. SuperTrend
   
   - ATR-based
   - Make multiplier and ATR period configurable

A signal should only be generated when multiple indicators confirm the same direction. Otherwise return WAIT.

Signal logic

Create a scoring system using:

- EMA trend alignment
- Bollinger Band position
- SuperTrend direction
- Recent candle momentum

Display the resulting confidence from 0–100%.

Default minimum confidence:
75%.

Do NOT claim that the strategy guarantees winning trades.

Dashboard controls

Include:

- Asset selector
- Timeframe selector
- Minimum confidence setting
- EMA settings
- Bollinger Bands settings
- SuperTrend settings
- Start/Stop signal engine
- Demo mode
- Signal history
- Win/loss statistics entered by the user

Telegram

Prepare Telegram integration using environment variables:

- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID

Send a Telegram message whenever a qualifying CALL or PUT signal is generated.

Do not hard-code credentials.

Example alert format:

RYU SIGNAL BOT
Asset: EUR/USD
Signal: CALL
Confidence: 82%
Entry: 10:35:20
Expiry: 5 minutes
SuperTrend: UP
EMA: BULLISH
Bollinger: CONFIRMED

Data architecture

Create a clean data-provider interface so the application can receive market candles from a supported data source.

Do NOT pretend to have an official Pocket Option API.

The application must clearly display:
LIVE, DELAYED, or DISCONNECTED depending on feed status.

Include a demo/mock data mode so the dashboard can run even when no market-data API credentials are configured.

Flask API endpoints

Create:
GET /
GET /api/status
GET /api/signal
GET /api/settings
POST /api/settings
GET /api/history
POST /api/start
POST /api/stop

Return JSON from API endpoints.

Render deployment

The application must listen on Render's PORT environment variable.

Use this start command:

gunicorn main:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120

Create a requirements.txt containing all required Python packages.

Important

Keep the project simple enough to deploy from GitHub to Render.

Do not create unnecessary folders or complicated dependencies.

The final project must include:

- main.py
- requirements.txt
- README.md

Make sure the Python code is syntactically complete and runnable.

Do not use triple-quoted HTML strings in a way that can cause unterminated-string syntax errors.

Return the COMPLETE contents of every file, clearly separated by filename.
