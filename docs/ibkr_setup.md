# Interactive Brokers TWS Setup for QuantCore

## Overview
QuantCore connects to IBKR through Trader Workstation (TWS) or IB Gateway.
TWS must be running and logged in before QuantCore starts.

## Step 1: Download TWS
- Go to: interactivebrokers.com/en/trading/tws.php
- Download "Trader Workstation" (the full version, not the mobile app)
- Install and launch it

## Step 2: Log In to Paper Trading Account
- At the TWS login screen, enter your IBKR username and password
- IMPORTANT: Change "Trading Mode" to "Paper Trading" before clicking login
- Your paper account starts with "DU" (e.g., DU1234567)
- Paper accounts come with $1,000,000 in simulated cash

## Step 3: Enable API Access in TWS
This is the most common setup mistake — do this carefully:
- In TWS, go to: Edit > Global Configuration (or File > Global Configuration on Mac)
- In the left panel, navigate to: API > Settings
- Check: "Enable ActiveX and Socket Clients"
- Set Socket port to: 7497
- Check: "Allow connections from localhost only" (security)
- UNCHECK: "Read-Only API" (QuantCore needs to submit orders)
- Check: "Download open orders on connection"
- Click Apply, then OK
- TWS will ask to restart — click OK and log back in

## Step 4: Verify API is Working
After TWS restarts and you log back in:
- Look for a green dot in the bottom right of TWS — this means API is enabled
- Run the QuantCore verification:
  python scripts/verify_ibkr.py

## Step 5: Configure QuantCore
In your .env file:
  IBKR_HOST=127.0.0.1
  IBKR_PORT=7497
  IBKR_CLIENT_ID=1

## Common Issues

### "Cannot connect to TWS/Gateway"
- TWS is not running — start it and log in
- API is not enabled — follow Step 3 again
- Wrong port — verify Socket port in TWS is 7497

### "clientId already in use"
- Another program is connected with clientId=1
- Change IBKR_CLIENT_ID=2 in .env

### "Not connected" errors during trading
- TWS disconnected from IBKR servers (happens occasionally)
- Check TWS — it may need you to accept a notification or re-login
- QuantCore will attempt to reconnect automatically

### Paper account not showing correct balance
- Paper accounts reset at the start of each trading day
- The $1,000,000 starting balance is restored daily
- P&L from previous day is NOT carried over in paper trading (this is an IBKR limitation)
- To track multi-day P&L, rely on QuantCore's logs, not the TWS account balance

### Market data shows delayed or no data
- IBKR charges for real-time market data subscriptions
- Use --data-source alpaca instead (free real-time data via Alpaca IEX feed)
- The hybrid mode (Alpaca data + IBKR execution) is the recommended configuration

## Running QuantCore with IBKR

### Recommended: Hybrid mode (Alpaca data + IBKR execution)
Requires: TWS running + Alpaca API keys in .env
```bash
python scripts/paper_trade.py \
  --duration 5d \
  --strategy momentum \
  --data-source alpaca \
  --broker ibkr \
  --risk-config config/risk.yaml
```

### IBKR only (data + execution from IBKR)
Requires: TWS running + IBKR market data subscription
```bash
python scripts/paper_trade.py \
  --duration 5d \
  --strategy momentum \
  --data-source ibkr \
  --broker ibkr \
  --risk-config config/risk.yaml
```

## IB Gateway (Advanced — For Unattended Operation)
IB Gateway is a lightweight alternative to TWS for automated trading:
- Download from: interactivebrokers.com/en/trading/ibgateway.php
- Uses less memory than TWS
- No GUI — connects headlessly
- Paper port: 4002 (set IBKR_PORT=4002 in .env)
- Live port: 4001
- Recommended for production deployment on a server
