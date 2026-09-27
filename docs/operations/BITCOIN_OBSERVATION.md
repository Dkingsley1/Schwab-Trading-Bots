# Bitcoin Observation

The existing native 15-minute accrual job runs `bitcoin-price-watch --json`.
It uses no Coinbase credentials or account endpoints. The three observers are:

- `btc_intraday_breakout_watch`: last closed 5-minute close versus the prior 12-bar high.
- `btc_intraday_pullback_watch`: last close versus the prior 20-bar distribution; a lower-range observation is not a buy signal.
- `btc_swing_trend_watch`: closed 6-hour price and 20/50-bar exponential averages.

All calculations use the existing validated public candle transport. Two
requests supply all three profiles; missing, duplicate, gapped, nonfinite or
incomplete windows produce unavailable observations. Source bar-close times
remain separate from report generation time. This is periodic research
observation, not tick-level monitoring or executable price discovery.

The owner holds a kernel singleton lock, respects system OFF, and has a
30-second native outer deadline. It retains one latest report, not a second
raw candle database. The local crypto workspace exposes its freshness and
observations through the existing snapshot response.

No bot registry admission, training, paper fills, live orders, withdrawals or
capital allocation occur. The user's approximate $40 balance is not verified
cash. Live consideration requires actual balances/holdings, account fee tier,
spread/slippage, product minimums, net-of-cost forward evidence, explicit loss
limits and separate execution authorization. Price motion alone does not
establish a profitable strategy. Coinbase fees depend on the account tier and
maker/taker execution, not merely the selected order label.
