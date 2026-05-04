import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_python import resolve_runtime_python
from core.exotic_derivatives_plumbing import is_exotic_derivative_sleeve


VENV_PY = resolve_runtime_python(PROJECT_ROOT)
SHADOW_LOOP = PROJECT_ROOT / "scripts" / "run_shadow_training_loop.py"
LOAD_RUNTIME_ENV = PROJECT_ROOT / "scripts" / "ops" / "load_runtime_env.sh"

SLEEVE_DEFAULTS = {
    "volatility": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,SVXY,TQQQ,SQQQ,SPXL,SPXS,SOXL,SOXS",
        "context_symbols": "$VIX.X,SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,SVXY",
        "interval": "90",
        "threshold_shift": "+0.05",
    },
    "pairs_correlation": {
        "symbols": "SPY,QQQ,IWM,DIA,XLK,XLF,XLE,XLV,XLI,XLY,XLP,XLU,TLT,GLD,UUP,IBIT,ETHA,AAPL,MSFT,NVDA,AMD,AVGO,TSM",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,IBIT,ETHA,XLK,XLF,XLE,XLV",
        "interval": "120",
        "threshold_shift": "+0.04",
    },
    "stat_arb_market_neutral": {
        "symbols": "AAPL,MSFT,NVDA,AMD,AVGO,TSM,ASML,MU,AMZN,GOOG,GOOGL,META,NFLX,ORCL,CRM,ADBE,JPM,BAC,GS,MS,V,MA",
        "context_symbols": "SPY,QQQ,IWM,XLK,XLF,SMH,SOXX,UUP,TLT,VIXY",
        "interval": "120",
        "threshold_shift": "+0.06",
    },
    "earnings_event": {
        "symbols": "AAPL,MSFT,NVDA,AMD,AVGO,TSM,ASML,MU,AMZN,GOOG,GOOGL,META,NFLX,DIS,ORCL,CRM,ADBE,NOW,PLTR,TSLA,COST,WMT,HD,JPM,BAC,GS,LLY,UNH",
        "context_symbols": "SPY,QQQ,IWM,XLK,XLY,XLF,XLV,SMH,VIXY,TLT,UUP",
        "interval": "90",
        "threshold_shift": "+0.05",
    },
    "commodity_inflation": {
        "symbols": "DBC,USO,UNG,CORN,WEAT,SLV,GLD,CPER,URA,XLE,XOP,OIH,XOM,CVX,COP,EOG,SLB,MPC,VLO,UUP,FXC,FXA,EEM,EWZ",
        "context_symbols": "DBC,USO,UNG,GLD,SLV,UUP,FXC,FXA,TLT,TIP,XLE,XOP,EEM,EWZ",
        "interval": "120",
        "threshold_shift": "+0.05",
    },
    "international_macro": {
        "symbols": "EFA,EEM,IEFA,VEA,VWO,VGK,EWJ,FXI,EWZ,INDA,EWU,EWG,EWQ,EWC,EWA,EWW,EWY,EWT,IXUS,UUP,FXE,FXY,FXB,FXC,FXA",
        "context_symbols": "SPY,QQQ,TLT,GLD,UUP,FXE,FXY,FXB,FXC,FXA,EFA,EEM,EWJ,FXI,EWZ,INDA",
        "interval": "150",
        "threshold_shift": "+0.05",
    },
    "market_making_liquidity": {
        "symbols": "SPY,QQQ,IWM,DIA,XLK,XLF,XLE,XLV,SMH,SOXX,AAPL,MSFT,NVDA,AMD,TSLA,AMZN,META,GOOGL,JPM,XOM,TLT,GLD,UUP,UVXY,VIXY",
        "context_symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,SMH",
        "interval": "60",
        "threshold_shift": "+0.07",
    },
    "short_bias_hedge": {
        "symbols": "SH,PSQ,SDS,QID,SPXU,SQQQ,SOXS,UVXY,VIXY,TLT,GLD,UUP,XLU,XLP,XLV,SPY,QQQ,IWM,SMH,XLF,XLE,HYG,JNK",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,HYG,JNK,VIXY,UVXY,XLU,XLP,XLV",
        "interval": "90",
        "threshold_shift": "+0.06",
    },
    "single_name_options_event": {
        "symbols": "NVDA,AMD,TSLA,AAPL,MSFT,AMZN,META,GOOGL,AVGO,SMCI,MSTR,COIN,PLTR,NFLX,LLY,UNH,JPM,XOM,BA,CAT,ORCL,CRM,ADBE,COST,WMT",
        "context_symbols": "SPY,QQQ,IWM,XLK,XLY,XLF,XLV,SMH,VIXY,UVXY,TLT,UUP",
        "interval": "75",
        "threshold_shift": "+0.06",
    },
    "rates_credit_macro": {
        "symbols": "TLT,IEF,SHY,TLH,TIP,LQD,HYG,JNK,USHY,AGG,BND,MUB,IGIB,FLOT,BIL,SGOV,KRE,XLF,JPM,BAC,GS,MS,UUP,GLD",
        "context_symbols": "TLT,IEF,SHY,TIP,LQD,HYG,JNK,AGG,BND,UUP,GLD,KRE,XLF,SPY,QQQ,VIXY",
        "interval": "120",
        "threshold_shift": "+0.06",
    },
    "cash_rotation_tactical": {
        "symbols": "BIL,SGOV,SHY,IEF,TLT,AGG,BND,FLOT,USFR,TFLO,MINT,NEAR,SPY,QQQ,IWM,GLD,UUP,XLU,XLP,XLV,SPLV,VTV",
        "context_symbols": "BIL,SGOV,SHY,IEF,TLT,AGG,BND,FLOT,SPY,QQQ,IWM,GLD,UUP,VIXY,HYG,LQD",
        "interval": "150",
        "threshold_shift": "+0.08",
    },
    "futures_index_intraday": {
        "symbols": "/ES,/NQ,/YM,/RTY,SPY,QQQ,DIA,IWM,TQQQ,SQQQ,UVXY,VIXY",
        "context_symbols": "/ES,/NQ,/YM,/RTY,SPY,QQQ,IWM,TLT,UUP,VIXY,XLK,XLF,SMH",
        "interval": "45",
        "threshold_shift": "+0.07",
    },
    "futures_rates_curve": {
        "symbols": "/ZT,/ZF,/ZN,/ZB,TLT,IEF,SHY,TLH,TIP,LQD,HYG,JNK,UUP,GLD",
        "context_symbols": "/ZT,/ZF,/ZN,/ZB,TLT,IEF,SHY,TIP,LQD,HYG,UUP,GLD,SPY,QQQ,VIXY",
        "interval": "75",
        "threshold_shift": "+0.07",
    },
    "futures_commodity_macro": {
        "symbols": "/CL,/NG,/GC,/SI,/HG,USO,UNG,GLD,SLV,CPER,DBC,XLE,XOP,OIH,UUP,TLT,TIP",
        "context_symbols": "/CL,/NG,/GC,/SI,/HG,USO,UNG,GLD,SLV,DBC,XLE,XOP,UUP,TLT,TIP,SPY,VIXY",
        "interval": "90",
        "threshold_shift": "+0.06",
    },
    "crypto_futures_basis": {
        "symbols": "BTC-USD,ETH-USD,SOL-USD,IBIT,FBTC,ETHA,ETHE,MSTR,COIN,MARA,RIOT,CLSK",
        "context_symbols": "BTC-USD,ETH-USD,SOL-USD,IBIT,FBTC,ETHA,MSTR,COIN,SPY,QQQ,UUP,VIXY",
        "interval": "60",
        "threshold_shift": "+0.07",
    },
    "futures_event_reaction": {
        "symbols": "/ES,/NQ,/RTY,/ZT,/ZF,/ZN,/ZB,/CL,/GC,SPY,QQQ,IWM,TLT,UUP,GLD,USO,VIXY",
        "context_symbols": "/ES,/NQ,/RTY,/ZN,/CL,/GC,SPY,QQQ,IWM,TLT,UUP,GLD,VIXY,XLK,XLF,XLE",
        "interval": "45",
        "threshold_shift": "+0.08",
    },
    "options_on_futures": {
        "symbols": "/ES,/NQ,/YM,/RTY,/ZT,/ZF,/ZN,/ZB,/CL,/GC,SPY,QQQ,IWM,TLT,GLD,USO,UUP,VIXY",
        "context_symbols": "/ES,/NQ,/RTY,/ZN,/CL,/GC,SPY,QQQ,IWM,TLT,GLD,USO,UUP,VIXY,UVXY",
        "interval": "60",
        "threshold_shift": "+0.07",
    },
    "options_on_futures_aggressive": {
        "symbols": "/ES,/NQ,/RTY,/CL,/GC,SPY,QQQ,IWM,TQQQ,SQQQ,USO,GLD,VIXY,UVXY",
        "context_symbols": "/ES,/NQ,/RTY,/ZN,/CL,/GC,SPY,QQQ,IWM,TLT,GLD,USO,UUP,VIXY,UVXY",
        "interval": "45",
        "threshold_shift": "-0.01",
    },
    "compound_options": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,AAPL,MSFT,NVDA,AMD,AVGO,TSLA,META,AMZN,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,HYG,LQD,/ES,/NQ,/ZN,/CL,/GC",
        "interval": "180",
        "threshold_shift": "+0.10",
        "domain": "exotic_derivatives",
        "family": "compound_options",
        "correlation_peers": "options_on_futures,single_name_options_event,volatility,rates_credit_macro,pairs_correlation",
    },
    "swaptions": {
        "symbols": "/ZT,/ZF,/ZN,/ZB,TLT,IEF,SHY,TLH,TIP,LQD,HYG,JNK,AGG,BND,KRE,XLF,JPM,BAC,GS,MS,UUP,GLD",
        "context_symbols": "/ZT,/ZF,/ZN,/ZB,TLT,IEF,SHY,TIP,LQD,HYG,JNK,AGG,UUP,GLD,SPY,QQQ,VIXY,KRE,XLF",
        "interval": "180",
        "threshold_shift": "+0.10",
        "domain": "exotic_derivatives",
        "family": "swaptions",
        "correlation_peers": "futures_rates_curve,rates_credit_macro,cash_rotation_tactical,volatility,short_bias_hedge",
    },
    "structured_products": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,LQD,HYG,JNK,GLD,UUP,XLK,XLF,XLE,XLV,XLY,AAPL,MSFT,NVDA,TSLA,JPM,BAC,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,XLV,SPY,QQQ",
        "interval": "240",
        "threshold_shift": "+0.12",
        "domain": "exotic_derivatives",
        "family": "structured_products",
        "correlation_peers": "pairs_correlation,rates_credit_macro,single_name_options_event,volatility,cash_rotation_tactical",
    },
    "synthetic_cdo": {
        "symbols": "LQD,HYG,JNK,USHY,EMB,AGG,BND,TLT,IEF,KRE,XLF,JPM,BAC,C,GS,MS,BLK,SCHW,SPY,QQQ,VIXY,UUP",
        "context_symbols": "LQD,HYG,JNK,USHY,EMB,AGG,BND,TLT,IEF,KRE,XLF,JPM,BAC,SPY,QQQ,VIXY,UUP,GLD",
        "interval": "240",
        "threshold_shift": "+0.12",
        "domain": "exotic_derivatives",
        "family": "synthetic_cdo",
        "correlation_peers": "rates_credit_macro,short_bias_hedge,pairs_correlation,volatility,structured_products",
    },
    "rainbow_options": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,USO,UUP,VIXY,IBIT,ETHA,XLK,XLF,XLE,XLV,XLY,AAPL,MSFT,NVDA,TSLA,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,USO,UUP,VIXY,IBIT,ETHA,XLK,XLF,XLE,XLV,HYG,LQD,/ES,/NQ,/CL,/GC",
        "interval": "180",
        "threshold_shift": "+0.10",
        "domain": "exotic_derivatives",
        "family": "rainbow_options",
        "correlation_peers": "pairs_correlation,volatility,single_name_options_event,commodity_inflation,crypto_futures_basis,second_third_order_greeks",
    },
    "cdo_squared": {
        "symbols": "LQD,HYG,JNK,USHY,EMB,AGG,BND,TLT,IEF,KRE,XLF,JPM,BAC,C,GS,MS,SPY,QQQ,VIXY,UUP",
        "context_symbols": "LQD,HYG,JNK,USHY,EMB,AGG,BND,TLT,IEF,KRE,XLF,SPY,QQQ,VIXY,UUP,GLD",
        "interval": "300",
        "threshold_shift": "+0.14",
        "domain": "exotic_derivatives",
        "family": "cdo_squared",
        "correlation_peers": "synthetic_cdo,structured_products,rates_credit_macro,tail_risk_parity,black_swan_hedging",
    },
    "cdo_cubed": {
        "symbols": "LQD,HYG,JNK,USHY,EMB,AGG,BND,TLT,IEF,KRE,XLF,JPM,BAC,C,GS,MS,SPY,QQQ,VIXY,UUP",
        "context_symbols": "LQD,HYG,JNK,USHY,EMB,AGG,BND,TLT,IEF,KRE,XLF,SPY,QQQ,VIXY,UUP,GLD",
        "interval": "300",
        "threshold_shift": "+0.16",
        "domain": "exotic_derivatives",
        "family": "cdo_cubed",
        "correlation_peers": "cdo_squared,synthetic_cdo,tail_risk_parity,black_swan_hedging,rates_credit_macro",
    },
    "variance_volatility_swaps": {
        "symbols": "SPY,QQQ,IWM,DIA,VIXY,UVXY,SVXY,TLT,GLD,UUP,AAPL,MSFT,NVDA,TSLA,AMD,AVGO,META,AMZN,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,SVXY,XLK,XLF,XLE,/ES,/NQ,/CL,/GC",
        "interval": "180",
        "threshold_shift": "+0.12",
        "domain": "exotic_derivatives",
        "family": "variance_volatility_swaps",
        "correlation_peers": "volatility,options_on_futures,single_name_options_event,black_swan_hedging,second_third_order_greeks",
    },
    "barrier_lookback_options": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,USO,UUP,VIXY,AAPL,MSFT,NVDA,TSLA,AMD,AVGO,META,AMZN,JPM,XOM,LLY",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,USO,UUP,VIXY,UVXY,XLK,XLF,XLE,XLV,/ES,/NQ,/CL,/GC",
        "interval": "180",
        "threshold_shift": "+0.12",
        "domain": "exotic_derivatives",
        "family": "barrier_lookback_options",
        "correlation_peers": "options_flow,volatility,variance_volatility_swaps,compound_options,market_making_liquidity",
    },
    "second_third_order_greeks": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,AAPL,MSFT,NVDA,TSLA,AMD,AVGO,META,AMZN,JPM,XOM,LLY",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,XLV,SMH,/ES,/NQ,/CL,/GC",
        "interval": "150",
        "threshold_shift": "+0.12",
        "domain": "exotic_derivatives",
        "family": "second_third_order_greeks",
        "correlation_peers": "single_name_options_event,options_on_futures,variance_volatility_swaps,barrier_lookback_options,rainbow_options",
    },
    "high_frequency_market_making": {
        "symbols": "SPY,QQQ,IWM,DIA,XLK,XLF,XLE,XLV,SMH,AAPL,MSFT,NVDA,AMD,TSLA,AMZN,META,GOOGL,JPM,XOM,TLT,GLD,UUP,VIXY",
        "context_symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,SMH,/ES,/NQ,/RTY",
        "interval": "45",
        "threshold_shift": "+0.10",
        "domain": "exotic_derivatives",
        "family": "high_frequency_market_making",
        "correlation_peers": "market_making_liquidity,execution_quality,intraday_aggressive,futures_index_intraday,infrastructure_risk",
    },
    "tail_risk_parity": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,SHY,TIP,LQD,HYG,JNK,GLD,UUP,VIXY,UVXY,XLU,XLP,XLV,BIL,SGOV",
        "context_symbols": "SPY,QQQ,IWM,TLT,IEF,SHY,TIP,LQD,HYG,JNK,GLD,UUP,VIXY,UVXY,XLU,XLP,XLV",
        "interval": "240",
        "threshold_shift": "+0.14",
        "domain": "exotic_derivatives",
        "family": "tail_risk_parity",
        "correlation_peers": "short_bias_hedge,volatility,rates_credit_macro,cash_rotation_tactical,black_swan_hedging",
    },
    "black_swan_hedging": {
        "symbols": "SPY,QQQ,IWM,DIA,VIXY,UVXY,SVXY,TLT,GLD,UUP,HYG,JNK,LQD,XLU,XLP,XLV,AAPL,MSFT,NVDA,TSLA",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,HYG,JNK,LQD,XLU,XLP,XLV,/ES,/NQ,/ZN,/CL,/GC",
        "interval": "180",
        "threshold_shift": "+0.14",
        "domain": "exotic_derivatives",
        "family": "black_swan_hedging",
        "correlation_peers": "tail_risk_parity,volatility,short_bias_hedge,variance_volatility_swaps,cdo_cubed",
    },
    "sovereign_debt_macro": {
        "symbols": "TLT,IEF,SHY,TLH,TIP,LQD,HYG,JNK,EMB,AGG,BND,BIL,SGOV,UUP,FXE,FXY,FXB,FXC,FXA,EFA,EEM,EWJ,FXI,EWZ,INDA",
        "context_symbols": "/ZT,/ZF,/ZN,/ZB,TLT,IEF,SHY,TIP,LQD,HYG,JNK,EMB,UUP,FXE,FXY,FXB,FXC,FXA,EFA,EEM,SPY,QQQ,VIXY",
        "interval": "240",
        "threshold_shift": "+0.12",
        "domain": "exotic_derivatives",
        "family": "sovereign_debt_macro",
        "correlation_peers": "swaptions,futures_rates_curve,rates_credit_macro,fx_macro,international_macro",
    },
    "gamma_scalping": {
        "symbols": "SPY,QQQ,IWM,DIA,VIXY,UVXY,AAPL,MSFT,NVDA,TSLA,AMD,AVGO,META,AMZN,LLY,JPM,XOM,TLT,GLD,UUP",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,SMH,/ES,/NQ,/RTY",
        "interval": "75",
        "threshold_shift": "+0.10",
        "domain": "exotic_derivatives",
        "family": "gamma_scalping",
        "correlation_peers": "options_flow,second_third_order_greeks,volatility,market_making_liquidity,order_flow_market_microstructure",
    },
    "statistical_arbitrage": {
        "symbols": "AAPL,MSFT,NVDA,AMD,AVGO,TSM,ASML,MU,AMZN,GOOG,GOOGL,META,NFLX,ORCL,CRM,ADBE,JPM,BAC,GS,MS,V,MA,SPY,QQQ,IWM",
        "context_symbols": "SPY,QQQ,IWM,XLK,XLF,SMH,SOXX,UUP,TLT,VIXY,AAPL,MSFT,NVDA,AMD,JPM,BAC",
        "interval": "90",
        "threshold_shift": "+0.08",
        "domain": "exotic_derivatives",
        "family": "statistical_arbitrage",
        "correlation_peers": "stat_arb_market_neutral,pairs_correlation,market_making_liquidity,cross_asset_basis_training,order_flow_market_microstructure",
    },
    "vanna_volga_hedging": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,AAPL,MSFT,NVDA,TSLA,AMD,AVGO,META,AMZN,JPM,XOM,LLY",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,XLV,SMH,/ES,/NQ,/CL,/GC",
        "interval": "150",
        "threshold_shift": "+0.12",
        "domain": "exotic_derivatives",
        "family": "vanna_volga_hedging",
        "correlation_peers": "second_third_order_greeks,variance_volatility_swaps,volatility,rainbow_options,dispersion_trading",
    },
    "order_flow_market_microstructure": {
        "symbols": "SPY,QQQ,IWM,DIA,XLK,XLF,XLE,XLV,SMH,AAPL,MSFT,NVDA,AMD,TSLA,AMZN,META,GOOGL,JPM,XOM,TLT,GLD,UUP,VIXY",
        "context_symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,SMH,/ES,/NQ,/RTY",
        "interval": "45",
        "threshold_shift": "+0.10",
        "domain": "exotic_derivatives",
        "family": "order_flow_market_microstructure",
        "correlation_peers": "high_frequency_market_making,market_making_liquidity,execution_quality,intraday_aggressive,gamma_scalping",
    },
    "dispersion_trading": {
        "symbols": "SPY,QQQ,IWM,DIA,XLK,XLF,XLE,XLV,SMH,AAPL,MSFT,NVDA,AMD,AVGO,TSLA,AMZN,META,GOOGL,JPM,XOM,LLY",
        "context_symbols": "SPY,QQQ,IWM,DIA,XLK,XLF,XLE,XLV,SMH,VIXY,UVXY,TLT,UUP,AAPL,MSFT,NVDA,AMD,TSLA",
        "interval": "150",
        "threshold_shift": "+0.12",
        "domain": "exotic_derivatives",
        "family": "dispersion_trading",
        "correlation_peers": "pairs_correlation,rainbow_options,volatility,second_third_order_greeks,volatility_arbitrage",
    },
    "cross_asset_basis_training": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,SHY,LQD,HYG,JNK,GLD,USO,UUP,FXE,FXY,FXB,FXC,FXA,IBIT,ETHA,MSTR,COIN",
        "context_symbols": "/ES,/NQ,/RTY,/ZN,/CL,/GC,SPY,QQQ,IWM,TLT,GLD,USO,UUP,IBIT,ETHA,HYG,LQD,EFA,EEM",
        "interval": "120",
        "threshold_shift": "+0.10",
        "domain": "exotic_derivatives",
        "family": "cross_asset_basis_training",
        "correlation_peers": "pairs_correlation,statistical_arbitrage,crypto_futures_basis,futures_rates_curve,futures_commodity_macro",
    },
    "volatility_arbitrage": {
        "symbols": "SPY,QQQ,IWM,DIA,VIXY,UVXY,SVXY,TLT,GLD,UUP,AAPL,MSFT,NVDA,TSLA,AMD,AVGO,META,AMZN,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,SVXY,XLK,XLF,XLE,SMH,/ES,/NQ,/CL,/GC",
        "interval": "120",
        "threshold_shift": "+0.12",
        "domain": "exotic_derivatives",
        "family": "volatility_arbitrage",
        "correlation_peers": "variance_volatility_swaps,volatility,dispersion_trading,vanna_volga_hedging,black_swan_hedging",
    },
    "quant_pricing_models": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,AAPL,MSFT,NVDA,TSLA,AMD,AVGO,META,AMZN,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,/ES,/NQ,/ZN,/CL,/GC",
        "interval": "180",
        "threshold_shift": "+0.12",
        "domain": "quant_models",
        "family": "quant_pricing_models",
        "correlation_peers": "options_on_futures,volatility_arbitrage,variance_volatility_swaps,second_third_order_greeks",
    },
    "state_space_models": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,AAPL,MSFT,NVDA,TSLA,JPM,XOM,XLK,XLF,XLE,XLV",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,XLK,XLF,XLE,/ES,/NQ,/ZN,/CL",
        "interval": "150",
        "threshold_shift": "+0.10",
        "domain": "quant_models",
        "family": "state_space_models",
        "correlation_peers": "statistical_arbitrage,pairs_correlation,order_flow_market_microstructure,market_making_liquidity",
    },
    "tail_dependency_risk": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,UVXY,XLU,XLP,XLV,BIL,SGOV",
        "context_symbols": "SPY,QQQ,IWM,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,UVXY,/ES,/NQ,/ZN,/CL,/GC",
        "interval": "240",
        "threshold_shift": "+0.14",
        "domain": "quant_models",
        "family": "tail_dependency_risk",
        "correlation_peers": "tail_risk_parity,black_swan_hedging,cdo_cubed,sovereign_debt_macro",
    },
    "optimization_research": {
        "symbols": "SPY,QQQ,IWM,DIA,XLK,XLF,XLE,XLV,SMH,AAPL,MSFT,NVDA,AMD,TSLA,JPM,XOM,TLT,GLD,UUP",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,XLK,XLF,XLE,SMH,/ES,/NQ,/RTY",
        "interval": "300",
        "threshold_shift": "+0.14",
        "domain": "quant_models",
        "family": "optimization_research",
        "correlation_peers": "statistical_arbitrage,pairs_correlation,execution_quality,infrastructure_risk",
    },
    "nlp_sentiment_agents": {
        "symbols": "SPY,QQQ,IWM,DIA,AAPL,MSFT,NVDA,TSLA,AMD,META,AMZN,JPM,XOM,TLT,GLD,UUP,VIXY",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,XLK,XLF,XLE,XLV,/ES,/NQ,/ZN,/CL",
        "interval": "180",
        "threshold_shift": "+0.10",
        "domain": "quant_models",
        "family": "nlp_sentiment_agents",
        "correlation_peers": "earnings_event,macro_bulletin,sec_edgar,source_verification,news_source_credibility",
    },
    "adaptive_architectures": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,AAPL,MSFT,NVDA,TSLA,JPM,XOM,XLK,XLF,XLE,SMH",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,XLK,XLF,XLE,SMH,/ES,/NQ,/ZN,/CL",
        "interval": "180",
        "threshold_shift": "+0.12",
        "domain": "quant_models",
        "family": "adaptive_architectures",
        "correlation_peers": "state_space_models,optimization_research,order_flow_market_microstructure,statistical_arbitrage",
    },
    "adversarial_ml_security": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,AAPL,MSFT,NVDA,TSLA,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,XLK,XLF,SMH,/ES,/NQ",
        "interval": "300",
        "threshold_shift": "+0.16",
        "domain": "quant_models",
        "family": "adversarial_ml_security",
        "correlation_peers": "source_verification,training_quality_control,model_lifecycle,nlp_sentiment_agents",
    },
    "low_latency_orchestration": {
        "symbols": "SPY,QQQ,IWM,DIA,XLK,XLF,XLE,SMH,AAPL,MSFT,NVDA,TSLA,AMD,JPM,XOM,VIXY",
        "context_symbols": "SPY,QQQ,IWM,VIXY,UVXY,XLK,XLF,XLE,SMH,/ES,/NQ,/RTY",
        "interval": "60",
        "threshold_shift": "+0.12",
        "domain": "quant_models",
        "family": "low_latency_orchestration",
        "correlation_peers": "order_flow_market_microstructure,high_frequency_market_making,market_making_liquidity,execution_quality",
    },
    "alternative_data_ingestion": {
        "symbols": "SPY,QQQ,IWM,DIA,AAPL,MSFT,NVDA,TSLA,AMD,META,AMZN,JPM,XOM,TLT,GLD,UUP,VIXY",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,XLK,XLF,XLE,XLV,/ES,/NQ,/ZN,/CL",
        "interval": "240",
        "threshold_shift": "+0.12",
        "domain": "quant_models",
        "family": "alternative_data_ingestion",
        "correlation_peers": "nlp_sentiment_agents,macro_bulletin,sec_edgar,source_verification",
    },
    "privacy_zkp_controls": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY",
        "interval": "600",
        "threshold_shift": "+0.18",
        "domain": "quant_models",
        "family": "privacy_zkp_controls",
        "correlation_peers": "security_audit,report_quality,source_verification,quant_pricing_models",
    },
    "gpu_quant_acceleration": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,AAPL,MSFT,NVDA,TSLA,AMD,AVGO,META,AMZN,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,SMH,/ES,/NQ,/ZN,/CL,/GC",
        "interval": "300",
        "threshold_shift": "+0.16",
        "domain": "quant_models",
        "family": "gpu_quant_acceleration",
        "correlation_peers": "quant_pricing_models,state_space_models,optimization_research,second_third_order_greeks",
    },
    "qemc_path_volatility": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,AAPL,MSFT,NVDA,TSLA,AMD,AVGO,META,AMZN,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,/ES,/NQ,/ZN,/CL,/GC",
        "interval": "300",
        "threshold_shift": "+0.16",
        "domain": "quant_models",
        "family": "qemc_path_volatility",
        "correlation_peers": "quant_pricing_models,gpu_quant_acceleration,barrier_lookback_options,volatility_arbitrage",
    },
    "transport_topology_research": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,XLV,SMH",
        "context_symbols": "SPY,QQQ,IWM,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,/ES,/NQ,/ZN,/CL",
        "interval": "300",
        "threshold_shift": "+0.16",
        "domain": "quant_models",
        "family": "transport_topology_research",
        "correlation_peers": "tail_dependency_risk,pairs_correlation,cross_asset_basis_training,state_space_models",
    },
    "neural_sde_kan_hedging": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,AAPL,MSFT,NVDA,TSLA,AMD,AVGO,META,AMZN,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,SMH,/ES,/NQ,/ZN,/CL,/GC",
        "interval": "300",
        "threshold_shift": "+0.16",
        "domain": "quant_models",
        "family": "neural_sde_kan_hedging",
        "correlation_peers": "state_space_models,adaptive_architectures,second_third_order_greeks,vanna_volga_hedging",
    },
    "order_flow_toxicity": {
        "symbols": "SPY,QQQ,IWM,DIA,XLK,XLF,XLE,SMH,AAPL,MSFT,NVDA,TSLA,AMD,JPM,XOM,VIXY,UVXY",
        "context_symbols": "SPY,QQQ,IWM,VIXY,UVXY,XLK,XLF,XLE,SMH,/ES,/NQ,/RTY",
        "interval": "60",
        "threshold_shift": "+0.14",
        "domain": "quant_models",
        "family": "order_flow_toxicity",
        "correlation_peers": "order_flow_market_microstructure,high_frequency_market_making,low_latency_orchestration,execution_quality",
    },
    "signature_hawkes_generators": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,AAPL,MSFT,NVDA,TSLA,AMD,META,AMZN,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,/ES,/NQ,/ZN,/CL,/GC",
        "interval": "300",
        "threshold_shift": "+0.16",
        "domain": "quant_models",
        "family": "signature_hawkes_generators",
        "correlation_peers": "qemc_path_volatility,transport_topology_research,order_flow_toxicity,macro_bulletin",
    },
    "crowd_physics_games": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,XLV,SMH,AAPL,MSFT,NVDA,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,/ES,/NQ,/ZN,/CL",
        "interval": "300",
        "threshold_shift": "+0.16",
        "domain": "quant_models",
        "family": "crowd_physics_games",
        "correlation_peers": "tail_dependency_risk,tail_risk_parity,black_swan_hedging,adaptive_architectures",
    },
    "lit_order_book_transformers": {
        "symbols": "SPY,QQQ,IWM,DIA,XLK,XLF,XLE,SMH,AAPL,MSFT,NVDA,TSLA,AMD,JPM,XOM,VIXY,UVXY,TQQQ,SQQQ",
        "context_symbols": "SPY,QQQ,IWM,VIXY,UVXY,XLK,XLF,XLE,SMH,/ES,/NQ,/RTY",
        "interval": "45",
        "threshold_shift": "+0.16",
        "domain": "quant_models",
        "family": "lit_order_book_transformers",
        "correlation_peers": "order_flow_toxicity,order_flow_market_microstructure,high_frequency_market_making,low_latency_orchestration",
    },
    "critic_hmm_pinsde": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,AAPL,MSFT,NVDA,TSLA,AMD,AVGO,META,AMZN,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,SMH,/ES,/NQ,/ZN,/CL,/GC",
        "interval": "240",
        "threshold_shift": "+0.16",
        "domain": "quant_models",
        "family": "critic_hmm_pinsde",
        "correlation_peers": "state_space_models,adaptive_architectures,neural_sde_kan_hedging,transport_topology_research",
    },
    "causal_omni_symbolic": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,AAPL,MSFT,NVDA,TSLA,AMD,META,AMZN,JPM,XOM,LLY,UNH,COST,WMT",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,XLK,XLF,XLE,XLV,/ES,/NQ,/ZN,/CL",
        "interval": "300",
        "threshold_shift": "+0.16",
        "domain": "quant_models",
        "family": "causal_omni_symbolic",
        "correlation_peers": "alternative_data_ingestion,nlp_sentiment_agents,critic_hmm_pinsde,source_verification",
    },
    "rlbf_dms_equivariant": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,AAPL,MSFT,NVDA,TSLA,AMD,AVGO,META,AMZN,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,SMH,/ES,/NQ,/ZN,/CL,/GC",
        "interval": "240",
        "threshold_shift": "+0.16",
        "domain": "quant_models",
        "family": "rlbf_dms_equivariant",
        "correlation_peers": "critic_hmm_pinsde,lit_order_book_transformers,adaptive_architectures,gpu_quant_acceleration",
    },
    "arbitrage_execution_safety": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,SMH,AAPL,MSFT,NVDA,TSLA,AMD,JPM,XOM,TQQQ,SQQQ",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,SMH,/ES,/NQ,/RTY,/ZN,/CL",
        "interval": "240",
        "threshold_shift": "+0.16",
        "domain": "quant_models",
        "family": "arbitrage_execution_safety",
        "correlation_peers": "quant_pricing_models,rlbf_dms_equivariant,lit_order_book_transformers,low_latency_orchestration",
    },
    "geometry_spillover_durability": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,XLV,SMH,AAPL,MSFT,NVDA,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,XLV,SMH,/ES,/NQ,/RTY,/ZN,/CL,/GC",
        "interval": "300",
        "threshold_shift": "+0.16",
        "domain": "quant_models",
        "family": "geometry_spillover_durability",
        "correlation_peers": "tail_dependency_risk,transport_topology_research,signature_hawkes_generators,adaptive_architectures",
    },
    "institutional_data_plumbing": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,SMH,AAPL,MSFT,NVDA,JPM,XOM,BTC-USD,ETH-USD,SOL-USD,IBIT,ETHA,MSTR,COIN",
        "context_symbols": "SPY,QQQ,IWM,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,SMH,BTC-USD,ETH-USD,SOL-USD,IBIT,ETHA,MSTR,COIN,/ES,/NQ,/RTY,/ZN,/CL,/GC",
        "interval": "900",
        "min_interval": "600",
        "threshold_shift": "+0.18",
        "domain": "quant_models",
        "family": "institutional_data_plumbing",
        "correlation_peers": "alternative_data_ingestion,nlp_sentiment_agents,order_flow_market_microstructure,low_latency_orchestration,causal_omni_symbolic,feature_store",
        "source_gated": "1",
        "source_profile": "shift_lob_dex,quantconnect_mia,bloomberg_lunarcrush,whale_glassnode,flink_mlx,feast_tecton",
    },
    "lobdif_crisis_microstructure": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,SMH,AAPL,MSFT,NVDA,TSLA,AMD,JPM,XOM,TQQQ,SQQQ",
        "context_symbols": "SPY,QQQ,IWM,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,SMH,/ES,/NQ,/RTY,/ZN,/CL,/GC",
        "interval": "420",
        "min_interval": "300",
        "threshold_shift": "+0.18",
        "domain": "quant_models",
        "family": "lobdif_crisis_microstructure",
        "correlation_peers": "lit_order_book_transformers,order_flow_toxicity,order_flow_market_microstructure,arbitrage_execution_safety,low_latency_orchestration",
        "source_gated": "1",
        "source_profile": "listed_quotes,listed_option_chains,market_microstructure_proxy,synthetic_crisis_replay",
    },
    "macro_crisis_scenario_lab": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,SHY,LQD,HYG,JNK,GLD,UUP,VIXY,UVXY,XLF,KRE,XLRE,XLE,XLK,SMH,XLV,XLP,XLU,XRT,JETS,AAPL,MSFT,NVDA,JPM,BAC,SCHW,XOM,DBC,USO",
        "context_symbols": "SPY,QQQ,IWM,TLT,IEF,SHY,LQD,HYG,JNK,GLD,UUP,VIXY,UVXY,XLF,KRE,XLRE,XLE,XLV,XLP,XLU,XRT,JETS,DBC,USO,/ES,/NQ,/RTY,/ZN,/CL,/GC",
        "interval": "900",
        "min_interval": "600",
        "threshold_shift": "+0.18",
        "domain": "quant_models",
        "family": "macro_crisis_scenario_lab",
        "correlation_peers": "black_swan_hedging,tail_risk_parity,sovereign_debt_macro,geometry_spillover_durability,tail_dependency_risk",
        "source_gated": "1",
        "source_profile": "fed_2026_supervisory_scenarios,fed_2026_source_plumbing,fed_2026_stress_module_map,public_macro_stress_dataset,covid_2020_pandemic_replay,synthetic_crisis_replay",
    },
    "xva_counterparty_margin": {
        "symbols": "SPY,QQQ,TLT,IEF,SHY,LQD,HYG,JNK,AGG,BND,KRE,XLF,JPM,BAC,GS,MS,SCHW,UUP,GLD,VIXY",
        "context_symbols": "SPY,QQQ,TLT,IEF,SHY,LQD,HYG,JNK,AGG,BND,KRE,XLF,JPM,BAC,GS,MS,UUP,GLD,VIXY,/ZT,/ZF,/ZN,/ZB",
        "interval": "900",
        "min_interval": "600",
        "threshold_shift": "+0.20",
        "domain": "quant_models",
        "family": "xva_counterparty_margin",
        "correlation_peers": "rates_credit_macro,credit_derivatives_cdx_cds,repo_securities_lending,synthetic_cdo,tail_dependency_risk",
        "source_gated": "1",
        "source_profile": "otc_counterparty_proxy,isda_simm_proxy,collateral_margin_proxy,credit_spread_proxy",
    },
    "credit_derivatives_cdx_cds": {
        "symbols": "LQD,HYG,JNK,USHY,EMB,AGG,BND,TLT,IEF,KRE,XLF,JPM,BAC,C,GS,MS,BLK,SCHW,SPY,QQQ,VIXY,UUP",
        "context_symbols": "LQD,HYG,JNK,USHY,EMB,AGG,BND,TLT,IEF,KRE,XLF,JPM,BAC,SPY,QQQ,VIXY,UUP,GLD,/ZN,/ZB",
        "interval": "900",
        "min_interval": "600",
        "threshold_shift": "+0.20",
        "domain": "quant_models",
        "family": "credit_derivatives_cdx_cds",
        "correlation_peers": "synthetic_cdo,cdo_squared,cdo_cubed,rates_credit_macro,xva_counterparty_margin",
        "source_gated": "1",
        "source_profile": "cdx_itraxx_proxy,single_name_cds_proxy,hazard_recovery_curve,tranche_waterfall",
    },
    "securitized_products_mbs_abs_clo": {
        "symbols": "MBB,VMBS,AGG,BND,TLT,IEF,SHY,LQD,HYG,JNK,USHY,KRE,XLF,JPM,BAC,GS,MS,SPY,QQQ,VIXY,UUP",
        "context_symbols": "MBB,VMBS,AGG,BND,TLT,IEF,SHY,LQD,HYG,JNK,KRE,XLF,JPM,BAC,SPY,QQQ,VIXY,UUP,/ZN,/ZB",
        "interval": "1200",
        "min_interval": "900",
        "threshold_shift": "+0.22",
        "domain": "quant_models",
        "family": "securitized_products_mbs_abs_clo",
        "correlation_peers": "rates_credit_macro,credit_derivatives_cdx_cds,sovereign_debt_macro,macro_crisis_scenario_lab",
        "source_gated": "1",
        "source_profile": "mbs_abs_clo_proxy,prepayment_oas_proxy,loan_pool_credit_proxy,public_deal_documents",
    },
    "repo_securities_lending": {
        "symbols": "BIL,SGOV,SHY,IEF,TLT,AGG,BND,USFR,TFLO,FLOT,LQD,HYG,JNK,KRE,XLF,SPY,QQQ,IWM,UUP,VIXY",
        "context_symbols": "BIL,SGOV,SHY,IEF,TLT,AGG,BND,USFR,TFLO,FLOT,LQD,HYG,JNK,KRE,XLF,SPY,QQQ,IWM,UUP,VIXY,/ZT,/ZF,/ZN",
        "interval": "900",
        "min_interval": "600",
        "threshold_shift": "+0.20",
        "domain": "quant_models",
        "family": "repo_securities_lending",
        "correlation_peers": "xva_counterparty_margin,sovereign_debt_macro,cash_rotation_tactical,market_data_tape_normalization",
        "source_gated": "1",
        "source_profile": "repo_sofr_proxy,securities_lending_proxy,borrow_fee_proxy,short_interest_locate_pressure",
    },
    "market_data_tape_normalization": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,XLV,SMH,AAPL,MSFT,NVDA,TSLA,AMD,JPM,XOM,TQQQ,SQQQ",
        "context_symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,XLV,SMH,AAPL,MSFT,NVDA,TSLA,AMD,JPM,XOM,/ES,/NQ,/RTY",
        "interval": "600",
        "min_interval": "420",
        "threshold_shift": "+0.20",
        "domain": "quant_models",
        "family": "market_data_tape_normalization",
        "correlation_peers": "order_flow_market_microstructure,institutional_data_plumbing,lit_order_book_transformers,lobdif_crisis_microstructure",
        "source_gated": "1",
        "source_profile": "opra_nbbo_taq_sip_proxy,mbo_mbp_depth_proxy,dark_pool_off_exchange_proxy",
    },
    "provider_adapter_verification": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,SMH,BTC-USD,ETH-USD,SOL-USD,IBIT,ETHA,MSTR,COIN",
        "context_symbols": "SPY,QQQ,IWM,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,SMH,BTC-USD,ETH-USD,SOL-USD,IBIT,ETHA,MSTR,COIN,/ES,/NQ,/ZN,/CL,/GC",
        "interval": "1200",
        "min_interval": "900",
        "threshold_shift": "+0.22",
        "domain": "quant_models",
        "family": "provider_adapter_verification",
        "correlation_peers": "institutional_data_plumbing,alternative_data_ingestion,market_data_tape_normalization,proof_quantum_formal_backends",
        "source_gated": "1",
        "source_profile": "provider_capability_matrix,credential_gate_state,vendor_entitlement_audit,rate_limit_freshness",
    },
    "proof_quantum_formal_backends": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,SMH,AAPL,MSFT,NVDA,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,SMH,AAPL,MSFT,NVDA,JPM,XOM,/ES,/NQ,/ZN",
        "interval": "1200",
        "min_interval": "900",
        "threshold_shift": "+0.22",
        "domain": "quant_models",
        "family": "proof_quantum_formal_backends",
        "correlation_peers": "privacy_zkp_controls,arbitrage_execution_safety,qemc_path_volatility,provider_adapter_verification",
        "source_gated": "1",
        "source_profile": "zkp_proof_stack,formal_model_checker,quantum_backend,photonic_backend,backend_fallback_safety",
    },
    "model_risk_validation": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,SMH,AAPL,MSFT,NVDA,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,SMH,AAPL,MSFT,NVDA,JPM,XOM,/ES,/NQ,/ZN,/CL,/GC",
        "interval": "900",
        "min_interval": "600",
        "threshold_shift": "+0.20",
        "domain": "quant_models",
        "family": "model_risk_validation",
        "correlation_peers": "feature_quality_data_confidence,system_governor_expansion,portfolio_construction,macro_crisis_scenario_lab",
        "source_gated": "0",
        "source_profile": "calibration_decay,challenger_drift,overfit_leakage,stress_replay",
    },
    "transaction_cost_slippage_intelligence": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,XLV,SMH,AAPL,MSFT,NVDA,TSLA,AMD,JPM,XOM,TQQQ,SQQQ",
        "context_symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,XLV,SMH,AAPL,MSFT,NVDA,TSLA,AMD,JPM,XOM,/ES,/NQ,/RTY",
        "interval": "420",
        "min_interval": "300",
        "threshold_shift": "+0.22",
        "domain": "quant_models",
        "family": "transaction_cost_slippage_intelligence",
        "correlation_peers": "liquidity_regime,market_data_tape_normalization,order_flow_market_microstructure,paper_trade_lock",
        "source_gated": "1",
        "source_profile": "spread_decay,fill_realism,queue_position,route_quality,paper_live_slippage",
    },
    "portfolio_construction": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,SHY,TIP,LQD,HYG,JNK,AGG,BND,GLD,SLV,DBC,UUP,EFA,EEM,XLK,XLF,XLE,XLV,SMH",
        "context_symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,SHY,TIP,LQD,HYG,JNK,AGG,BND,GLD,SLV,DBC,UUP,EFA,EEM,XLK,XLF,XLE,XLV,SMH,/ES,/NQ,/RTY,/ZN,/CL,/GC",
        "interval": "900",
        "min_interval": "600",
        "threshold_shift": "+0.20",
        "domain": "quant_models",
        "family": "portfolio_construction",
        "correlation_peers": "tail_risk_parity,black_swan_hedging,rates_credit_macro,model_risk_validation",
        "source_gated": "0",
        "source_profile": "exposure_netting,hedge_ratio,capital_efficiency,sleeve_conflict",
    },
    "event_intelligence": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,SHY,LQD,HYG,JNK,GLD,UUP,VIXY,KRE,XLF,XLK,SMH,AAPL,MSFT,NVDA,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,IEF,SHY,LQD,HYG,JNK,GLD,UUP,VIXY,KRE,XLF,XLK,SMH,AAPL,MSFT,NVDA,JPM,XOM,/ES,/NQ,/ZN,/CL,/GC",
        "interval": "600",
        "min_interval": "420",
        "threshold_shift": "+0.22",
        "domain": "quant_models",
        "family": "event_intelligence",
        "correlation_peers": "macro_crisis_scenario_lab,sovereign_debt_macro,rates_credit_macro,liquidity_regime",
        "source_gated": "1",
        "source_profile": "fed_speaker,treasury_auction,cpi_pce_nfp,earnings_cluster,geopolitical_shock",
    },
    "feature_quality_data_confidence": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,SMH,AAPL,MSFT,NVDA,JPM,XOM,BTC-USD,ETH-USD",
        "context_symbols": "SPY,QQQ,IWM,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,SMH,AAPL,MSFT,NVDA,JPM,XOM,BTC-USD,ETH-USD,/ES,/NQ,/ZN",
        "interval": "600",
        "min_interval": "420",
        "threshold_shift": "+0.20",
        "domain": "quant_models",
        "family": "feature_quality_data_confidence",
        "correlation_peers": "provider_adapter_verification,model_risk_validation,feature_store,system_governor_expansion",
        "source_gated": "0",
        "source_profile": "missing_data,stale_feature,source_disagreement,label_confidence",
    },
    "liquidity_regime": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,XLV,SMH,AAPL,MSFT,NVDA,TSLA,AMD,JPM,XOM,TQQQ,SQQQ",
        "context_symbols": "SPY,QQQ,IWM,DIA,TLT,GLD,UUP,VIXY,UVXY,XLK,XLF,XLE,XLV,SMH,AAPL,MSFT,NVDA,TSLA,AMD,JPM,XOM,/ES,/NQ,/RTY",
        "interval": "300",
        "min_interval": "240",
        "threshold_shift": "+0.23",
        "domain": "quant_models",
        "family": "liquidity_regime",
        "correlation_peers": "transaction_cost_slippage_intelligence,market_data_tape_normalization,order_flow_toxicity,lobdif_crisis_microstructure",
        "source_gated": "1",
        "source_profile": "auction_imbalance,quote_fade,thin_book,halt_reopen,liquidity_cliff",
    },
    "system_governor_expansion": {
        "symbols": "SPY,QQQ,IWM,DIA,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,SMH,AAPL,MSFT,NVDA,JPM,XOM",
        "context_symbols": "SPY,QQQ,IWM,TLT,IEF,LQD,HYG,JNK,GLD,UUP,VIXY,XLK,XLF,XLE,SMH,AAPL,MSFT,NVDA,JPM,XOM,/ES,/NQ,/ZN",
        "interval": "900",
        "min_interval": "600",
        "threshold_shift": "+0.24",
        "domain": "quant_models",
        "family": "system_governor_expansion",
        "correlation_peers": "memory_efficiency,storage_backpressure,global_halt_refresh,feature_quality_data_confidence,liquidity_regime",
        "source_gated": "0",
        "source_profile": "collector_priority,cpu_memory_backlog,global_halt_pressure,adaptive_sampling",
    },
}


def _runtime_profile(simulate: bool) -> str:
    profile = os.getenv("BOT_RUNTIME_PROFILE", "sim" if simulate else "live").strip().lower()
    return profile if profile in {"sim", "live"} else ("sim" if simulate else "live")


def _bootstrap_runtime_env(base_env: dict[str, str], profile: str) -> dict[str, str]:
    if not LOAD_RUNTIME_ENV.exists():
        return base_env
    source_cmd = (
        f"source {shlex.quote(str(LOAD_RUNTIME_ENV))} {shlex.quote(profile)} --quiet >/dev/null 2>&1 && env -0"
    )
    result = subprocess.run(
        ["/bin/zsh", "-lc", source_cmd],
        cwd=str(PROJECT_ROOT),
        env=base_env,
        capture_output=True,
        text=False,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        return base_env
    merged = base_env.copy()
    for chunk in result.stdout.split(b"\0"):
        if not chunk or b"=" not in chunk:
            continue
        key, value = chunk.split(b"=", 1)
        merged[key.decode("utf-8", "ignore")] = value.decode("utf-8", "ignore")
    return merged


def _env_name(profile: str, suffix: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in profile.upper()).strip("_")
    return f"{cleaned}_{suffix}"


def main(default_profile: str | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a dedicated collect-only specialized shadow sleeve.")
    parser.add_argument("--profile", default=default_profile, choices=sorted(SLEEVE_DEFAULTS))
    parser.add_argument("--broker", default=os.getenv("DATA_BROKER", "schwab"), choices=["schwab", "coinbase"])
    parser.add_argument("--simulate", action="store_true", help="Use simulated market feed.")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--context-symbols", default=None)
    parser.add_argument("--interval-seconds", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("SPECIALIZED_SLEEVE_MAX_ITERS", "0")))
    parser.add_argument("--auto-retrain", action="store_true", default=False)
    args = parser.parse_args()

    if not args.profile:
        parser.error("--profile is required")

    if not VENV_PY.exists():
        print(f"ERROR: missing venv python: {VENV_PY}")
        return 2
    if not SHADOW_LOOP.exists():
        print(f"ERROR: missing shadow loop script: {SHADOW_LOOP}")
        return 2

    defaults = SLEEVE_DEFAULTS[args.profile]
    symbols = args.symbols or os.getenv(_env_name(args.profile, "SYMBOLS"), defaults["symbols"])
    context_symbols = args.context_symbols or os.getenv(_env_name(args.profile, "CONTEXT_SYMBOLS"), defaults["context_symbols"])
    interval_seconds = int(args.interval_seconds or os.getenv(_env_name(args.profile, "INTERVAL"), defaults["interval"]))
    min_interval_seconds = int(os.getenv(_env_name(args.profile, "MIN_INTERVAL"), defaults.get("min_interval", "30")))

    env = _bootstrap_runtime_env(os.environ.copy(), _runtime_profile(args.simulate))
    env["MARKET_DATA_ONLY"] = "1"
    env["ALLOW_ORDER_EXECUTION"] = "0"
    env["AUTO_RETRAIN_ON_GOVERNANCE"] = "1" if args.auto_retrain else "0"
    env["SHADOW_PROFILE"] = args.profile
    env["SHADOW_DOMAIN"] = str(defaults.get("domain") or "equities")
    env["SLEEVE_LIFECYCLE_STATE"] = "data_collection_only"
    env["TRAINING_EXCLUDED_UNTIL_READY"] = "1"
    env["SLEEVE_COLLECTION_FAMILY"] = str(defaults.get("family") or args.profile)
    env["SLEEVE_CORRELATION_PEERS"] = str(defaults.get("correlation_peers") or "")
    if str(defaults.get("source_gated") or "0") == "1":
        env["SOURCE_CREDENTIAL_GATED"] = "1"
        env["SLEEVE_SOURCE_PROFILE"] = str(defaults.get("source_profile") or "")
        env["INSTITUTIONAL_DATA_PLUMBING_SOURCE_PROFILE"] = str(defaults.get("source_profile") or "")
        env["ALLOW_SYNTHETIC_VENDOR_PULLS"] = "0"
        env["FEATURE_STORE_SYMMETRY_GUARD_ENABLED"] = "1"
        env["EVENT_STREAM_PIPELINE_GUARD_ENABLED"] = "1"
    exotic_research_only = env["SHADOW_DOMAIN"] == "exotic_derivatives" or is_exotic_derivative_sleeve(args.profile)
    quant_research_only = env["SHADOW_DOMAIN"] == "quant_models"
    env["EXOTIC_DERIVATIVE_RESEARCH_ONLY"] = "1" if exotic_research_only else "0"
    env["EXOTIC_PROXY_PLUMBING_ENABLED"] = "1" if exotic_research_only else env.get("EXOTIC_PROXY_PLUMBING_ENABLED", "0")
    env["EXOTIC_DIRECT_EXECUTION_ALLOWED"] = "0" if exotic_research_only else env.get("EXOTIC_DIRECT_EXECUTION_ALLOWED", "0")
    if exotic_research_only:
        env["TOP_BOT_PAPER_TRADING_ENABLED"] = "0"
        env["TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED"] = "0"
        env["EXECUTION_LANE_ENABLED"] = "0"
        env["MASTER_EXECUTION_LANE_ENABLED"] = "0"
        env["INLINE_PAPER_EXECUTION_ENABLED"] = "0"
        env["RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR"] = "0"
        env["TOP_BOT_ENABLE_LIVE_EXECUTION"] = "0"
    if quant_research_only:
        env["QUANT_MODEL_FEATURE_PLUMBING_ENABLED"] = "1"
        env["QUANT_MODEL_RESEARCH_ONLY"] = "1"
        env["TOP_BOT_PAPER_TRADING_ENABLED"] = "0"
        env["TOP_BOT_PAPER_TRADING_OPTIONS_ENABLED"] = "0"
        env["EXECUTION_LANE_ENABLED"] = "0"
        env["MASTER_EXECUTION_LANE_ENABLED"] = "0"
        env["INLINE_PAPER_EXECUTION_ENABLED"] = "0"
        env["RUN_ALL_SLEEVES_WITH_LIVE_EXECUTOR"] = "0"
        env["TOP_BOT_ENABLE_LIVE_EXECUTION"] = "0"
    env.setdefault("SHADOW_THRESHOLD_SHIFT", defaults["threshold_shift"])

    cmd = [
        str(VENV_PY),
        str(SHADOW_LOOP),
        "--broker",
        args.broker,
        "--profile",
        args.profile,
        "--domain",
        env["SHADOW_DOMAIN"],
        "--symbols",
        symbols,
        "--context-symbols",
        context_symbols,
        "--interval-seconds",
        str(max(interval_seconds, min_interval_seconds, 30)),
        "--max-iterations",
        str(args.max_iterations),
    ]
    if args.simulate:
        cmd.append("--simulate")
    if args.auto_retrain:
        cmd.append("--auto-retrain")

    print("Starting specialized collect-only shadow sleeve...")
    print("Profile:", args.profile)
    print("Symbols:", symbols)
    print("Context symbols:", context_symbols)
    print("Collect only:", env["SLEEVE_LIFECYCLE_STATE"])
    print("Command:", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
