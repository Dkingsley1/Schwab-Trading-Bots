#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.ops.long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload
else:
    from .long_runtime_common import PROJECT_ROOT, iso_now, ordered_unique, write_payload


DEFAULT_OUT_PATH = PROJECT_ROOT / "governance" / "health" / "sleeve_ticker_universe_latest.json"
DEFAULT_OVERRIDE_PATH = PROJECT_ROOT / "config" / ".env.sleeve_ticker_universe_override"
UNIVERSE_VERSION = "sleeve_ticker_universe_v2"

AI_INFRA_POWER_SYMBOLS = [
    "VRT", "GEV", "CEG", "NRG", "VST", "PWR", "ALAB", "CRWV", "CLS", "JBL", "SANM", "COHR",
]
FACTOR_STYLE_SYMBOLS = ["MTUM", "QUAL", "USMV", "VLUE", "IWF", "IWD", "IWO", "IWN", "IWC", "SPHB"]
CREDIT_LIQUIDITY_SYMBOLS = ["BKLN", "SRLN", "JAAA", "CLOA", "PFF", "PGX"]
VOLATILITY_CURVE_SYMBOLS = ["VXX", "VIXM", "VXZ"]
MARKET_PLUMBING_SYMBOLS = ["NDAQ", "CBOE", "MKTX", "KBE", "KIE", "IAI", "ARES", "OWL", "CG"]
GLOBAL_BELLWETHER_SYMBOLS = ["NVO", "AZN", "BABA", "PDD", "BIDU", "JD", "PGR", "ALL", "PHM", "TOL", "NVR", "BRK.B"]

TICKER_1000_EXPANSION_SECTIONS: dict[str, list[str]] = {
    "breadth_style_etfs": [
        "ACWI", "VT", "VXUS", "QQQM", "QQQJ", "MGK", "VUG", "VONG", "VB", "VO", "IJH", "IJR", "IJS", "IJT", "IJJ", "SCHA",
        "SCHM", "SCHX", "SCHG", "SCHV", "SCHB", "SPHQ", "SPYG", "SPYV", "IVE", "IVW", "IUSG", "IUSV", "RPG", "RPV", "EFV",
        "EFG", "GMF", "EPP", "ILF", "AIA", "AAXJ",
    ],
    "industry_country_etfs": [
        "BBH", "IHE", "IHI", "IHF", "IHA", "KCE", "KBWB", "EMQQ", "CQQQ", "MCHI", "ASHR", "GXC", "DXJ", "HEDJ", "FEZ",
        "EZU", "SMIN", "PIN", "KSA", "UAE", "EIS", "ARGT", "EWZS", "ECH", "EPU", "EIDO",
    ],
    "thematic_commodity_credit_etfs": [
        "ARKG", "ARKQ", "ARKF", "TAN", "FAN", "ICLN", "QCLN", "PBW", "LIT", "REMX", "COPX", "PICK", "SLX", "WOOD", "CUT",
        "MOO", "PHO", "FIW", "CGW", "GRID", "HAIL", "DRIV", "IDRV", "LRNZ", "ROBO", "IRBO", "ROBT", "XSD", "PSI", "FTXL",
        "XNTK", "FDN", "XWEB", "XSW", "XTL", "PKB", "REZ", "KBWY", "MORT", "REM", "PFFD", "VRP", "ANGL", "FALN", "HYLB",
        "SHYG", "SJNK", "BSCO", "BSCQ", "IBHE", "IBHF", "UCO", "SCO", "UNL", "UGA", "BNO", "OILK", "USL", "JO", "NIB",
        "BAL", "CANE", "BAR", "AAAU", "GLTR",
    ],
    "ai_power_uranium_infra": [
        "CCJ", "LEU", "UEC", "NXE", "DNN", "UUUU", "SMR", "OKLO", "BWXT", "NNE", "LTBR", "URNM", "FIX", "STRL", "ACM", "J",
        "FLR", "MTZ", "DY", "PRIM", "WSO", "TT", "AYI", "GNRC", "HUBB",
    ],
    "international_adrs": [
        "SONY", "TM", "HMC", "ASX", "UMC", "SAP", "ASND", "ARGX", "SNY", "NVS", "GSK", "BHP", "RIO", "VALE", "SHEL", "BP",
        "TTE", "ENB", "SU", "PBR", "EC", "NU", "MELI", "SE", "STNE", "XP", "GLOB", "DESP", "ERJ", "CPNG", "GRAB", "MUFG",
        "BILI", "LI",
    ],
    "liquid_us_equities": [
        "A", "AAL", "AAP", "ABCB", "ABG", "ABM", "ABR", "ABUS", "ACCO", "ACHC", "ACLS", "ACMR", "ACAD", "ACGL", "ADMA",
        "ADT", "ADUS", "AEE", "AEM", "AEHR", "AEIS", "AES", "AFL", "AGCO", "AGIO", "AGNC", "AIRC", "AKAM", "ALB", "ALGM",
        "ALGN", "ALIT", "ALK", "ALLE", "AM", "AMBA", "AMCR", "AME", "AMH", "AMKR", "AMP", "AMR", "AMRC", "AMWD", "AN",
        "ANF", "ANSS", "AOS", "APA", "APG", "APLS", "AQN", "AR", "ARE", "ARCC", "ARCO", "ARHS", "ARLP", "ARQT", "ARRY",
        "ASH", "ASO", "ASTS", "ATI", "ATR", "ATSG", "ATO", "ATRC", "AVA", "AVB", "AVNT", "AVT", "AVY", "AWK", "AX", "AXON",
        "AZEK", "BAH", "BALL", "BANF", "BAX", "BBIO", "BBY", "BC", "BCC", "BCE", "BDC", "BE", "BEAM", "BEN", "BF.B", "BGC",
        "BG", "BHF", "BIO", "BIIB", "BJ", "BKH", "BKR", "BL", "BLD", "BLDR", "BLKB", "BMRN", "BOKF", "BOX", "BR", "BRO",
        "BRKR", "BRX", "BSY", "BURL", "BWA", "BYD", "CACI", "CAG", "CAH", "CALM", "CAR", "CARG", "CASY", "CATY", "CBRE",
        "CCI", "CDW", "CELH", "CFLT", "CHD", "CHE", "CHH", "CHPT", "CHRW", "CIEN", "CINF", "CLF", "CLH", "CLX", "CMA",
        "CMC", "CNA", "CNC", "CNM", "CNQ", "CNS", "CNX", "CNXC", "COKE", "COLB", "COLM", "COO", "COPA", "COR", "CORT",
        "COTY", "CPB", "CPRT", "CPT", "CR", "CRDO", "CRGY", "CRL", "CRS", "CSIQ", "CSGP", "CSL", "CTRA", "CUBE", "CUK",
        "CVLT", "CW", "DAY", "DBX", "DCI", "DCOM", "DDS", "DECK", "DINO", "DLR", "DLTR", "DNUT", "DOC", "DOCN", "DOCS",
        "DOCU", "DOV", "DOW", "DRI", "DT", "DTE", "DUOL", "DV", "EAT", "EBC", "ECL", "EEFT", "EFX", "EG", "EHC", "EIX",
        "EL", "ELF", "EME", "EMN", "EMR", "ENOV", "ENPH", "ENTG", "EPRT", "EQH", "EQR", "EQT", "ERIE", "ES", "ESE", "ESGR",
        "ESS", "ESTC", "EVR", "EVRG", "EW", "EWBC", "EXAS", "EXPD", "EXPE", "EXR", "FHN", "FICO", "FITB", "FIVE", "FIVN",
        "FLEX", "FLO", "FMC", "FND", "FNF", "FOX", "FOXA", "FRT", "FSLR", "FTI", "FTRE", "FTS", "FUL", "FWONK", "FYBR",
        "G", "GATX", "GBCI", "GDDY", "GEHC", "GFF",
    ],
}

TICKER_UNIVERSE_TARGET_COUNT = 1000
TICKER_UNIVERSE_HOT_COUNT = 150
TICKER_UNIVERSE_STANDARD_COUNT = 500

RUNTIME_INTAKE_ENV = {
    "TICKER_NEWS_MAX_SYMBOLS": "1000",
    "TICKER_NEWS_LIMIT_PER_SYMBOL": "4",
    "TICKER_NEWS_MAX_RUNTIME_SECONDS": "420",
    "TICKER_NEWS_TIMEOUT_SECONDS": "3",
    "TICKER_NEWS_SLEEP_SECONDS": "0.01",
    "FREE_EQUITY_REFERENCE_MAX_SYMBOLS": "240",
    "FREE_EQUITY_REFERENCE_MAX_RUNTIME_SECONDS": "120",
    "FREE_EQUITY_REFERENCE_TIMEOUT_SECONDS": "3",
    "SEC_EDGAR_MAX_RUNTIME_SECONDS": "120",
    "SEC_EDGAR_MAX_ARCHIVE_FETCHES": "1",
    "MARKET_MICRO_MAX_RUNTIME_SECONDS": "120",
}

STORAGE_OPTIMIZATION_ENV = {
    "TICKER_UNIVERSE_STORAGE_PROFILE": "tiered_1000_guarded",
    "TICKER_UNIVERSE_SLOW_TIER_DEFER_ON_STORAGE_PRESSURE": "1",
    "TICKER_UNIVERSE_CONTEXT_WRITE_COMPACTION": "1",
    "RETENTION_STALE_STAGE_ENABLED": "1",
    "RETENTION_STALE_PURGE_ENABLED": "1",
    "RETENTION_STALE_PURGE_MAX_GB": "8",
    "RETENTION_ARCHIVE_COLD_EXPORT_FORMAT": "parquet",
    "RETENTION_ARCHIVE_COLD_EXPORT_COMPRESSION": "zstd",
    "SQL_LINK_SERVICE_JSON_FILE_SYNC_MIN_INTERVAL_SECONDS": "900",
}

DATA_INTAKE_ROUTES: dict[str, dict[str, Any]] = {
    "market_micro_context": {
        "groups": ["TICKER_UNIVERSE_STANDARD_SYMBOLS"],
        "mode": "runtime_env_MARKET_MICRO_SYMBOLS_bounded_500",
        "storage_policy": "features_only_runtime_capped",
    },
    "free_equity_reference_context": {
        "groups": ["TICKER_UNIVERSE_STANDARD_SYMBOLS"],
        "mode": "runtime_env_FREE_EQUITY_REFERENCE_SYMBOLS_max_240_per_cycle",
        "storage_policy": "defer_remaining_symbols_by_runtime_budget",
    },
    "sec_edgar_context": {
        "groups": ["TICKER_UNIVERSE_HOT_SYMBOLS"],
        "mode": "runtime_env_SEC_EDGAR_SYMBOLS_hot_only",
        "storage_policy": "archive_fetches_capped",
    },
    "schwab_symbol_news": {
        "groups": ["TICKER_UNIVERSE_ALL_SYMBOLS"],
        "mode": "sleeve_ticker_universe_latest_with_crypto_filtered_runtime_capped",
        "storage_policy": "limit_per_symbol_4",
    },
    "ticker_news_context": {
        "groups": ["TICKER_UNIVERSE_ALL_SYMBOLS"],
        "mode": "runtime_env_TICKER_NEWS_MAX_SYMBOLS_1000",
        "storage_policy": "rss_runtime_capped_limit_per_symbol_4",
    },
    "long_term_sector_rotation": {
        "groups": ["LONG_TERM_SECTOR_SYMBOLS", "LONG_TERM_SECTOR_CONTEXT_SYMBOLS"],
        "mode": "runtime_env_LONG_TERM_SECTOR_SYMBOLS",
    },
    "bond_shadow": {
        "groups": ["BOND_SYMBOLS", "BOND_CONTEXT_SYMBOLS"],
        "mode": "runtime_env_BOND_SYMBOLS",
    },
}

UNIVERSES: dict[str, list[str]] = {
    "SHADOW_SYMBOLS_CORE": [
        "SPY", "QQQ", "DIA", "IWM", "MDY", "VOO", "VTI", "IVV", "SPLG", "RSP",
        *FACTOR_STYLE_SYMBOLS,
        "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "TSM", "ASML", "MU", "ARM", "SMH", "SOXX", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "INTC",
        "AMZN", "GOOG", "GOOGL", "META", "NFLX", "DIS", "WBD", "ORCL", "CRM", "ADBE", "NOW", "PLTR", "SNOW", "SHOP", "UBER", "ABNB",
        "IBM", "ACN", "INTU", "ADP", "PANW", "CRWD", "FTNT", "ANET", "CDNS", "SNPS", "ADSK", "WDAY", "MDB", "DDOG", "NET", "ZS",
        "MRVL", "ADI", "NXPI", "MPWR", "ON", "MCHP", "GFS",
        *AI_INFRA_POWER_SYMBOLS,
        "JPM", "BAC", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA", "C", "WFC", "COF",
        "PNC", "USB", "BK", "BX", "KKR", "APO", "SPGI", "MCO", "ICE", "CME",
        *MARKET_PLUMBING_SYMBOLS,
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "ABT", "PFE", "ISRG", "TMO", "DHR", "AMGN", "GILD", "BMY", "CVS", "CI", "ELV", "HUM", "MDT", "SYK", "BSX", "REGN", "VRTX", "ZTS",
        "NVO", "AZN",
        "COST", "WMT", "HD", "LOW", "MCD", "NKE", "SBUX", "TGT", "TJX", "CMG", "YUM", "DPZ", "ORLY", "AZO", "ROST",
        "CAT", "DE", "GE", "BA", "RTX", "LMT", "NOC", "HON", "ETN", "UPS", "FDX", "UNP", "CSX", "NSC", "WM", "RSG", "PH", "CARR", "ROK",
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "OXY", "LNG", "PSX", "KMI", "WMB", "HAL", "FANG",
        "LIN", "SHW", "APD", "FCX", "NEM", "NEE", "SO", "DUK", "AMT", "EQIX", "PLD",
        "BKNG", "MAR", "HLT", "DAL", "UAL", "LUV",
        "DELL", "HPE", "HPQ", "STX", "WDC", "SNDK", "TEL", "APH", "GLW",
        "TMUS", "CMCSA", "CHTR", "EA", "TTWO",
        "PYPL", "FI", "FIS", "GPN",
        "TRV", "AON", "MMC", "CB", "AJG", "MET", "PRU", "AIG", "PGR", "ALL", "BRK.B",
        "RCL", "CCL", "NCLH", "GM", "F", "LEN", "DHI", "PHM", "TOL", "NVR",
        "BABA", "PDD", "BIDU", "JD",
    ],
    "SHADOW_SYMBOLS_VOLATILE": [
        "SOXL", "SOXS", "TQQQ", "SQQQ", "SPXL", "SPXS", "LABU", "LABD", "UVXY", "VIXY", "SVXY", *VOLATILITY_CURVE_SYMBOLS,
        "FNGU", "FNGD", "TECL", "TECS", "WEBL", "WEBS", "NAIL", "DRV", "ERX", "ERY", "BOIL", "KOLD", "TMF", "TMV",
        "MSTR", "SMCI", "COIN", "TSLA", "AMD", "NVDA", "PLTR", "ARM", "MARA", "RIOT", "CLSK", "HUT", "IREN", "CORZ", "WULF", "APLD", "CIFR", "BTDR", "BITF", "HIVE", "BTBT", "DGHI", "HOOD", "RBLX", "AFRM", "UPST",
        "CVNA", "SOFI", "DKNG", "RIVN", "LCID", "APP", "U", "PATH", "SNAP", "IONQ", "SOUN", "AI",
        "IBIT", "FBTC", "BITB", "ARKB", "ETHA", "ETHE", "BITX", "BITU", "BITI", "WGMI",
    ],
    "SHADOW_SYMBOLS_DEFENSIVE": [
        "TLT", "GLD", "XLV", "XLU", "XLP", "MO", "HYG", "LQD", "UUP", "XLE", "XLF", "XLI", "XLK", "XLY", "XLC", "XLB", "XLRE",
        "XAR", "KRE", "XOP", "IEF", "SHY", "TIP", "TLH", "JNK", "AGG", "BND", "MUB", "IGIB", "USHY", "FLOT", "VGIT", "VCIT", "EMB", "BIL", "SGOV", "USFR", "TFLO",
        "SHV", "JPST", "ICSH", "BSV", "VCSH", "BIV", "GOVT", "MBB",
        "SCHD", "VIG", "DGRO", "HDV", "NOBL", "VYM", "DIVO", "JEPI", "JEPQ", "SPLV", "VTV", "VIGI", "IDV", "DVY", "SPHD", "SDY",
        "JNJ", "PG", "KO", "PEP", "MCD", "ABBV", "ABT", "MRK", "PFE", "T", "VZ", "O", "VICI", "MAIN", "CL", "KMB", "GIS", "HSY", "MDLZ", "KR", "SO", "DUK", "NEE", "ED", "D", "WMT", "COST", "UPS", "WM",
        "PCG", "AEP", "EXC", "SRE", "PEG", "XEL",
        "ITA", "LMT", "NOC", "RTX", "GD", "LHX", "LDOS", "IBB", "XBI", "ITB", "XRT", "VNQ", "IYR",
    ],
    "SHADOW_SYMBOLS_COMMOD_FX_INTL": [
        "DBC", "USO", "UNG", "CORN", "WEAT", "SOYB", "SLV", "GLD", "CPER", "URA", "XME", "GDX", "GDXJ",
        "PDBC", "COMT", "BCI", "DBA", "DBB", "DBE", "IAU", "SGOL", "PPLT", "PALL",
        "UUP", "FXE", "FXY", "FXB", "FXC", "FXA", "CYB", "EUO", "YCS", "UDN", "CEW", "DBV",
        "EFA", "EEM", "IEFA", "VEA", "VWO", "VGK", "EWJ", "FXI", "EWZ", "INDA", "EWU", "EWG", "EWQ", "EWC", "EWA", "EWW", "EWY", "EWT", "IXUS",
        "EPI", "EZA", "TUR", "EWS", "EWH", "ACWX", "EWI", "EWP", "EWN", "EWL", "EWK",
    ],
    "DIVIDEND_SYMBOLS": [
        "SCHD", "VIG", "DGRO", "HDV", "NOBL", "VYM", "DIVO", "JEPI", "JEPQ", "SPYD", "DIV", "FDVV", "SCHY", "SDY",
        "VIGI", "IDV", "DVY", "SPHD", "DGRW",
        "JNJ", "PG", "KO", "PEP", "MCD", "MO", "PM", "ABBV", "ABT", "MRK", "PFE", "T", "VZ", "O", "VICI", "MAIN",
        "CL", "KMB", "GIS", "HSY", "MDLZ", "KR", "SO", "DUK", "NEE", "ED", "D",
        "XOM", "CVX", "COP", "KMI", "MPC", "PSX", "VLO", "EOG", "SLB", "WMB", "MSFT", "AAPL", "COST", "HD", "LOW", "JPM", "BLK", "WMT", "TJX", "UNP", "UPS", "WM", "RSG", "IBM", "ADP",
    ],
    "DIVIDEND_QUALITY_SYMBOLS": [
        "SCHD", "VIG", "DGRO", "HDV", "NOBL", "VYM", "DIVO", "SCHY", "VIGI", "DGRW", "JNJ", "PG", "KO", "PEP", "MCD", "ABBV", "ABT", "MRK", "CL", "KMB", "GIS", "HSY", "MDLZ", "XOM", "CVX", "COP", "O", "VICI", "NEE", "SO", "DUK", "MSFT", "AAPL", "COST", "HD", "LOW", "WMT", "ADP",
    ],
    "BOND_SYMBOLS": [
        "TLT", "IEF", "SHY", "TIP", "LQD", "HYG", "JNK", "AGG", "BND", "TLH", "MUB", "IGIB", "USHY", "FLOT", "VGIT", "VCIT", "EMB", "BIL", "SGOV", "USFR", "TFLO", "MINT", "NEAR", "VTIP", "SCHP",
        "SHV", "JPST", "ICSH", "BSV", "VCSH", "BIV", "GOVT", "MBB", "VGSH", "VGLT", "EDV", "ZROZ", "IUSB", "SCHO", "SCHR", "SCHZ",
        *CREDIT_LIQUIDITY_SYMBOLS,
    ],
    "BOND_CONTEXT_SYMBOLS": [
        "UUP", "GLD", "SPY", "QQQ", "TLT", "IEF", "TLH", "VGIT", "SHY", "TIP", "VTIP", "SCHP", "LQD", "IGIB", "HYG", "JNK", "USHY", "AGG", "BND", "MUB", *CREDIT_LIQUIDITY_SYMBOLS, "XLU", "XLF", "XLE", "VIXY", *VOLATILITY_CURVE_SYMBOLS, "DBC", "USO",
        "SHV", "JPST", "VCSH", "BIV", "GOVT", "MBB", "EDV", "ZROZ", "TBT", "TBF", "TMF", "TMV", "NEE", "XHB", "IYR",
    ],
    "FX_SYMBOLS": ["UUP", "USDU", "FXE", "FXY", "FXB", "FXC", "FXA", "CYB", "EUO", "YCS", "UDN", "CEW", "DBV"],
    "FX_CONTEXT_SYMBOLS": ["SPY", "QQQ", "TLT", "GLD", "UUP", "USDU", "FXE", "FXY", "FXB", "FXC", "FXA", "EFA", "EEM", "USO", "DBC"],
    "COINBASE_WATCH_SYMBOLS": [
        "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LTC-USD", "LINK-USD", "DOGE-USD", "XRP-USD", "ADA-USD", "DOT-USD", "BCH-USD", "UNI-USD", "AAVE-USD", "ATOM-USD", "NEAR-USD", "OP-USD", "ARB-USD",
        "ETC-USD", "XLM-USD", "HBAR-USD", "ALGO-USD", "FIL-USD", "SUI-USD", "INJ-USD", "SEI-USD", "TIA-USD", "PEPE-USD", "SHIB-USD", "BONK-USD", "WIF-USD", "ONDO-USD", "RENDER-USD",
    ],
    "COINBASE_FUTURES_WATCH_SYMBOLS": [
        "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LTC-USD", "LINK-USD", "DOGE-USD", "XRP-USD", "ADA-USD", "DOT-USD", "BCH-USD", "UNI-USD", "AAVE-USD", "ATOM-USD", "NEAR-USD",
    ],
    "COINBASE_WEBSOCKET_SYMBOLS": ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LTC-USD", "LINK-USD", "DOGE-USD", "XRP-USD", "ADA-USD", "DOT-USD", "BCH-USD", "UNI-USD", "AAVE-USD", "ETC-USD"],
    "CRYPTO_MARKET_CONTEXT_SYMBOLS": [
        "BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LTC-USD", "LINK-USD", "DOGE-USD", "XRP-USD", "ADA-USD", "DOT-USD", "BCH-USD", "UNI-USD", "AAVE-USD", "ATOM-USD",
        "NEAR-USD", "OP-USD", "ARB-USD", "ETC-USD", "XLM-USD", "HBAR-USD", "ALGO-USD", "FIL-USD", "SUI-USD", "INJ-USD", "SEI-USD", "TIA-USD", "PEPE-USD", "SHIB-USD", "BONK-USD", "WIF-USD", "ONDO-USD", "RENDER-USD",
    ],
    "LONG_TERM_SECTOR_SYMBOLS": [
        "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "SMH", "SOXX", "ITB", "KRE", "IBB", "ITA", "JETS", "XOP", "OIH", "XME", "GDX", "URA",
        "XBI", "IYR", "VNQ", "XRT", "XHB", "IYT", "IYW", "IGV", "CLOU", "BOTZ", "ARKK", "ARKW", "CIBR", "HACK", "SKYY", "PAVE", "KWEB",
        "VGT", "VHT", "VCR", "VDC", "VIS", "VAW", "VOX", "VPU",
        *FACTOR_STYLE_SYMBOLS, "KBE", "KIE", "IAI",
    ],
    "LONG_TERM_SECTOR_CONTEXT_SYMBOLS": ["SPY", "QQQ", "IWM", "TLT", "GLD", "UUP", "USDU", "VIXY", *VOLATILITY_CURVE_SYMBOLS, "HYG", "LQD", "EFA", "EEM", "DBC", "USO"],
}


def _csv(values: list[str]) -> str:
    return ",".join(ordered_unique(str(value).strip().upper() for value in values if str(value).strip()))


def _unique_symbols(env: dict[str, str]) -> list[str]:
    symbols: list[str] = []
    for key, value in env.items():
        if key.startswith("SLEEVE_TICKER_UNIVERSE_"):
            continue
        symbols.extend(part.strip().upper() for part in value.split(",") if part.strip())
    return ordered_unique(symbols)


def _flatten_sections(sections: dict[str, list[str]]) -> list[str]:
    symbols: list[str] = []
    for values in sections.values():
        symbols.extend(values)
    return ordered_unique(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip())


def _equity_like(symbols: list[str]) -> list[str]:
    return [symbol for symbol in symbols if symbol and not symbol.endswith("-USD")]


def _target_tiers(env: dict[str, str]) -> dict[str, list[str]]:
    base_symbols = _unique_symbols(env)
    expansion_symbols = [symbol for symbol in _flatten_sections(TICKER_1000_EXPANSION_SECTIONS) if symbol not in set(base_symbols)]
    all_symbols = ordered_unique([*base_symbols, *expansion_symbols])[:TICKER_UNIVERSE_TARGET_COUNT]
    hot_symbols = ordered_unique(
        [
            *UNIVERSES["SHADOW_SYMBOLS_CORE"],
            *UNIVERSES["SHADOW_SYMBOLS_VOLATILE"],
            *UNIVERSES["SHADOW_SYMBOLS_DEFENSIVE"],
            *UNIVERSES["BOND_CONTEXT_SYMBOLS"],
            *UNIVERSES["COINBASE_WEBSOCKET_SYMBOLS"],
        ]
    )[:TICKER_UNIVERSE_HOT_COUNT]
    standard_symbols = all_symbols[:TICKER_UNIVERSE_STANDARD_COUNT]
    standard_set = set(standard_symbols)
    slow_symbols = [symbol for symbol in all_symbols if symbol not in standard_set]
    return {
        "TICKER_UNIVERSE_ALL_SYMBOLS": all_symbols,
        "TICKER_UNIVERSE_HOT_SYMBOLS": hot_symbols,
        "TICKER_UNIVERSE_STANDARD_SYMBOLS": standard_symbols,
        "TICKER_UNIVERSE_SLOW_SYMBOLS": slow_symbols,
        "MARKET_MICRO_SYMBOLS": standard_symbols,
        "FREE_EQUITY_REFERENCE_SYMBOLS": standard_symbols,
        "SEC_EDGAR_SYMBOLS": _equity_like(hot_symbols),
        "EXTENDED_QUANT_SYMBOLS": all_symbols,
    }


def _override_lines(payload: dict[str, Any]) -> list[str]:
    env = payload.get("env_overrides") if isinstance(payload.get("env_overrides"), dict) else {}
    lines = [
        "# Auto-managed by scripts/ops/sleeve_ticker_universe_expansion.py",
        f"# Generated at {payload.get('timestamp_utc') or iso_now()}",
    ]
    for key in sorted(env):
        lines.append(f"{key}={shlex.quote(str(env[key]))}")
    return lines


def build_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    del project_root
    base_env = {key: _csv(values) for key, values in UNIVERSES.items()}
    tiered_env = {key: _csv(values) for key, values in _target_tiers(base_env).items()}
    symbol_env = {**base_env, **tiered_env}
    counts = {key: len(value.split(",")) if value else 0 for key, value in symbol_env.items()}
    unique_symbols = _unique_symbols(symbol_env)
    group_slot_count = sum(counts.values())
    sleeve_groups = {
        "equity_core": ["SHADOW_SYMBOLS_CORE", "SHADOW_SYMBOLS_VOLATILE", "SHADOW_SYMBOLS_DEFENSIVE"],
        "cross_asset": ["SHADOW_SYMBOLS_COMMOD_FX_INTL", "FX_SYMBOLS", "FX_CONTEXT_SYMBOLS"],
        "income_rates": ["DIVIDEND_SYMBOLS", "DIVIDEND_QUALITY_SYMBOLS", "BOND_SYMBOLS", "BOND_CONTEXT_SYMBOLS"],
        "crypto": ["COINBASE_WATCH_SYMBOLS", "COINBASE_FUTURES_WATCH_SYMBOLS", "COINBASE_WEBSOCKET_SYMBOLS", "CRYPTO_MARKET_CONTEXT_SYMBOLS"],
        "long_term_sector": ["LONG_TERM_SECTOR_SYMBOLS", "LONG_TERM_SECTOR_CONTEXT_SYMBOLS"],
        "tiered_1000_data_universe": [
            "TICKER_UNIVERSE_HOT_SYMBOLS",
            "TICKER_UNIVERSE_STANDARD_SYMBOLS",
            "TICKER_UNIVERSE_SLOW_SYMBOLS",
            "TICKER_UNIVERSE_ALL_SYMBOLS",
        ],
    }
    return {
        "timestamp_utc": iso_now(),
        "schema_version": 1,
        "ok": True,
        "overall_status": "ready",
        "universe_version": UNIVERSE_VERSION,
        "env_overrides": {
            **symbol_env,
            **RUNTIME_INTAKE_ENV,
            **STORAGE_OPTIMIZATION_ENV,
            "TICKER_UNIVERSE_TARGET_COUNT": str(TICKER_UNIVERSE_TARGET_COUNT),
            "SLEEVE_TICKER_UNIVERSE_ENABLED": "1",
            "SLEEVE_TICKER_UNIVERSE_VERSION": UNIVERSE_VERSION,
            "SLEEVE_TICKER_UNIVERSE_POLICY": "tiered_1000_market_data_only_with_provider_guarded_crypto_websocket_subset",
        },
        "symbol_counts": counts,
        "unique_symbol_count": len(unique_symbols),
        "group_slot_count": group_slot_count,
        "sleeve_groups": sleeve_groups,
        "data_intake_routes": DATA_INTAKE_ROUTES,
        "data_intake_runtime_env": dict(RUNTIME_INTAKE_ENV),
        "storage_optimization_env": dict(STORAGE_OPTIMIZATION_ENV),
        "expansion_sections": {
            "ai_power_data_center_infrastructure": AI_INFRA_POWER_SYMBOLS,
            "factor_style_rotation": FACTOR_STYLE_SYMBOLS,
            "credit_liquidity_stress": CREDIT_LIQUIDITY_SYMBOLS,
            "volatility_curve": VOLATILITY_CURVE_SYMBOLS,
            "market_plumbing_financial_structure": MARKET_PLUMBING_SYMBOLS,
            "global_sector_bellwethers": GLOBAL_BELLWETHER_SYMBOLS,
            **TICKER_1000_EXPANSION_SECTIONS,
        },
        "tier_contract": {
            "target_symbol_count": TICKER_UNIVERSE_TARGET_COUNT,
            "hot_symbol_count": len(tiered_env["TICKER_UNIVERSE_HOT_SYMBOLS"].split(",")),
            "standard_symbol_count": len(tiered_env["TICKER_UNIVERSE_STANDARD_SYMBOLS"].split(",")),
            "slow_symbol_count": len(tiered_env["TICKER_UNIVERSE_SLOW_SYMBOLS"].split(",")),
            "all_symbol_count": len(tiered_env["TICKER_UNIVERSE_ALL_SYMBOLS"].split(",")),
            "hot_tier_policy": "direct context, SEC, and fast health-sensitive collection",
            "standard_tier_policy": "market micro plus free public reference within runtime budgets",
            "slow_tier_policy": "symbol news, training breadth, and deferred reference collection only",
        },
        "safety_contract": {
            "market_data_only": "1",
            "adds_live_execution": False,
            "coinbase_websocket_subset": "kept_smaller_than_full_crypto_watchlist",
            "applies_through_runtime_env_override": True,
            "unique_symbol_target": len(unique_symbols),
            "slow_tier_deferred_on_storage_pressure": True,
        },
    }


def apply_payload(
    project_root: Path,
    payload: dict[str, Any],
    *,
    out_path: Path = DEFAULT_OUT_PATH,
    override_path: Path = DEFAULT_OVERRIDE_PATH,
) -> dict[str, Any]:
    out = out_path if out_path.is_absolute() else project_root / out_path
    override = override_path if override_path.is_absolute() else project_root / override_path
    override.parent.mkdir(parents=True, exist_ok=True)
    override.write_text("\n".join(_override_lines(payload)) + "\n", encoding="utf-8")
    payload = dict(payload)
    payload["apply_result"] = {
        "applied": True,
        "override_path": str(override),
        "health_path": str(out),
    }
    write_payload(out, payload)
    payload["out_path"] = str(out)
    return payload


def _print_human(payload: dict[str, Any]) -> None:
    counts = payload.get("symbol_counts") if isinstance(payload.get("symbol_counts"), dict) else {}
    print(
        "sleeve_ticker_universe "
        f"status={payload.get('overall_status')} "
        f"unique={payload.get('unique_symbol_count')} "
        f"core={counts.get('SHADOW_SYMBOLS_CORE')} "
        f"defensive={counts.get('SHADOW_SYMBOLS_DEFENSIVE')} "
        f"crypto={counts.get('COINBASE_WATCH_SYMBOLS')}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Expand ticker universes across applicable live-data sleeves.")
    parser.add_argument("--apply", action="store_true", help="Write health artifact and runtime env override.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH, help="Health artifact path.")
    parser.add_argument("--override", type=Path, default=DEFAULT_OVERRIDE_PATH, help="Runtime env override path.")
    args = parser.parse_args(argv)

    payload = build_payload(PROJECT_ROOT)
    if args.apply:
        payload = apply_payload(PROJECT_ROOT, payload, out_path=args.out, override_path=args.override)
    else:
        payload = {
            **payload,
            "apply_result": {
                "applied": False,
                "override_path": str(args.override if args.override.is_absolute() else PROJECT_ROOT / args.override),
                "health_path": str(args.out if args.out.is_absolute() else PROJECT_ROOT / args.out),
            },
            "out_path": str(args.out if args.out.is_absolute() else PROJECT_ROOT / args.out),
        }

    if args.json:
        print(json.dumps(payload, ensure_ascii=True, indent=2))
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
