#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OVERRIDE = PROJECT_ROOT / "config" / ".env.memory_efficiency_override"
DEFAULT_OUT = PROJECT_ROOT / "governance" / "health" / "memory_efficiency_control_latest.json"
DEFAULT_REGISTRY = PROJECT_ROOT / "master_bot_registry.json"

DRAIN_FRIENDLY_SQL_OVERRIDES = {
    "SQL_LINK_SERVICE_INTERVAL_SECONDS": "12",
    "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "30",
    "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "180",
    "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "240000",
    "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "180000",
    "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "25",
}

CONCENTRATED_DRAIN_SQL_OVERRIDES = {
    **DRAIN_FRIENDLY_SQL_OVERRIDES,
    "SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN": "1",
    "SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS": "420",
    "SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE": "60",
    "SQL_LINK_SERVICE_SHARD_TRADING_STATE_CHECKPOINT_LINES": "1000",
    "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_STATE_CHECKPOINT_LINES": "1000",
    "SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE": "12000",
}

FALLBACK_PRESETS: dict[str, dict[str, str]] = {
    "air_safe": {
        "COINBASE_SNAPSHOT_MAX_WORKERS": "2",
        "COINBASE_CACHE_MAX_ENTRIES": "128",
        "COINBASE_WEBSOCKET_BOOK_DEPTH": "6",
        "TRADE_BEHAVIOR_BATCH_SIZE": "768",
        "ASYNC_PIPELINE_WORKERS": "3",
        "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "96",
        "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "24",
        "SCHWAB_NEWS_CACHE_MAX_SYMBOLS": "32",
        "SCHWAB_OPTIONS_CHAIN_CACHE_MAX_SYMBOLS": "32",
        "RUNTIME_TRAIN_SAMPLE_STRIDE_FLOOR": "2",
        "RUNTIME_TRAIN_BATCH_SIZE_CAP": "64",
        "RUNTIME_TRAIN_MAX_SAMPLES": "12000",
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "80000",
        "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "50000",
        "AUTO_RETRAIN_SWAP_SOFT_MAX_GB": "8",
        "AUTO_RETRAIN_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "88",
        "MEMORY_THROTTLE_SWAP_SOFT_MAX_GB": "8",
        "MEMORY_THROTTLE_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "88",
        "SQLITE_TEMP_STORE_MODE": "FILE",
        "SQLITE_CACHE_SIZE_KB": "12288",
        "SQLITE_MMAP_SIZE_MB": "128",
        "BOT_OPS_SQLITE_TEMP_STORE_MODE": "FILE",
        "BOT_OPS_SQLITE_CACHE_SIZE_KB": "4096",
        "BOT_OPS_SQLITE_MMAP_SIZE_MB": "32",
        "TOP_BOT_PAPER_TRADING_TOP_N": "4",
        "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "1",
    },
    "pro_balanced": {
        "COINBASE_SNAPSHOT_MAX_WORKERS": "3",
        "COINBASE_CACHE_MAX_ENTRIES": "256",
        "COINBASE_WEBSOCKET_BOOK_DEPTH": "8",
        "TRADE_BEHAVIOR_BATCH_SIZE": "1024",
        "ASYNC_PIPELINE_WORKERS": "4",
        "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "160",
        "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "40",
        "SCHWAB_NEWS_CACHE_MAX_SYMBOLS": "48",
        "SCHWAB_OPTIONS_CHAIN_CACHE_MAX_SYMBOLS": "48",
        "RUNTIME_TRAIN_SAMPLE_STRIDE_FLOOR": "1",
        "RUNTIME_TRAIN_BATCH_SIZE_CAP": "96",
        "RUNTIME_TRAIN_MAX_SAMPLES": "20000",
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "100000",
        "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "70000",
        "AUTO_RETRAIN_SWAP_SOFT_MAX_GB": "16",
        "AUTO_RETRAIN_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "82",
        "MEMORY_THROTTLE_SWAP_SOFT_MAX_GB": "16",
        "MEMORY_THROTTLE_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "82",
        "SQLITE_TEMP_STORE_MODE": "MEMORY",
        "SQLITE_CACHE_SIZE_KB": "20480",
        "SQLITE_MMAP_SIZE_MB": "256",
        "BOT_OPS_SQLITE_TEMP_STORE_MODE": "MEMORY",
        "BOT_OPS_SQLITE_CACHE_SIZE_KB": "6144",
        "BOT_OPS_SQLITE_MMAP_SIZE_MB": "48",
        "TOP_BOT_PAPER_TRADING_TOP_N": "5",
        "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "2",
    },
    "max_throughput": {
        "COINBASE_SNAPSHOT_MAX_WORKERS": "4",
        "COINBASE_CACHE_MAX_ENTRIES": "512",
        "COINBASE_WEBSOCKET_BOOK_DEPTH": "10",
        "TRADE_BEHAVIOR_BATCH_SIZE": "1536",
        "ASYNC_PIPELINE_WORKERS": "6",
        "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "256",
        "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "64",
        "SCHWAB_NEWS_CACHE_MAX_SYMBOLS": "72",
        "SCHWAB_OPTIONS_CHAIN_CACHE_MAX_SYMBOLS": "72",
        "RUNTIME_TRAIN_SAMPLE_STRIDE_FLOOR": "1",
        "RUNTIME_TRAIN_BATCH_SIZE_CAP": "96",
        "RUNTIME_TRAIN_MAX_SAMPLES": "32000",
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "140000",
        "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "90000",
        "AUTO_RETRAIN_SWAP_SOFT_MAX_GB": "24",
        "AUTO_RETRAIN_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "75",
        "MEMORY_THROTTLE_SWAP_SOFT_MAX_GB": "24",
        "MEMORY_THROTTLE_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "75",
        "SQLITE_TEMP_STORE_MODE": "MEMORY",
        "SQLITE_CACHE_SIZE_KB": "32768",
        "SQLITE_MMAP_SIZE_MB": "512",
        "BOT_OPS_SQLITE_TEMP_STORE_MODE": "MEMORY",
        "BOT_OPS_SQLITE_CACHE_SIZE_KB": "8192",
        "BOT_OPS_SQLITE_MMAP_SIZE_MB": "96",
        "TOP_BOT_PAPER_TRADING_TOP_N": "5",
        "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "2",
    },
    "constrained": {
        "COINBASE_SNAPSHOT_MAX_WORKERS": "2",
        "COINBASE_CACHE_MAX_ENTRIES": "96",
        "COINBASE_WEBSOCKET_BOOK_DEPTH": "4",
        "TRADE_BEHAVIOR_BATCH_SIZE": "512",
        "ASYNC_PIPELINE_WORKERS": "2",
        "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "64",
        "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "16",
        "SCHWAB_NEWS_CACHE_MAX_SYMBOLS": "24",
        "SCHWAB_OPTIONS_CHAIN_CACHE_MAX_SYMBOLS": "24",
        "RUNTIME_TRAIN_SAMPLE_STRIDE_FLOOR": "3",
        "RUNTIME_TRAIN_BATCH_SIZE_CAP": "48",
        "RUNTIME_TRAIN_MAX_SAMPLES": "8000",
        "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "60000",
        "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "40000",
        "AUTO_RETRAIN_SWAP_SOFT_MAX_GB": "6",
        "AUTO_RETRAIN_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "90",
        "MEMORY_THROTTLE_SWAP_SOFT_MAX_GB": "6",
        "MEMORY_THROTTLE_SWAP_IGNORE_IF_FREE_PCT_AT_LEAST": "90",
        "SQLITE_TEMP_STORE_MODE": "FILE",
        "SQLITE_CACHE_SIZE_KB": "6144",
        "SQLITE_MMAP_SIZE_MB": "48",
        "SQLITE_ANALYZE_ENABLED": "0",
        "BOT_OPS_SQLITE_TEMP_STORE_MODE": "FILE",
        "BOT_OPS_SQLITE_CACHE_SIZE_KB": "2048",
        "BOT_OPS_SQLITE_MMAP_SIZE_MB": "8",
        "TOP_BOT_PAPER_TRADING_TOP_N": "3",
        "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "1",
    },
}

CREATIVE_SESSION_OVERLAYS: dict[str, dict[str, dict[str, str]]] = {
    "active": {
        "__default__": {
            "SQL_LINK_SERVICE_INTERVAL_SECONDS": "90",
            "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "360",
            "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "1500",
            "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "70000",
            "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "40000",
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "900",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "300",
            "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "600",
            "OPS_WATCHDOG_REFRESH_MAX_AGE_SECONDS": "3600",
            "COINBASE_SNAPSHOT_MAX_WORKERS": "2",
            "COINBASE_CACHE_MAX_ENTRIES": "96",
            "TRADE_BEHAVIOR_BATCH_SIZE": "512",
            "ASYNC_PIPELINE_WORKERS": "2",
            "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "64",
            "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "16",
            "RUNTIME_TRAIN_BATCH_SIZE_CAP": "48",
            "RUNTIME_TRAIN_MAX_SAMPLES": "8000",
            "SQLITE_TEMP_STORE_MODE": "FILE",
            "SQLITE_CACHE_SIZE_KB": "10240",
            "SQLITE_MMAP_SIZE_MB": "96",
            "BOT_OPS_SQLITE_TEMP_STORE_MODE": "FILE",
            "BOT_OPS_SQLITE_CACHE_SIZE_KB": "3072",
            "BOT_OPS_SQLITE_MMAP_SIZE_MB": "24",
            "TOP_BOT_PAPER_TRADING_TOP_N": "3",
            "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "1",
        },
        "max_throughput": {
            "SQL_LINK_SERVICE_INTERVAL_SECONDS": "60",
            "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "240",
            "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "1200",
            "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "90000",
            "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "55000",
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "420",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "180",
            "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "420",
            "OPS_WATCHDOG_REFRESH_MAX_AGE_SECONDS": "2400",
            "COINBASE_SNAPSHOT_MAX_WORKERS": "3",
            "COINBASE_CACHE_MAX_ENTRIES": "192",
            "TRADE_BEHAVIOR_BATCH_SIZE": "768",
            "ASYNC_PIPELINE_WORKERS": "4",
            "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "128",
            "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "32",
            "RUNTIME_TRAIN_BATCH_SIZE_CAP": "64",
            "RUNTIME_TRAIN_MAX_SAMPLES": "14000",
            "SQLITE_TEMP_STORE_MODE": "FILE",
            "SQLITE_CACHE_SIZE_KB": "12288",
            "SQLITE_MMAP_SIZE_MB": "128",
            "BOT_OPS_SQLITE_TEMP_STORE_MODE": "FILE",
            "BOT_OPS_SQLITE_CACHE_SIZE_KB": "4096",
            "BOT_OPS_SQLITE_MMAP_SIZE_MB": "32",
            "TOP_BOT_PAPER_TRADING_TOP_N": "4",
            "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "1",
        },
    },
    "hot": {
        "__default__": {
            "SQL_LINK_SERVICE_INTERVAL_SECONDS": "120",
            "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "480",
            "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "1800",
            "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "50000",
            "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "30000",
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "1200",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "420",
            "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "900",
            "OPS_WATCHDOG_REFRESH_MAX_AGE_SECONDS": "5400",
            "COINBASE_SNAPSHOT_MAX_WORKERS": "1",
            "COINBASE_CACHE_MAX_ENTRIES": "64",
            "TRADE_BEHAVIOR_BATCH_SIZE": "384",
            "ASYNC_PIPELINE_WORKERS": "2",
            "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "48",
            "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "12",
            "RUNTIME_TRAIN_BATCH_SIZE_CAP": "32",
            "RUNTIME_TRAIN_MAX_SAMPLES": "6000",
            "SQLITE_TEMP_STORE_MODE": "FILE",
            "SQLITE_CACHE_SIZE_KB": "8192",
            "SQLITE_MMAP_SIZE_MB": "64",
            "SQLITE_ANALYZE_ENABLED": "0",
            "BOT_OPS_SQLITE_TEMP_STORE_MODE": "FILE",
            "BOT_OPS_SQLITE_CACHE_SIZE_KB": "2048",
            "BOT_OPS_SQLITE_MMAP_SIZE_MB": "16",
            "TOP_BOT_PAPER_TRADING_TOP_N": "2",
            "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "0",
        },
        "max_throughput": {
            "SQL_LINK_SERVICE_INTERVAL_SECONDS": "90",
            "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "360",
            "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "1500",
            "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "70000",
            "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "42000",
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "900",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "300",
            "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "600",
            "OPS_WATCHDOG_REFRESH_MAX_AGE_SECONDS": "3600",
            "COINBASE_SNAPSHOT_MAX_WORKERS": "2",
            "COINBASE_CACHE_MAX_ENTRIES": "96",
            "TRADE_BEHAVIOR_BATCH_SIZE": "512",
            "ASYNC_PIPELINE_WORKERS": "3",
            "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "96",
            "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "24",
            "RUNTIME_TRAIN_BATCH_SIZE_CAP": "48",
            "RUNTIME_TRAIN_MAX_SAMPLES": "9000",
            "SQLITE_TEMP_STORE_MODE": "FILE",
            "SQLITE_CACHE_SIZE_KB": "10240",
            "SQLITE_MMAP_SIZE_MB": "96",
            "SQLITE_ANALYZE_ENABLED": "0",
            "BOT_OPS_SQLITE_TEMP_STORE_MODE": "FILE",
            "BOT_OPS_SQLITE_CACHE_SIZE_KB": "3072",
            "BOT_OPS_SQLITE_MMAP_SIZE_MB": "24",
            "TOP_BOT_PAPER_TRADING_TOP_N": "3",
            "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "1",
        },
    },
    "dual_pro": {
        "__default__": {
            "SQL_LINK_SERVICE_INTERVAL_SECONDS": "150",
            "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "600",
            "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "2400",
            "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "40000",
            "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "24000",
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "1800",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "600",
            "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "1200",
            "OPS_WATCHDOG_REFRESH_MAX_AGE_SECONDS": "7200",
            "COINBASE_SNAPSHOT_MAX_WORKERS": "1",
            "COINBASE_CACHE_MAX_ENTRIES": "48",
            "TRADE_BEHAVIOR_BATCH_SIZE": "256",
            "ASYNC_PIPELINE_WORKERS": "1",
            "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "32",
            "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "8",
            "RUNTIME_TRAIN_BATCH_SIZE_CAP": "24",
            "RUNTIME_TRAIN_MAX_SAMPLES": "4000",
            "SQLITE_TEMP_STORE_MODE": "FILE",
            "SQLITE_CACHE_SIZE_KB": "4096",
            "SQLITE_MMAP_SIZE_MB": "24",
            "SQLITE_ANALYZE_ENABLED": "0",
            "BOT_OPS_SQLITE_TEMP_STORE_MODE": "FILE",
            "BOT_OPS_SQLITE_CACHE_SIZE_KB": "1024",
            "BOT_OPS_SQLITE_MMAP_SIZE_MB": "8",
            "TOP_BOT_PAPER_TRADING_TOP_N": "1",
            "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "0",
        },
    },
}


CREATIVE_SESSION_OVERLAYS.update(
    {
        "logic_pro": {
            "__default__": {
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "120",
                "SQL_LINK_SERVICE_JSON_FILE_SYNC_MIN_INTERVAL_SECONDS": "600",
                "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "600",
                "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "2400",
                "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "40000",
                "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "24000",
                "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "1500",
                "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "600",
                "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "1200",
                "COINBASE_SNAPSHOT_MAX_WORKERS": "1",
                "ASYNC_PIPELINE_WORKERS": "1",
                "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "32",
                "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "8",
                "RUNTIME_TRAIN_BATCH_SIZE_CAP": "24",
                "RUNTIME_TRAIN_MAX_SAMPLES": "4000",
                "SQLITE_TEMP_STORE_MODE": "FILE",
                "SQLITE_CACHE_SIZE_KB": "4096",
                "SQLITE_MMAP_SIZE_MB": "24",
                "BOT_OPS_SQLITE_TEMP_STORE_MODE": "FILE",
                "BOT_OPS_SQLITE_CACHE_SIZE_KB": "1024",
                "BOT_OPS_SQLITE_MMAP_SIZE_MB": "8",
                "TOP_BOT_PAPER_TRADING_TOP_N": "1",
                "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "0",
                "CREATIVE_AUDIO_PRIORITY": "1",
                "LOGIC_PRO_AUDIO_PRIORITY": "1",
            }
        },
        "logic_pro_hot": {
            "__default__": {
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "180",
                "SQL_LINK_SERVICE_JSON_FILE_SYNC_MIN_INTERVAL_SECONDS": "900",
                "SQL_LINK_SERVICE_AUTO_HOT_RETENTION": "0",
                "SQL_LINK_SERVICE_AUTO_QUEUE_RETENTION": "0",
                "SQL_LINK_SERVICE_AUTO_LOCAL_FALLBACK_PRUNE": "0",
                "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "1800",
                "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "900",
                "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "1800",
                "COINBASE_SNAPSHOT_MAX_WORKERS": "1",
                "ASYNC_PIPELINE_WORKERS": "1",
                "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "24",
                "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "6",
                "RUNTIME_TRAIN_BATCH_SIZE_CAP": "16",
                "RUNTIME_TRAIN_MAX_SAMPLES": "2500",
                "TOP_BOT_PAPER_TRADING_TOP_N": "1",
                "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "0",
                "CREATIVE_AUDIO_PRIORITY": "1",
                "LOGIC_PRO_AUDIO_PRIORITY": "1",
            }
        },
        "final_cut_pro": {
            "__default__": {
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "150",
                "SQL_LINK_SERVICE_JSON_FILE_SYNC_MIN_INTERVAL_SECONDS": "900",
                "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "720",
                "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "3000",
                "SQL_LINK_SERVICE_HOT_BATCH_SIZE": "36000",
                "SQL_LINK_SERVICE_QUEUE_BATCH_SIZE": "22000",
                "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "1800",
                "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "720",
                "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "1500",
                "COINBASE_SNAPSHOT_MAX_WORKERS": "1",
                "ASYNC_PIPELINE_WORKERS": "1",
                "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "28",
                "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "8",
                "RUNTIME_TRAIN_BATCH_SIZE_CAP": "24",
                "RUNTIME_TRAIN_MAX_SAMPLES": "3500",
                "SQLITE_TEMP_STORE_MODE": "FILE",
                "SQLITE_CACHE_SIZE_KB": "4096",
                "SQLITE_MMAP_SIZE_MB": "16",
                "BOT_OPS_SQLITE_TEMP_STORE_MODE": "FILE",
                "BOT_OPS_SQLITE_CACHE_SIZE_KB": "1024",
                "BOT_OPS_SQLITE_MMAP_SIZE_MB": "8",
                "TOP_BOT_PAPER_TRADING_TOP_N": "1",
                "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "0",
                "CREATIVE_MEDIA_PRIORITY": "1",
                "FINAL_CUT_MEDIA_PRIORITY": "1",
            }
        },
        "final_cut_pro_hot": {
            "__default__": {
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "240",
                "SQL_LINK_SERVICE_JSON_FILE_SYNC_MIN_INTERVAL_SECONDS": "1200",
                "SQL_LINK_SERVICE_AUTO_HOT_RETENTION": "0",
                "SQL_LINK_SERVICE_AUTO_QUEUE_RETENTION": "0",
                "SQL_LINK_SERVICE_AUTO_LOCAL_FALLBACK_PRUNE": "0",
                "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "2400",
                "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "1200",
                "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "2400",
                "COINBASE_SNAPSHOT_MAX_WORKERS": "1",
                "ASYNC_PIPELINE_WORKERS": "1",
                "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "20",
                "RUNTIME_SLOW_BOT_CACHE_MAX_SYMBOLS": "6",
                "RUNTIME_TRAIN_BATCH_SIZE_CAP": "16",
                "RUNTIME_TRAIN_MAX_SAMPLES": "2500",
                "TOP_BOT_PAPER_TRADING_TOP_N": "1",
                "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "0",
                "CREATIVE_MEDIA_PRIORITY": "1",
                "FINAL_CUT_MEDIA_PRIORITY": "1",
            }
        },
        "cooldown": {
            "__default__": {
                "SQL_LINK_SERVICE_INTERVAL_SECONDS": "150",
                "SQL_LINK_SERVICE_JSON_FILE_SYNC_MIN_INTERVAL_SECONDS": "900",
                "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "900",
                "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "2400",
                "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "1500",
                "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "720",
                "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "1500",
                "COINBASE_SNAPSHOT_MAX_WORKERS": "1",
                "ASYNC_PIPELINE_WORKERS": "1",
                "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "32",
                "RUNTIME_TRAIN_BATCH_SIZE_CAP": "24",
                "RUNTIME_TRAIN_MAX_SAMPLES": "4000",
                "TOP_BOT_PAPER_TRADING_TOP_N": "1",
                "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "0",
            }
        },
    }
)


CO_RUNNING_SESSION_OVERLAYS: dict[str, dict[str, dict[str, str]]] = {
    "light_competition": {
        "__default__": {
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "420",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "210",
            "COINBASE_SNAPSHOT_MAX_WORKERS": "3",
            "ASYNC_PIPELINE_WORKERS": "4",
            "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "128",
        },
        "portable_throughput": {
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "300",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "180",
            "COINBASE_SNAPSHOT_MAX_WORKERS": "3",
            "ASYNC_PIPELINE_WORKERS": "4",
            "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "128",
        },
    },
    "interactive": {
        "__default__": {
            "SQL_LINK_SERVICE_INTERVAL_SECONDS": "90",
            "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "300",
            "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "1200",
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "600",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "240",
            "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "480",
            "COINBASE_SNAPSHOT_MAX_WORKERS": "2",
            "ASYNC_PIPELINE_WORKERS": "3",
            "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "96",
            "RUNTIME_TRAIN_BATCH_SIZE_CAP": "64",
            "RUNTIME_TRAIN_MAX_SAMPLES": "12000",
            "TOP_BOT_PAPER_TRADING_TOP_N": "3",
            "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "1",
        },
        "max_throughput": {
            "SQL_LINK_SERVICE_INTERVAL_SECONDS": "75",
            "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "240",
            "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "1050",
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "420",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "180",
            "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "420",
            "COINBASE_SNAPSHOT_MAX_WORKERS": "3",
            "ASYNC_PIPELINE_WORKERS": "4",
            "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "128",
            "RUNTIME_TRAIN_BATCH_SIZE_CAP": "72",
            "RUNTIME_TRAIN_MAX_SAMPLES": "14000",
            "TOP_BOT_PAPER_TRADING_TOP_N": "4",
            "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "1",
        },
    },
    "heavy_competition": {
        "__default__": {
            "SQL_LINK_SERVICE_INTERVAL_SECONDS": "120",
            "SQL_LINK_SERVICE_HOT_MIN_INTERVAL_SECONDS": "480",
            "SQL_LINK_SERVICE_QUEUE_MIN_INTERVAL_SECONDS": "1800",
            "ONE_NUMBERS_REFRESH_INTERVAL_SECONDS": "1200",
            "INGESTION_BACKPRESSURE_REFRESH_INTERVAL_SECONDS": "420",
            "DATA_SOURCE_DIVERGENCE_REFRESH_INTERVAL_SECONDS": "900",
            "COINBASE_SNAPSHOT_MAX_WORKERS": "1",
            "ASYNC_PIPELINE_WORKERS": "2",
            "RUNTIME_FEATURE_CACHE_MAX_ENTRIES": "48",
            "RUNTIME_TRAIN_BATCH_SIZE_CAP": "32",
            "RUNTIME_TRAIN_MAX_SAMPLES": "6000",
            "TOP_BOT_PAPER_TRADING_TOP_N": "2",
            "TOP_BOT_PAPER_TRADING_OPTIONS_TOP_N": "0",
        }
    },
}

QUANT_MODEL_CAPS_BY_PROFILE: dict[str, dict[str, str]] = {
    "constrained": {
        "BOT_MLX_OPTIONAL": "1",
        "MLX_METAL_JIT": "0",
        "QUANT_MODEL_MLX_COMPILE_ENABLED": "0",
        "QUANT_MODEL_LAZY_LIBRARY_IMPORTS": "1",
        "QUANT_MODEL_RESEARCH_ONLY": "1",
        "QUANT_MODEL_MAX_WORKERS": "1",
        "QUANT_MODEL_MONTE_CARLO_PATHS": "192",
        "QUANT_MODEL_QUASI_MONTE_CARLO_PATHS": "160",
        "QUANT_MODEL_LATIN_HYPERCUBE_PATHS": "160",
        "QUANT_MODEL_FINITE_DIFF_GRID": "40",
        "QUANT_MODEL_FFT_GRID": "256",
        "QUANT_MODEL_TRINOMIAL_STEPS": "32",
        "QUANT_MODEL_PARTICLE_COUNT": "96",
        "QUANT_MODEL_GA_POPULATION": "8",
        "QUANT_MODEL_GA_GENERATIONS": "3",
        "QUANT_MODEL_ACTOR_CRITIC_ROLLOUTS": "32",
        "QUANT_MODEL_GRAPH_NODE_CAP": "12",
        "QUANT_MODEL_MICROSTRUCTURE_LOOKBACK": "120",
        "QUANT_MODEL_REGIME_FILTER_STATES": "2",
        "QUANT_MODEL_GPU_MONTE_CARLO_PATHS": "8192",
        "QUANT_MODEL_GPU_KALMAN_WINDOW": "64",
        "QUANT_MODEL_NEURAL_SDE_STEPS": "12",
        "QUANT_MODEL_SIGNATURE_DEPTH": "2",
        "QUANT_MODEL_HAWKES_WINDOWS": "12",
        "QUANT_MODEL_TRANSFORMER_SEQUENCE": "32",
        "QUANT_MODEL_LAPLACIAN_NODE_CAP": "12",
        "QUANT_MODEL_CRITIC_REPLAY_CAP": "64",
        "QUANT_MODEL_NHHMM_STATES": "2",
        "QUANT_MODEL_PIN_SDE_STEPS": "12",
        "QUANT_MODEL_DML_CROSSFIT_FOLDS": "2",
        "QUANT_MODEL_CROSS_MODAL_EMBED_DIM": "64",
        "QUANT_MODEL_RLBF_BACKTRACK_CAP": "4",
        "QUANT_MODEL_DMS_STEPS": "12",
        "QUANT_MODEL_EQUIVARIANT_CHANNELS": "8",
        "QUANT_MODEL_DAINN_LAYERS": "2",
        "QUANT_MODEL_MARKOV_EXEC_STATES": "3",
        "QUANT_MODEL_DIFF_BACKTEST_STEPS": "12",
        "QUANT_MODEL_DURABILITY_SCENARIOS": "4",
        "QUANT_MODEL_INFO_GEOMETRY_DIM": "4",
        "QUANT_MODEL_GAT_HEADS": "2",
        "QUANT_MODEL_WALLET_INTENT_CAP": "4",
        "QUANT_MODEL_SIGNATURE_KERNEL_DEPTH": "2",
        "QUANT_MODEL_HYBRID_OPT_ITERATIONS": "6",
        "QUANT_MODEL_FORMAL_CHECKS": "6",
    },
    "air_safe": {
        "BOT_MLX_OPTIONAL": "1",
        "MLX_METAL_JIT": "0",
        "QUANT_MODEL_MLX_COMPILE_ENABLED": "0",
        "QUANT_MODEL_LAZY_LIBRARY_IMPORTS": "1",
        "QUANT_MODEL_RESEARCH_ONLY": "1",
        "QUANT_MODEL_MAX_WORKERS": "1",
        "QUANT_MODEL_MONTE_CARLO_PATHS": "384",
        "QUANT_MODEL_QUASI_MONTE_CARLO_PATHS": "256",
        "QUANT_MODEL_LATIN_HYPERCUBE_PATHS": "256",
        "QUANT_MODEL_FINITE_DIFF_GRID": "56",
        "QUANT_MODEL_FFT_GRID": "384",
        "QUANT_MODEL_TRINOMIAL_STEPS": "48",
        "QUANT_MODEL_PARTICLE_COUNT": "128",
        "QUANT_MODEL_GA_POPULATION": "10",
        "QUANT_MODEL_GA_GENERATIONS": "4",
        "QUANT_MODEL_ACTOR_CRITIC_ROLLOUTS": "48",
        "QUANT_MODEL_GRAPH_NODE_CAP": "18",
        "QUANT_MODEL_MICROSTRUCTURE_LOOKBACK": "180",
        "QUANT_MODEL_REGIME_FILTER_STATES": "2",
        "QUANT_MODEL_GPU_MONTE_CARLO_PATHS": "32768",
        "QUANT_MODEL_GPU_KALMAN_WINDOW": "128",
        "QUANT_MODEL_NEURAL_SDE_STEPS": "24",
        "QUANT_MODEL_SIGNATURE_DEPTH": "2",
        "QUANT_MODEL_HAWKES_WINDOWS": "24",
        "QUANT_MODEL_TRANSFORMER_SEQUENCE": "48",
        "QUANT_MODEL_LAPLACIAN_NODE_CAP": "18",
        "QUANT_MODEL_CRITIC_REPLAY_CAP": "96",
        "QUANT_MODEL_NHHMM_STATES": "2",
        "QUANT_MODEL_PIN_SDE_STEPS": "18",
        "QUANT_MODEL_DML_CROSSFIT_FOLDS": "2",
        "QUANT_MODEL_CROSS_MODAL_EMBED_DIM": "96",
        "QUANT_MODEL_RLBF_BACKTRACK_CAP": "6",
        "QUANT_MODEL_DMS_STEPS": "18",
        "QUANT_MODEL_EQUIVARIANT_CHANNELS": "12",
        "QUANT_MODEL_DAINN_LAYERS": "3",
        "QUANT_MODEL_MARKOV_EXEC_STATES": "4",
        "QUANT_MODEL_DIFF_BACKTEST_STEPS": "24",
        "QUANT_MODEL_DURABILITY_SCENARIOS": "8",
        "QUANT_MODEL_INFO_GEOMETRY_DIM": "6",
        "QUANT_MODEL_GAT_HEADS": "3",
        "QUANT_MODEL_WALLET_INTENT_CAP": "6",
        "QUANT_MODEL_SIGNATURE_KERNEL_DEPTH": "2",
        "QUANT_MODEL_HYBRID_OPT_ITERATIONS": "10",
        "QUANT_MODEL_FORMAL_CHECKS": "10",
    },
    "pro_balanced": {
        "BOT_MLX_OPTIONAL": "1",
        "MLX_METAL_JIT": "0",
        "QUANT_MODEL_MLX_COMPILE_ENABLED": "0",
        "QUANT_MODEL_LAZY_LIBRARY_IMPORTS": "1",
        "QUANT_MODEL_RESEARCH_ONLY": "1",
        "QUANT_MODEL_MAX_WORKERS": "2",
        "QUANT_MODEL_MONTE_CARLO_PATHS": "768",
        "QUANT_MODEL_QUASI_MONTE_CARLO_PATHS": "512",
        "QUANT_MODEL_LATIN_HYPERCUBE_PATHS": "512",
        "QUANT_MODEL_FINITE_DIFF_GRID": "72",
        "QUANT_MODEL_FFT_GRID": "768",
        "QUANT_MODEL_TRINOMIAL_STEPS": "72",
        "QUANT_MODEL_PARTICLE_COUNT": "192",
        "QUANT_MODEL_GA_POPULATION": "14",
        "QUANT_MODEL_GA_GENERATIONS": "5",
        "QUANT_MODEL_ACTOR_CRITIC_ROLLOUTS": "80",
        "QUANT_MODEL_GRAPH_NODE_CAP": "24",
        "QUANT_MODEL_MICROSTRUCTURE_LOOKBACK": "240",
        "QUANT_MODEL_REGIME_FILTER_STATES": "3",
        "QUANT_MODEL_GPU_MONTE_CARLO_PATHS": "250000",
        "QUANT_MODEL_GPU_KALMAN_WINDOW": "256",
        "QUANT_MODEL_NEURAL_SDE_STEPS": "48",
        "QUANT_MODEL_SIGNATURE_DEPTH": "3",
        "QUANT_MODEL_HAWKES_WINDOWS": "48",
        "QUANT_MODEL_TRANSFORMER_SEQUENCE": "80",
        "QUANT_MODEL_LAPLACIAN_NODE_CAP": "32",
        "QUANT_MODEL_CRITIC_REPLAY_CAP": "160",
        "QUANT_MODEL_NHHMM_STATES": "3",
        "QUANT_MODEL_PIN_SDE_STEPS": "36",
        "QUANT_MODEL_DML_CROSSFIT_FOLDS": "3",
        "QUANT_MODEL_CROSS_MODAL_EMBED_DIM": "192",
        "QUANT_MODEL_RLBF_BACKTRACK_CAP": "10",
        "QUANT_MODEL_DMS_STEPS": "36",
        "QUANT_MODEL_EQUIVARIANT_CHANNELS": "24",
        "QUANT_MODEL_DAINN_LAYERS": "4",
        "QUANT_MODEL_MARKOV_EXEC_STATES": "6",
        "QUANT_MODEL_DIFF_BACKTEST_STEPS": "48",
        "QUANT_MODEL_DURABILITY_SCENARIOS": "12",
        "QUANT_MODEL_INFO_GEOMETRY_DIM": "10",
        "QUANT_MODEL_GAT_HEADS": "4",
        "QUANT_MODEL_WALLET_INTENT_CAP": "10",
        "QUANT_MODEL_SIGNATURE_KERNEL_DEPTH": "3",
        "QUANT_MODEL_HYBRID_OPT_ITERATIONS": "18",
        "QUANT_MODEL_FORMAL_CHECKS": "16",
    },
    "max_throughput": {
        "BOT_MLX_OPTIONAL": "1",
        "MLX_METAL_JIT": "0",
        "QUANT_MODEL_MLX_COMPILE_ENABLED": "0",
        "QUANT_MODEL_LAZY_LIBRARY_IMPORTS": "1",
        "QUANT_MODEL_RESEARCH_ONLY": "1",
        "QUANT_MODEL_MAX_WORKERS": "3",
        "QUANT_MODEL_MONTE_CARLO_PATHS": "1024",
        "QUANT_MODEL_QUASI_MONTE_CARLO_PATHS": "768",
        "QUANT_MODEL_LATIN_HYPERCUBE_PATHS": "768",
        "QUANT_MODEL_FINITE_DIFF_GRID": "96",
        "QUANT_MODEL_FFT_GRID": "1024",
        "QUANT_MODEL_TRINOMIAL_STEPS": "96",
        "QUANT_MODEL_PARTICLE_COUNT": "256",
        "QUANT_MODEL_GA_POPULATION": "16",
        "QUANT_MODEL_GA_GENERATIONS": "6",
        "QUANT_MODEL_ACTOR_CRITIC_ROLLOUTS": "128",
        "QUANT_MODEL_GRAPH_NODE_CAP": "36",
        "QUANT_MODEL_MICROSTRUCTURE_LOOKBACK": "360",
        "QUANT_MODEL_REGIME_FILTER_STATES": "3",
        "QUANT_MODEL_GPU_MONTE_CARLO_PATHS": "1000000",
        "QUANT_MODEL_GPU_KALMAN_WINDOW": "512",
        "QUANT_MODEL_NEURAL_SDE_STEPS": "96",
        "QUANT_MODEL_SIGNATURE_DEPTH": "3",
        "QUANT_MODEL_HAWKES_WINDOWS": "96",
        "QUANT_MODEL_TRANSFORMER_SEQUENCE": "128",
        "QUANT_MODEL_LAPLACIAN_NODE_CAP": "48",
        "QUANT_MODEL_CRITIC_REPLAY_CAP": "256",
        "QUANT_MODEL_NHHMM_STATES": "4",
        "QUANT_MODEL_PIN_SDE_STEPS": "72",
        "QUANT_MODEL_DML_CROSSFIT_FOLDS": "4",
        "QUANT_MODEL_CROSS_MODAL_EMBED_DIM": "256",
        "QUANT_MODEL_RLBF_BACKTRACK_CAP": "16",
        "QUANT_MODEL_DMS_STEPS": "72",
        "QUANT_MODEL_EQUIVARIANT_CHANNELS": "32",
        "QUANT_MODEL_DAINN_LAYERS": "6",
        "QUANT_MODEL_MARKOV_EXEC_STATES": "10",
        "QUANT_MODEL_DIFF_BACKTEST_STEPS": "96",
        "QUANT_MODEL_DURABILITY_SCENARIOS": "24",
        "QUANT_MODEL_INFO_GEOMETRY_DIM": "16",
        "QUANT_MODEL_GAT_HEADS": "6",
        "QUANT_MODEL_WALLET_INTENT_CAP": "16",
        "QUANT_MODEL_SIGNATURE_KERNEL_DEPTH": "4",
        "QUANT_MODEL_HYBRID_OPT_ITERATIONS": "32",
        "QUANT_MODEL_FORMAL_CHECKS": "24",
    },
}

EXPANSION_PRESSURE_OVERLAYS: dict[str, dict[str, str]] = {
    "large": {
        "SLEEVE_MASTER_ROLLUP_ENABLED": "1",
        "GRAND_MASTER_READS_SLEEVE_ROLLUPS": "1",
        "COLLECTION_ONLY_DIRECT_TO_GRAND_MASTER": "0",
        "SPECIALIZED_SLEEVE_INTERVAL": "150",
        "SLEEVE_WORKERS_SPECIALIZED": "1",
        "SLEEVE_NICE_SPECIALIZED": "14",
        "SLOW_BOT_EVERY_N_ITERS": "2",
        "BOT_COOLDOWN_MIN_ITERS": "2",
        "BOT_COOLDOWN_MAX_ITERS": "5",
        "ALL_SLEEVES_MAX_RESTARTS_PER_HOUR": "30",
        "ALL_SLEEVES_RESTART_DELAY": "4",
        "LOG_SUB_BOT_DECISIONS": "1",
        "LOG_MASTER_VARIANT_DECISIONS": "1",
        "LOG_GRAND_MASTER_DECISIONS": "1",
        "LOG_OPTIONS_MASTER_DECISIONS": "1",
        "LOG_FUTURES_MASTER_DECISIONS": "1",
    },
    "massive": {
        "SLEEVE_MASTER_ROLLUP_ENABLED": "1",
        "GRAND_MASTER_READS_SLEEVE_ROLLUPS": "1",
        "COLLECTION_ONLY_DIRECT_TO_GRAND_MASTER": "0",
        "SPECIALIZED_SLEEVE_INTERVAL": "210",
        "SLEEVE_WORKERS_SPECIALIZED": "1",
        "SLEEVE_NICE_SPECIALIZED": "16",
        "SLOW_BOT_EVERY_N_ITERS": "3",
        "BOT_COOLDOWN_MIN_ITERS": "3",
        "BOT_COOLDOWN_MAX_ITERS": "7",
        "ALL_SLEEVES_MAX_RESTARTS_PER_HOUR": "22",
        "ALL_SLEEVES_RESTART_DELAY": "6",
        "ALL_SLEEVES_BREAKER_STARTUP_GRACE_SECONDS": "420",
        "ALL_SLEEVES_BREAKER_DATA_QUALITY_GRACE_SECONDS": "1200",
        "LOG_SUB_BOT_DECISIONS": "1",
        "LOG_MASTER_VARIANT_DECISIONS": "1",
        "LOG_GRAND_MASTER_DECISIONS": "1",
        "LOG_OPTIONS_MASTER_DECISIONS": "1",
        "LOG_FUTURES_MASTER_DECISIONS": "1",
    },
}


def _creative_overlay(level: str, base_tier: str) -> dict[str, str]:
    overlays = CREATIVE_SESSION_OVERLAYS.get(str(level or "").strip().lower())
    if not isinstance(overlays, dict):
        return {}
    tier_key = str(base_tier or "").strip()
    selected = overlays.get(tier_key) if isinstance(overlays.get(tier_key), dict) else overlays.get("__default__")
    return dict(selected or {})


def _creative_session_key(creative_session: dict[str, Any]) -> str:
    kind = str(creative_session.get("kind") or creative_session.get("creative_session_kind") or "").strip().lower()
    level = str(creative_session.get("level") or "").strip().lower()
    if kind and kind != "none" and kind in CREATIVE_SESSION_OVERLAYS:
        return kind
    return level


def _creative_session_overlay(creative_session: dict[str, Any], base_tier: str) -> dict[str, str]:
    key = _creative_session_key(creative_session)
    overlay = _creative_overlay(key, base_tier)
    if overlay:
        return overlay
    return _creative_overlay(str(creative_session.get("level") or ""), base_tier)


def _creative_pause_overrides(creative_session: dict[str, Any]) -> dict[str, str]:
    key = _creative_session_key(creative_session)
    level = str(creative_session.get("level") or "none").strip().lower()
    active = bool(creative_session.get("active", False)) or level in {"cooldown", "active", "hot", "dual_pro"}
    if not active:
        return {
            "CREATIVE_MODE_ACTIVE": "0",
            "CREATIVE_HEAVY_RESEARCH_PAUSED": "0",
            "CREATIVE_MODE_STATE": "none",
        }
    hard_pause = "1" if level in {"active", "hot", "dual_pro", "cooldown"} else "0"
    return {
        "CREATIVE_MODE_ACTIVE": "1",
        "CREATIVE_MODE_STATE": key or level or "active",
        "CREATIVE_MODE_LEVEL": level or key or "active",
        "CREATIVE_MODE_APPS": ",".join(str(app) for app in creative_session.get("apps", []) if str(app).strip()),
        "CREATIVE_HEAVY_RESEARCH_PAUSED": hard_pause,
        "RESOURCE_GUARD_OPTIONAL_BLOCK_ON_CREATIVE_SESSION_LEVELS": "active,dual_pro,hot,cooldown",
        "RESOURCE_GUARD_REFRESH_BLOCK_ON_CREATIVE_SESSION_LEVELS": "dual_pro,hot,cooldown",
        "TRAINING_RUNTIME_PAUSED_FOR_CREATIVE": hard_pause,
        "AUTO_RETRAIN_PAUSED_FOR_CREATIVE": hard_pause,
        "QUANT_RESEARCH_PAUSED_FOR_CREATIVE": hard_pause,
        "MLX_RESEARCH_PAUSED_FOR_CREATIVE": hard_pause,
        "REPORT_BUILD_PAUSED_FOR_CREATIVE": hard_pause,
        "RETENTION_MAINTENANCE_PAUSED_FOR_CREATIVE": hard_pause,
        "SQL_LINK_SERVICE_AUTO_HOT_RETENTION": "0",
        "SQL_LINK_SERVICE_AUTO_QUEUE_RETENTION": "0",
        "SQL_LINK_SERVICE_AUTO_LOCAL_FALLBACK_PRUNE": "0",
        "BOT_MLX_OPTIONAL": "1",
        "MLX_METAL_JIT": "0",
        "QUANT_MODEL_MLX_COMPILE_ENABLED": "0",
        "QUANT_MODEL_RESEARCH_ONLY": "1",
        "QUANT_MODEL_MAX_WORKERS": "1",
    }


def _co_running_overlay(level: str, base_tier: str) -> dict[str, str]:
    overlays = CO_RUNNING_SESSION_OVERLAYS.get(str(level or "").strip().lower())
    if not isinstance(overlays, dict):
        return {}
    tier_key = str(base_tier or "").strip()
    selected = overlays.get(tier_key) if isinstance(overlays.get(tier_key), dict) else overlays.get("__default__")
    return dict(selected or {})


def _ordered_unique(items: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        item = str(raw or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _memory_pressure_clear(resource_guard: dict[str, Any]) -> bool:
    state = str(resource_guard.get("memory_pressure_state") or "").strip().lower()
    kind = str(resource_guard.get("memory_pressure_kind") or "").strip().lower()
    swap_used_gb = _safe_float(resource_guard.get("swap_used_gb"), 0.0)
    return state in {"", "green", "normal", "ok", "none"} and kind in {"", "none", "green", "ok"} and swap_used_gb < 8.0


def _storage_pressure_clear(ingestion_storage: dict[str, Any]) -> bool:
    severity = str(ingestion_storage.get("severity") or "").strip().lower()
    pressure_index = _safe_float(ingestion_storage.get("pressure_index"), 0.0)
    return severity in {"", "ready", "stable", "low"} and pressure_index <= 0.25


def _cotenant_awareness(
    *,
    original_status: str,
    reasons: list[str],
    recommended_profile: str,
    base_tier: str,
    resource_guard: dict[str, Any],
    ingestion_storage: dict[str, Any],
    creative_session: dict[str, Any],
    co_running_session: dict[str, Any],
) -> dict[str, Any]:
    co_level = str(co_running_session.get("level") or "none").strip().lower()
    creative_level = str(creative_session.get("level") or "none").strip().lower()
    co_active = bool(co_running_session.get("active", False)) or co_level not in {"", "none"}
    creative_active = bool(creative_session.get("active", False)) or creative_level not in {"", "none"}
    memory_clear = _memory_pressure_clear(resource_guard)
    storage_clear = _storage_pressure_clear(ingestion_storage)
    soft_cotenant_only = bool(reasons) and all(
        reason in {"co_running_light_competition", "co_running_interactive"}
        for reason in reasons
    )

    adjusted_status = str(original_status or "ready")
    mode = "system_only"
    status_adjusted = False
    if co_active and soft_cotenant_only and memory_clear and storage_clear:
        adjusted_status = "ready"
        mode = "managed_cotenant"
        status_adjusted = adjusted_status != original_status
    elif co_active and co_level == "heavy_competition" and memory_clear and storage_clear:
        mode = "guarded_cotenant"
    elif creative_active:
        mode = "creative_cotenant"
    elif co_active:
        mode = "pressure_aware_cotenant"

    open_apps = _ordered_unique(
        list(co_running_session.get("apps") or []) + list(creative_session.get("apps") or [])
    )
    return {
        "aware": True,
        "active": co_active or creative_active,
        "mode": mode,
        "status_adjusted": status_adjusted,
        "original_status": str(original_status or ""),
        "overall_status": adjusted_status,
        "memory_pressure_clear": memory_clear,
        "storage_pressure_clear": storage_clear,
        "recommended_profile": recommended_profile,
        "base_tier": base_tier,
        "profile_capped": _profile_order(recommended_profile) < _profile_order(base_tier),
        "co_running_level": co_level,
        "creative_level": creative_level,
        "open_app_count": len(open_apps),
        "open_apps": open_apps,
        "co_running_classes": list(co_running_session.get("classes") or []),
        "policy": (
            "open_apps_managed_without_health_degradation"
            if status_adjusted
            else "open_apps_profile_caps_remain_guarded_until_pressure_clears"
        ),
    }


def _cotenant_awareness_overrides(awareness: dict[str, Any]) -> dict[str, str]:
    return {
        "MEMORY_GUARD_COTENANT_AWARE": "1",
        "MEMORY_GUARD_COTENANT_ACTIVE": "1" if bool(awareness.get("active", False)) else "0",
        "MEMORY_GUARD_COTENANT_MODE": str(awareness.get("mode") or "system_only"),
        "MEMORY_GUARD_COTENANT_LEVEL": str(awareness.get("co_running_level") or "none"),
        "MEMORY_GUARD_CREATIVE_LEVEL": str(awareness.get("creative_level") or "none"),
        "MEMORY_GUARD_OPEN_APP_COUNT": str(int(awareness.get("open_app_count") or 0)),
        "MEMORY_GUARD_OPEN_APPS": ",".join(str(app) for app in awareness.get("open_apps", []) if str(app).strip()),
        "MEMORY_GUARD_COTENANT_CLASSES": ",".join(
            str(name) for name in awareness.get("co_running_classes", []) if str(name).strip()
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _registry_rows(project_root: Path) -> list[dict[str, Any]]:
    registry = _load_json(project_root / "master_bot_registry.json")
    rows = registry.get("sub_bots") if isinstance(registry.get("sub_bots"), list) else []
    return [row for row in rows if isinstance(row, dict)]


def _expansion_session_summary(project_root: Path) -> dict[str, Any]:
    rows = _registry_rows(project_root)
    total = len(rows)
    active = sum(1 for row in rows if bool(row.get("active", False)))
    data_collection_active = sum(1 for row in rows if bool(row.get("data_collection_active", False)))
    collection_only = sum(
        1
        for row in rows
        if str(row.get("lifecycle_state") or "").strip().lower() == "data_collection_only"
    )
    sleeve_profiles = sorted(
        {
            str(row.get("sleeve_profile") or row.get("slot_kind") or "unknown").strip().lower()
            for row in rows
            if bool(row.get("active", False)) and str(row.get("sleeve_profile") or row.get("slot_kind") or "").strip()
        }
    )
    if data_collection_active >= 500 or active >= 550 or len(sleeve_profiles) >= 60:
        pressure_level = "massive"
    elif data_collection_active >= 250 or active >= 350 or len(sleeve_profiles) >= 32:
        pressure_level = "large"
    elif data_collection_active >= 120 or active >= 180 or len(sleeve_profiles) >= 18:
        pressure_level = "moderate"
    else:
        pressure_level = "normal"
    return {
        "registry_path": str(project_root / "master_bot_registry.json"),
        "total_bots": total,
        "active_bots": active,
        "data_collection_active_bots": data_collection_active,
        "collection_only_bots": collection_only,
        "sleeve_profile_count": len(sleeve_profiles),
        "sample_sleeve_profiles": sleeve_profiles[:20],
        "pressure_level": pressure_level,
    }


def _expansion_pressure_overlay(summary: dict[str, Any]) -> dict[str, str]:
    level = str(summary.get("pressure_level") or "normal").strip().lower()
    if level == "massive":
        return dict(EXPANSION_PRESSURE_OVERLAYS["massive"])
    if level == "large":
        return dict(EXPANSION_PRESSURE_OVERLAYS["large"])
    return {
        "SLEEVE_MASTER_ROLLUP_ENABLED": "1",
        "GRAND_MASTER_READS_SLEEVE_ROLLUPS": "1",
        "COLLECTION_ONLY_DIRECT_TO_GRAND_MASTER": "0",
    }


def _profile_order(name: str) -> int:
    return {"constrained": 0, "air_safe": 1, "pro_balanced": 2, "max_throughput": 3}.get(str(name or ""), 1)


def _cap_profile(current: str, maximum_profile: str) -> str:
    current_name = str(current or "air_safe")
    maximum_name = str(maximum_profile or "air_safe")
    if _profile_order(current_name) > _profile_order(maximum_name):
        return maximum_name
    return current_name


def _base_tier(apple_profile: dict[str, Any]) -> str:
    tier = str(apple_profile.get("applied_tier") or apple_profile.get("detected_tier") or "").strip()
    if tier in {"air_safe", "pro_balanced", "max_throughput"}:
        return tier
    memory_gb = _safe_float((apple_profile.get("hardware") or {}).get("memory_gb"), 0.0)
    if memory_gb >= 48.0:
        return "max_throughput"
    if memory_gb >= 24.0:
        return "pro_balanced"
    return "air_safe"


def _creative_session_summary(resource_guard: dict[str, Any]) -> dict[str, Any]:
    active_apps = resource_guard.get("creative_apps") if isinstance(resource_guard.get("creative_apps"), list) else []
    app_names = sorted({str(item).strip() for item in active_apps if str(item).strip()})
    app_count = int(_safe_float(resource_guard.get("creative_app_count"), float(len(app_names))))
    level = str(resource_guard.get("creative_session_level") or "").strip().lower()
    kind = str(resource_guard.get("creative_session_kind") or "").strip().lower()
    if app_count <= 0 and app_names:
        app_count = len(app_names)
    if app_count <= 0 and level and level != "none":
        app_count = 1
    if not level:
        level = "active" if app_count > 0 else "none"
    if not kind:
        lowered_names = {name.lower() for name in app_names}
        if level == "dual_pro" or {"logic pro", "final cut pro"}.issubset(lowered_names):
            kind = "dual_pro"
        elif "logic pro" in lowered_names:
            kind = "logic_pro_hot" if level == "hot" else "logic_pro"
        elif "final cut pro" in lowered_names:
            kind = "final_cut_pro_hot" if level == "hot" else "final_cut_pro"
        else:
            kind = level if level != "none" else "none"
    return {
        "active": app_count > 0,
        "app_count": app_count,
        "apps": app_names,
        "level": level,
        "kind": kind,
        "cooldown_active": bool(resource_guard.get("creative_cooldown_active", False)),
        "cooldown_remaining_seconds": _safe_float(resource_guard.get("creative_cooldown_remaining_seconds"), 0.0),
        "editing_app_cpu_sum": _safe_float(resource_guard.get("editing_app_cpu_sum"), 0.0),
    }


def _co_running_session_summary(resource_guard: dict[str, Any]) -> dict[str, Any]:
    active_classes = resource_guard.get("co_running_classes") if isinstance(resource_guard.get("co_running_classes"), list) else []
    active_apps = resource_guard.get("co_running_apps") if isinstance(resource_guard.get("co_running_apps"), list) else []
    level = str(resource_guard.get("co_running_session_level") or "").strip().lower()
    class_cpu = resource_guard.get("co_running_class_cpu") if isinstance(resource_guard.get("co_running_class_cpu"), dict) else {}
    app_count = len([str(item).strip() for item in active_apps if str(item).strip()])
    class_count = len([str(item).strip() for item in active_classes if str(item).strip()])
    if not level:
        level = "light_competition" if class_count > 0 else "none"
    return {
        "active": class_count > 0,
        "class_count": class_count,
        "classes": sorted({str(item).strip() for item in active_classes if str(item).strip()}),
        "app_count": app_count,
        "apps": sorted({str(item).strip() for item in active_apps if str(item).strip()}),
        "level": level,
        "cpu_sum": _safe_float(resource_guard.get("co_running_cpu_sum"), 0.0),
        "class_cpu": {str(key): _safe_float(value, 0.0) for key, value in sorted(class_cpu.items())},
    }


def _preferred_env_value(name: str, default: str, env_overrides: dict[str, str] | None = None) -> str:
    value = os.getenv(name, "")
    if str(value).strip():
        return str(value).strip()
    if isinstance(env_overrides, dict):
        candidate = env_overrides.get(name)
        if str(candidate or "").strip():
            return str(candidate).strip()
    return str(default)


def _storage_drain_active(ingestion_storage: dict[str, Any]) -> bool:
    storage = ingestion_storage.get("storage") if isinstance(ingestion_storage.get("storage"), dict) else {}
    backpressure = ingestion_storage.get("backpressure") if isinstance(ingestion_storage.get("backpressure"), dict) else {}
    backlog_drain_status = str(storage.get("backlog_drain_status") or "").strip().lower()
    recommended_mode = str(ingestion_storage.get("recommended_operating_mode") or "").strip().lower()
    total_pending = _safe_int(backpressure.get("total_pending_lines"), _safe_int(backpressure.get("core_pending_lines"), 0))
    return bool(
        backlog_drain_status in {"drain_active", "handoff_requested"}
        or recommended_mode == "maintenance_drain_window"
        or total_pending > 0
    )


def _sql_writer_coordination(backpressure_fleet: dict[str, Any], ingestion_storage: dict[str, Any]) -> dict[str, Any]:
    storage_backpressure = ingestion_storage.get("backpressure") if isinstance(ingestion_storage.get("backpressure"), dict) else {}
    active_drainer = backpressure_fleet.get("active_drainer") if isinstance(backpressure_fleet.get("active_drainer"), dict) else {}
    concentration = active_drainer.get("concentration") if isinstance(active_drainer.get("concentration"), dict) else {}
    request = backpressure_fleet.get("service_request") if isinstance(backpressure_fleet.get("service_request"), dict) else {}
    env = request.get("env_overrides") if isinstance(request.get("env_overrides"), dict) else {}
    total_pending = _safe_int(concentration.get("total_pending_lines"), _safe_int(storage_backpressure.get("total_pending_lines"), 0))
    top1_share = _safe_float(concentration.get("top1_share"), 0.0)
    top3_share = _safe_float(concentration.get("top3_share"), 0.0)
    concentrated = bool(concentration.get("concentrated", False)) or str(env.get("SQL_LINK_SERVICE_CONCENTRATED_CORE_DRAIN") or "").strip() == "1"
    if not concentrated and total_pending >= 5000 and (top1_share >= 0.45 or top3_share >= 0.75):
        concentrated = True
    overrides = CONCENTRATED_DRAIN_SQL_OVERRIDES if concentrated else DRAIN_FRIENDLY_SQL_OVERRIDES
    return {
        "source": "backpressure_drainer_fleet" if backpressure_fleet else "storage_backpressure",
        "active_drainer": str(active_drainer.get("name") or ""),
        "concentrated_core_drain": concentrated,
        "total_pending_lines": total_pending,
        "top1_share": round(top1_share, 6),
        "top3_share": round(top3_share, 6),
        "recommended_merge_max_seconds_per_cycle": _safe_int(overrides.get("SQL_LINK_SERVICE_MERGE_MAX_SECONDS_PER_CYCLE"), 25),
        "recommended_shard_link_timeout_seconds": _safe_int(overrides.get("SQL_LINK_SERVICE_SHARD_LINK_TIMEOUT_SECONDS"), 0),
        "recommended_aggressive_trading_max_lines_per_file": _safe_int(
            overrides.get("SQL_LINK_SERVICE_SHARD_AGGRESSIVE_TRADING_MAX_LINES_PER_FILE"),
            0,
        ),
    }


def _drain_friendly_sql_overrides(coordination: dict[str, Any]) -> dict[str, str]:
    if bool(coordination.get("concentrated_core_drain", False)):
        return dict(CONCENTRATED_DRAIN_SQL_OVERRIDES)
    return dict(DRAIN_FRIENDLY_SQL_OVERRIDES)


def _recommended_profile(
    base_tier: str,
    resource_guard: dict[str, Any],
    ingestion_storage: dict[str, Any],
    *,
    env_overrides: dict[str, str] | None = None,
) -> tuple[str, list[str], str, dict[str, Any], dict[str, str], dict[str, Any]]:
    memory_state = str(resource_guard.get("memory_pressure_state") or "").strip().lower()
    memory_kind = str(resource_guard.get("memory_pressure_kind") or "").strip().lower()
    swap_used_gb = _safe_float(resource_guard.get("swap_used_gb"), 0.0)
    compressed_store_gb = _safe_float(resource_guard.get("compressed_store_gb"), 0.0)
    compressor_gb = _safe_float(resource_guard.get("compressor_gb"), 0.0)
    storage_severity = str(ingestion_storage.get("severity") or "").strip().lower()
    reasons: list[str] = []
    recommended = base_tier
    status = "ready"
    creative_session = _creative_session_summary(resource_guard)
    creative_overlay: dict[str, str] = {}
    co_running_session = _co_running_session_summary(resource_guard)
    co_running_overlay: dict[str, str] = {}

    if memory_state == "red" or memory_kind in {"throttled", "red"}:
        recommended = "constrained"
        status = "blocked"
        reasons.append("memory_pressure_red")
    elif memory_state == "yellow" or memory_kind.startswith("swap_only"):
        recommended = "air_safe" if _profile_order(base_tier) >= _profile_order("air_safe") else base_tier
        status = "needs_work"
        reasons.append("memory_pressure_elevated")

    if storage_severity == "critical":
        recommended = "constrained"
        status = "blocked"
        reasons.append("storage_pressure_critical")
    elif storage_severity == "high" and recommended == "max_throughput":
        recommended = "pro_balanced"
        status = "needs_work"
        reasons.append("storage_pressure_high")

    if swap_used_gb >= 18.0 and recommended == "max_throughput":
        recommended = "pro_balanced"
        status = "needs_work"
        reasons.append("swap_usage_high")
    elif swap_used_gb >= 24.0:
        recommended = "constrained"
        status = "blocked"
        reasons.append("swap_usage_critical")

    if compressed_store_gb >= 28.0 or compressor_gb >= 16.0:
        recommended = _cap_profile(recommended, "constrained")
        status = "blocked" if status == "blocked" else "needs_work"
        reasons.append("compressed_memory_critical")
    elif compressed_store_gb >= 18.0 or compressor_gb >= 9.0:
        recommended = _cap_profile(recommended, "air_safe")
        if status == "ready":
            status = "needs_work"
        reasons.append("compressed_memory_high")

    creative_level = str(creative_session.get("level") or "none")
    creative_key = _creative_session_key(creative_session)
    if creative_level == "dual_pro":
        recommended = _cap_profile(
            recommended,
            _preferred_env_value("MEMORY_EFFICIENCY_CREATIVE_DUAL_PROFILE", "constrained", env_overrides),
        )
        if status != "blocked":
            status = "needs_work"
        reasons.append(f"creative_session_{creative_key or 'dual_pro'}")
        creative_overlay = _creative_session_overlay(creative_session, base_tier)
    elif creative_level == "hot":
        default_hot_profile = "constrained" if creative_key in {"logic_pro_hot", "final_cut_pro_hot"} else "air_safe"
        recommended = _cap_profile(
            recommended,
            _preferred_env_value("MEMORY_EFFICIENCY_CREATIVE_HOT_PROFILE", default_hot_profile, env_overrides),
        )
        if status != "blocked":
            status = "needs_work"
        reasons.append(f"creative_session_{creative_key or 'hot'}")
        creative_overlay = _creative_session_overlay(creative_session, base_tier)
    elif creative_level == "active":
        active_profile_env = "MEMORY_EFFICIENCY_CREATIVE_ACTIVE_MAX_PROFILE"
        active_profile_default = "air_safe"
        if creative_key == "logic_pro":
            active_profile_env = "MEMORY_EFFICIENCY_CREATIVE_LOGIC_PROFILE"
            active_profile_default = "pro_balanced"
        elif creative_key == "final_cut_pro":
            active_profile_env = "MEMORY_EFFICIENCY_CREATIVE_FINAL_CUT_PROFILE"
            active_profile_default = _preferred_env_value(
                "MEMORY_EFFICIENCY_CREATIVE_ACTIVE_MAX_PROFILE",
                "air_safe",
                env_overrides,
            )
        recommended = _cap_profile(
            recommended,
            _preferred_env_value(active_profile_env, active_profile_default, env_overrides),
        )
        if status == "ready":
            status = "needs_work"
        reasons.append(f"creative_session_{creative_key or 'active'}")
        creative_overlay = _creative_session_overlay(creative_session, base_tier)
    elif creative_level == "cooldown":
        recommended = _cap_profile(
            recommended,
            _preferred_env_value("MEMORY_EFFICIENCY_CREATIVE_COOLDOWN_PROFILE", "air_safe", env_overrides),
        )
        if status == "ready":
            status = "needs_work"
        reasons.append("creative_session_cooldown")
        creative_overlay = _creative_session_overlay(creative_session, base_tier)

    co_running_level = str(co_running_session.get("level") or "none")
    if co_running_level == "heavy_competition":
        recommended = _cap_profile(
            recommended,
            _preferred_env_value("MEMORY_EFFICIENCY_COTENANT_HEAVY_PROFILE", "constrained", env_overrides),
        )
        status = "blocked" if status == "blocked" else "needs_work"
        reasons.append("co_running_heavy_competition")
        co_running_overlay = _co_running_overlay("heavy_competition", base_tier)
    elif co_running_level == "interactive":
        recommended = _cap_profile(
            recommended,
            _preferred_env_value("MEMORY_EFFICIENCY_COTENANT_INTERACTIVE_MAX_PROFILE", "pro_balanced", env_overrides),
        )
        if status == "ready":
            status = "needs_work"
        reasons.append("co_running_interactive")
        co_running_overlay = _co_running_overlay("interactive", base_tier)
    elif co_running_level == "light_competition":
        if status == "ready":
            status = "needs_work"
        reasons.append("co_running_light_competition")
        co_running_overlay = _co_running_overlay("light_competition", base_tier)

    if not reasons:
        reasons.append("memory_headroom_ok")
    return recommended, reasons, status, creative_session, {**creative_overlay, **co_running_overlay}, co_running_session


def _override_lines(profile_name: str, env_overrides: dict[str, str]) -> list[str]:
    def _shell_assignment(name: str, value: str) -> str:
        return f"{name}={shlex.quote(str(value))}"

    lines = [
        "# Auto-managed by scripts/ops/memory_efficiency_control.py",
        _shell_assignment("BOT_MEMORY_EFFICIENCY_PROFILE", profile_name),
    ]
    for key, value in sorted(env_overrides.items()):
        lines.append(_shell_assignment(key, value))
    return lines


def _write_override(path: Path, profile_name: str, env_overrides: dict[str, str]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(_override_lines(profile_name, env_overrides)) + "\n"
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if current == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def build_payload(project_root: Path = PROJECT_ROOT, *, action: str, override_path: Path, changed: bool = False) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    health_root = project_root / "governance" / "health"
    resource_guard = _load_json(health_root / "resource_guard_latest.json")
    apple_profile = _load_json(health_root / "apple_silicon_profile_latest.json")
    ingestion_storage = _load_json(health_root / "ingestion_storage_control_latest.json")
    backpressure_fleet = _load_json(health_root / "backpressure_drainer_fleet_latest.json")
    expansion_session = _expansion_session_summary(project_root)
    expansion_overlay = _expansion_pressure_overlay(expansion_session)

    base_tier = _base_tier(apple_profile)
    base_env = apple_profile.get("env_overrides") if isinstance(apple_profile.get("env_overrides"), dict) else {}
    if not base_env:
        base_env = FALLBACK_PRESETS.get(base_tier, FALLBACK_PRESETS["air_safe"])
    recommended_profile, reasons, overall_status, creative_session, coexistence_overlay, co_running_session = _recommended_profile(
        base_tier,
        resource_guard,
        ingestion_storage,
        env_overrides=base_env,
    )
    cotenant_awareness = _cotenant_awareness(
        original_status=overall_status,
        reasons=reasons,
        recommended_profile=recommended_profile,
        base_tier=base_tier,
        resource_guard=resource_guard,
        ingestion_storage=ingestion_storage,
        creative_session=creative_session,
        co_running_session=co_running_session,
    )
    overall_status = str(cotenant_awareness.get("overall_status") or overall_status)
    recommended_env = {
        **base_env,
        **FALLBACK_PRESETS.get(recommended_profile, {}),
        **QUANT_MODEL_CAPS_BY_PROFILE.get(recommended_profile, {}),
        **coexistence_overlay,
        **_cotenant_awareness_overrides(cotenant_awareness),
        **_creative_pause_overrides(creative_session),
        **expansion_overlay,
    }
    drain_friendly_sql_active = _storage_drain_active(ingestion_storage)
    sql_writer_coordination = _sql_writer_coordination(backpressure_fleet, ingestion_storage)
    if drain_friendly_sql_active:
        recommended_env.update(_drain_friendly_sql_overrides(sql_writer_coordination))
    hardware = apple_profile.get("hardware") if isinstance(apple_profile.get("hardware"), dict) else {}
    unified_memory = apple_profile.get("unified_memory_telemetry") if isinstance(apple_profile.get("unified_memory_telemetry"), dict) else {}
    memory_gb = _safe_float(hardware.get("memory_gb"), 0.0)
    swap_used_gb = _safe_float(resource_guard.get("swap_used_gb"), 0.0)
    shared_pool = bool(unified_memory.get("shared_cpu_gpu_memory_pool", bool(hardware.get("is_apple_silicon", False)) or base_tier != "generic"))
    competitive_state = "strong"
    if swap_used_gb >= max(memory_gb * 0.2, 8.0):
        competitive_state = "eroding_under_swap"
    elif str(resource_guard.get("memory_pressure_state") or "").strip().lower() in {"yellow", "red"}:
        competitive_state = "constrained"

    return {
        "timestamp_utc": now.isoformat(),
        "schema_version": 1,
        "ok": overall_status == "ready",
        "overall_status": overall_status,
        "action": action,
        "base_tier": base_tier,
        "recommended_profile": recommended_profile,
        "changed": bool(changed),
        "override_path": str(override_path),
        "override_exists": bool(override_path.exists()),
        "reasons": reasons,
        "creative_session": creative_session,
        "co_running_session": co_running_session,
        "cotenant_awareness": cotenant_awareness,
        "expansion_session": expansion_session,
        "unified_memory_telemetry": {
            "memory_architecture": str(unified_memory.get("memory_architecture") or ("unified" if shared_pool else "system_memory")),
            "shared_cpu_gpu_memory_pool": shared_pool,
            "estimated_feature_cache_budget_gb": _safe_float(unified_memory.get("estimated_feature_cache_budget_gb"), round(memory_gb * 0.1, 3)),
            "estimated_live_inference_budget_gb": _safe_float(unified_memory.get("estimated_live_inference_budget_gb"), round(memory_gb * 0.06, 3)),
            "competitive_advantage_state": competitive_state,
            "copy_pressure_summary": str(unified_memory.get("copy_avoidance_summary") or ""),
        },
        "memory_snapshot": {
            "memory_pressure_state": str(resource_guard.get("memory_pressure_state") or ""),
            "memory_pressure_kind": str(resource_guard.get("memory_pressure_kind") or ""),
            "memory_free_pct": _safe_float(resource_guard.get("memory_free_pct"), 0.0),
            "swap_used_gb": _safe_float(resource_guard.get("swap_used_gb"), 0.0),
            "compressed_store_gb": _safe_float(resource_guard.get("compressed_store_gb"), 0.0),
            "compressor_gb": _safe_float(resource_guard.get("compressor_gb"), 0.0),
        },
        "storage_snapshot": {
            "severity": str(ingestion_storage.get("severity") or ""),
            "pressure_index": _safe_float(ingestion_storage.get("pressure_index"), 0.0),
            "estimated_core_drain_minutes": ((ingestion_storage.get("backpressure") or {}).get("estimated_core_drain_minutes") if isinstance(ingestion_storage.get("backpressure"), dict) else None),
            "drain_friendly_sql_active": drain_friendly_sql_active,
            "sql_writer_coordination": sql_writer_coordination,
            "backlog_drain_status": str(((ingestion_storage.get("storage") or {}).get("backlog_drain_status")) if isinstance(ingestion_storage.get("storage"), dict) else ""),
            "recommended_operating_mode": str(ingestion_storage.get("recommended_operating_mode") or ""),
        },
        "recommended_env_overrides": recommended_env,
        "quant_model_caps": QUANT_MODEL_CAPS_BY_PROFILE.get(recommended_profile, {}),
        "expansion_pressure_overrides": expansion_overlay,
        "recommendations": [
            "Reduce snapshot workers, cache sizes, and training batch size when swap grows faster than free memory recovers.",
            "Prefer air-safe or constrained profiles while storage pressure is high so compression, backlog, and long-lived caches do not cascade together.",
            "When Final Cut Pro or Logic Pro are active, slow refresh cadence and shrink background worker pools before memory pressure turns red.",
            "When browsers, IDEs, Docker, or virtualization tools are chewing CPU, trim fanout and refresh cadence so the live stack stays responsive instead of competing head-on.",
            "Use pressure-aware SQLite tuning so maintenance and ops-control queries shift temp storage to disk and shrink cache/mmap before macOS swap spikes.",
            "Cap Monte Carlo, adaptive architecture, graph, particle, and regime-filter work through QUANT_MODEL_* overrides before live collection fanout increases.",
            "Use sleeve-master rollups so the Grand Master reads compressed sleeve health instead of raw votes from every collection-only expansion bot.",
            "Keep MLX extension libraries optional and lazy-loaded; load VLM, graph, audio, SNN, and large quant kernels only inside sleeves that need them.",
        ],
        "source_files": {
            "resource_guard": str(health_root / "resource_guard_latest.json"),
            "apple_silicon_profile": str(health_root / "apple_silicon_profile_latest.json"),
            "ingestion_storage_control": str(health_root / "ingestion_storage_control_latest.json"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recommend or apply a memory-efficiency override tuned for Apple Silicon and live pressure.")
    parser.add_argument("action", choices=("status", "apply"))
    parser.add_argument("--override-file", default=str(DEFAULT_OVERRIDE))
    parser.add_argument("--out-file", default=str(DEFAULT_OUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    override_path = Path(args.override_file).expanduser()
    payload = build_payload(PROJECT_ROOT, action=args.action, override_path=override_path, changed=False)

    changed = False
    if args.action == "apply":
        changed = _write_override(
            override_path,
            str(payload.get("recommended_profile") or "air_safe"),
            payload.get("recommended_env_overrides") if isinstance(payload.get("recommended_env_overrides"), dict) else {},
        )
        payload = build_payload(PROJECT_ROOT, action=args.action, override_path=override_path, changed=changed)

    out_path = Path(args.out_file).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=True))
    else:
        print(
            "memory_efficiency_control "
            f"status={payload['overall_status']} "
            f"profile={payload.get('recommended_profile', '')} "
            f"changed={int(bool(payload.get('changed', False)))}"
        )
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
