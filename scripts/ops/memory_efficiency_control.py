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
    },
    "air_safe": {
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
    },
    "pro_balanced": {
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
    },
    "max_throughput": {
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
    },
}


def _creative_overlay(level: str, base_tier: str) -> dict[str, str]:
    overlays = CREATIVE_SESSION_OVERLAYS.get(str(level or "").strip().lower())
    if not isinstance(overlays, dict):
        return {}
    tier_key = str(base_tier or "").strip()
    selected = overlays.get(tier_key) if isinstance(overlays.get(tier_key), dict) else overlays.get("__default__")
    return dict(selected or {})


def _co_running_overlay(level: str, base_tier: str) -> dict[str, str]:
    overlays = CO_RUNNING_SESSION_OVERLAYS.get(str(level or "").strip().lower())
    if not isinstance(overlays, dict):
        return {}
    tier_key = str(base_tier or "").strip()
    selected = overlays.get(tier_key) if isinstance(overlays.get(tier_key), dict) else overlays.get("__default__")
    return dict(selected or {})


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
    if app_count <= 0 and app_names:
        app_count = len(app_names)
    if app_count <= 0 and level and level != "none":
        app_count = 1
    if not level:
        level = "active" if app_count > 0 else "none"
    return {
        "active": app_count > 0,
        "app_count": app_count,
        "apps": app_names,
        "level": level,
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

    creative_level = str(creative_session.get("level") or "none")
    if creative_level == "dual_pro":
        recommended = _cap_profile(
            recommended,
            _preferred_env_value("MEMORY_EFFICIENCY_CREATIVE_DUAL_PROFILE", "constrained", env_overrides),
        )
        if status != "blocked":
            status = "needs_work"
        reasons.append("creative_session_dual_pro")
        creative_overlay = _creative_overlay("dual_pro", base_tier)
    elif creative_level == "hot":
        recommended = _cap_profile(
            recommended,
            _preferred_env_value("MEMORY_EFFICIENCY_CREATIVE_HOT_PROFILE", "air_safe", env_overrides),
        )
        if status != "blocked":
            status = "needs_work"
        reasons.append("creative_session_hot")
        creative_overlay = _creative_overlay("hot", base_tier)
    elif creative_level == "active":
        recommended = _cap_profile(
            recommended,
            _preferred_env_value("MEMORY_EFFICIENCY_CREATIVE_ACTIVE_MAX_PROFILE", "air_safe", env_overrides),
        )
        if status == "ready":
            status = "needs_work"
        reasons.append("creative_session_active")
        creative_overlay = _creative_overlay("active", base_tier)

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
    recommended_env = {
        **base_env,
        **FALLBACK_PRESETS.get(recommended_profile, {}),
        **QUANT_MODEL_CAPS_BY_PROFILE.get(recommended_profile, {}),
        **coexistence_overlay,
    }
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
        },
        "recommended_env_overrides": recommended_env,
        "quant_model_caps": QUANT_MODEL_CAPS_BY_PROFILE.get(recommended_profile, {}),
        "recommendations": [
            "Reduce snapshot workers, cache sizes, and training batch size when swap grows faster than free memory recovers.",
            "Prefer air-safe or constrained profiles while storage pressure is high so compression, backlog, and long-lived caches do not cascade together.",
            "When Final Cut Pro or Logic Pro are active, slow refresh cadence and shrink background worker pools before memory pressure turns red.",
            "When browsers, IDEs, Docker, or virtualization tools are chewing CPU, trim fanout and refresh cadence so the live stack stays responsive instead of competing head-on.",
            "Use pressure-aware SQLite tuning so maintenance and ops-control queries shift temp storage to disk and shrink cache/mmap before macOS swap spikes.",
            "Cap Monte Carlo, adaptive architecture, graph, particle, and regime-filter work through QUANT_MODEL_* overrides before live collection fanout increases.",
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
