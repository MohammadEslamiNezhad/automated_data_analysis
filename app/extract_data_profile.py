"""
Dataset Profiler
----------------
Reads a tabular dataframe and produces a compact, LLM-ready profile summary.

Design notes:
- Thin class: each method is independent, takes/returns explicit data,
  no hidden state coupling between methods (unlike the original script,
  where `create_quality_report` silently depended on `duplicated` computed
  inside `create_summary`).
- Every "get_*" method returns a plain dict/JSON-serializable structure.
- `build_profile()` composes everything.
- `to_llm_summary()` produces a compact version (rounded numbers, top-N only)
  meant to be sent as a prompt to an LLM via OpenRouter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
import os 

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype, is_string_dtype


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class ProfilerConfig:
    high_cardinality_threshold_pct: float = 90.0
    categorical_unique_ratio_pct: float = 1.0   # unique/total <= this % => categorical
    outlier_iqr_multiplier: float = 1.5
    top_correlation_pairs: int = 10
    top_categorical_values: int = 5
    round_decimals: int = 3


# --------------------------------------------------------------------------- #
# Loading (kept separate from profiling so this class can later be fed by an API)
# --------------------------------------------------------------------------- #

def load_dataset(path: str, sheet_name: str = "Raw", header: int = 1) -> pd.DataFrame:
    if path.endswith(".xlsx"):
        return pd.read_excel(path, sheet_name=sheet_name, header=header)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type for: {path}")


# --------------------------------------------------------------------------- #
# Profiler
# --------------------------------------------------------------------------- #

class DatasetProfiler:
    def __init__(self, dataframe: pd.DataFrame, config: ProfilerConfig | None = None):
        self.df = dataframe.copy()
        self.config = config or ProfilerConfig()
        self._drop_all_nan_columns()

    # ---- cleanup -----------------------------------------------------------

    def _drop_all_nan_columns(self) -> None:
        """Drops 'Unnamed: n' columns that are entirely empty (common Excel artifact)."""
        unnamed_cols = self.df.columns[self.df.columns.astype(str).str.contains("Unnamed:")]
        empty_cols = [c for c in unnamed_cols if self.df[c].isnull().all()]
        self.df = self.df.drop(columns=empty_cols)

    # ---- column type splits (computed once, reused) -------------------------

    def _numeric_df(self) -> pd.DataFrame:
        numeric_cols = [c for c in self.df.columns if is_numeric_dtype(self.df[c])]
        return self.df[numeric_cols]

    def _datetime_df(self) -> pd.DataFrame:
        date_cols = [c for c in self.df.columns if is_datetime64_any_dtype(self.df[c])]
        return self.df[date_cols]

    def _categorical_df(self) -> pd.DataFrame:
        """
        Approximate categorical detection (v1, as requested):
        - numeric columns with a low unique/total ratio (e.g. encoded categories)
        - plus all object/string columns (not present in the original code,
          added so text categoricals aren't silently dropped)
        """
        threshold = self.config.categorical_unique_ratio_pct
        n = len(self.df) or 1

        def is_low_cardinality(col: pd.Series) -> bool:
            return (col.nunique() / n * 100) <= threshold

        numeric_df = self._numeric_df()
        numeric_categorical_cols = [c for c in numeric_df.columns if is_low_cardinality(numeric_df[c])]

        object_cols = [
            c for c in self.df.columns
            if is_string_dtype(self.df[c]) and not is_numeric_dtype(self.df[c])
        ]

        cols = list(dict.fromkeys(numeric_categorical_cols + object_cols))  # dedupe, keep order
        return self.df[cols]

    # ---- summaries -----------------------------------------------------------

    def get_basic_summary(self) -> dict[str, Any]:
        duplicated_mask = self.df.duplicated()
        return {
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1]),
            "duplicated_rows": int(duplicated_mask.sum()),
        }

    def get_quality_report(self) -> dict[str, Any]:
        n = len(self.df) or 1
        missing_counts = self.df.isnull().sum()
        constant_cols = [c for c in self.df.columns if self.df[c].nunique(dropna=False) == 1]
        unique_ratio_pct = self.df.nunique() / n * 100
        high_cardinality_cols = unique_ratio_pct[
            unique_ratio_pct > self.config.high_cardinality_threshold_pct
        ]

        return {
            "missing_counts": {k: int(v) for k, v in missing_counts.items() if v > 0},
            "missing_pct": {
                k: round(float(v) / n * 100, self.config.round_decimals)
                for k, v in missing_counts.items() if v > 0
            },
            "constant_columns": constant_cols,
            "high_cardinality_columns": {
                k: round(float(v), self.config.round_decimals)
                for k, v in high_cardinality_cols.items()
            },
        }

    def get_numeric_summary(self) -> dict[str, Any]:
        numeric_df = self._numeric_df()
        if numeric_df.empty:
            return {}

        describe = numeric_df.describe()
        iqr = describe.loc["75%"] - describe.loc["25%"]
        lower_bound = describe.loc["25%"] - self.config.outlier_iqr_multiplier * iqr
        upper_bound = describe.loc["75%"] + self.config.outlier_iqr_multiplier * iqr

        # Bug fix: an outlier is BELOW the lower bound OR ABOVE the upper bound,
        # not (greater than upper) AND (less than lower) -- that set is always empty.
        outlier_mask = (numeric_df > upper_bound) | (numeric_df < lower_bound)
        outlier_counts = outlier_mask.sum()

        r = self.config.round_decimals
        return {
            "describe": describe.round(r).to_dict(),
            "iqr": iqr.round(r).to_dict(),
            "lower_bound": lower_bound.round(r).to_dict(),
            "upper_bound": upper_bound.round(r).to_dict(),
            "skewness": numeric_df.skew().round(r).to_dict(),
            "outlier_counts": {k: int(v) for k, v in outlier_counts.items() if v > 0},
        }

    def get_categorical_summary(self) -> dict[str, Any]:
        categorical_df = self._categorical_df()
        if categorical_df.empty:
            return {}

        top_k = self.config.top_categorical_values
        summary: dict[str, Any] = {}
        for col in categorical_df.columns:
            value_counts = categorical_df[col].value_counts().head(top_k)
            summary[col] = {
                "unique_count": int(categorical_df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in value_counts.items()},
            }
        return summary

    def get_datetime_summary(self) -> dict[str, Any]:
        date_df = self._datetime_df()
        if date_df.empty:
            return {}

        summary: dict[str, Any] = {}
        for col in date_df.columns:
            summary[col] = {
                "min": str(date_df[col].min()),
                "max": str(date_df[col].max()),
                "missing": int(date_df[col].isnull().sum()),
            }
        return summary

    def get_correlation_summary(self) -> dict[str, Any]:
        numeric_df = self._numeric_df()
        if numeric_df.shape[1] < 2:
            return {}

        corr = numeric_df.corr()
        # Keep only the upper triangle (excluding the diagonal) so each pair
        # appears once and self-correlations are dropped.
        upper_triangle_mask = pd.DataFrame(
            [[i < j for j in range(len(corr))] for i in range(len(corr))],
            index=corr.index, columns=corr.columns,
        )
        pairs = corr.where(upper_triangle_mask)
        stacked = pairs.stack().dropna()
        top_pairs = stacked.reindex(stacked.abs().sort_values(ascending=False).index)
        top_pairs = top_pairs.head(self.config.top_correlation_pairs)

        return {
            f"{a} <-> {b}": round(float(v), self.config.round_decimals)
            for (a, b), v in top_pairs.items()
        }

    # ---- composition -----------------------------------------------------------

    def build_profile(self) -> dict[str, Any]:
        return {
            "basic_summary": self.get_basic_summary(),
            "quality_report": self.get_quality_report(),
            "numeric_summary": self.get_numeric_summary(),
            "categorical_summary": self.get_categorical_summary(),
            "datetime_summary": self.get_datetime_summary(),
            "top_correlations": self.get_correlation_summary(),
        }

    def to_llm_summary(self) -> str:
        """Compact JSON string, ready to embed in an LLM prompt."""
        profile = self.build_profile()
        return json.dumps(profile, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# OpenRouter client (thin wrapper, kept separate from profiling logic)
# --------------------------------------------------------------------------- #

class OpenRouterClient:
    """Minimal OpenRouter chat-completions client. Requires `requests`."""

    def __init__(self, api_key: str, model: str = "anthropic/claude-sonnet-4.5"):
        self.api_key = api_key
        self.model = model
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def summarize_dataset(self, llm_summary_json: str, extra_instructions: str = "") -> str:
        import requests  # local import so the module doesn't hard-require it

        system_prompt = (
            "You are a data analyst. You will receive a compact JSON profile of a "
            "tabular dataset (row/column counts, missing values, numeric stats, "
            "outlier counts, top categorical values, correlations). "
            "Summarize the dataset's key characteristics, data quality issues, "
            "and anything worth flagging before modeling. Be concise."
        )
        user_prompt = f"Dataset profile JSON:\n{llm_summary_json}\n\n{extra_instructions}"

        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]