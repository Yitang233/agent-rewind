"""Helpers shared by the analysis scripts (imported by analyze.py, analyze_v2.py, make_report_figures.py)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def rec_llm(r):
    """The recovery phase's LLM totals of one cell row."""
    return r["recovery"]["llm"]
