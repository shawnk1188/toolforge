"""
ToolForge Data Pipeline — download, validate, split, and format training data.

Modules:
  schema.py     — Canonical data format (Pydantic models)
  download.py   — HuggingFace download + format conversion
  validate.py   — Data quality checks + deduplication
  prepare.py    — Stratified splitting + eval dataset generation
  formatter.py  — Llama 3.2 chat template formatting for SFT
"""
