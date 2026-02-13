"""
Spreadsheet query tool using Gemini-generated pandas code.

Executes data queries on CSV/XLSX files by:
1. Sending schema + question to Gemini to generate pandas code
2. Validating the generated code against a security blocklist
3. Running in a restricted sandbox with limited builtins
"""
import logging

import numpy as np
import pandas as pd
from google import genai
from google.genai import types

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Code generation prompt
CODE_GEN_PROMPT = """You are a pandas code generator. Given a DataFrame schema and a user question, write Python pandas code to answer the question.

Rules:
- The DataFrame is available as `df`
- Store your final result in a variable called `result`
- Do NOT import anything — pandas is available as `pd`, numpy as `np`
- Be concise — no comments, no print statements
- For aggregations, return scalar values or small DataFrames
- Handle missing values gracefully (use dropna() when needed)

DataFrame schema:
{schema}

User question: {question}

Write ONLY the Python code, nothing else:"""

# Security blocklist — patterns forbidden in generated code
BLOCKED_PATTERNS = [
    "import ", "__import__", "exec(", "eval(", "compile(",
    "open(", "os.", "sys.", "subprocess",
    ".to_csv(", ".to_excel(", ".to_json(", ".to_parquet(",
    ".to_sql(", ".to_pickle(", ".to_hdf(",
    "globals(", "locals(", "getattr(", "setattr(", "delattr(",
    "breakpoint(", "__builtins__",
]

MAX_CODE_LENGTH = 2000
MAX_RESULT_LENGTH = 5000

# Restricted builtins for sandbox
SAFE_BUILTINS = {
    "len": len,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "sorted": sorted,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "True": True,
    "False": False,
    "None": None,
    "isinstance": isinstance,
    "print": lambda *a, **k: None,  # no-op
}


def _build_schema_info(df: pd.DataFrame) -> str:
    """Build schema info string for the LLM prompt."""
    lines = []
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append(f"Columns and types:")
    for col in df.columns:
        lines.append(f"  - {col}: {df[col].dtype}")
    lines.append(f"\nSample data (first 3 rows):")
    lines.append(df.head(3).to_string())
    return "\n".join(lines)


def _validate_code(code: str) -> str | None:
    """Validate generated code. Returns error message if invalid, None if OK."""
    if len(code) > MAX_CODE_LENGTH:
        return f"Generated code too long ({len(code)} chars, max {MAX_CODE_LENGTH})"

    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            return f"Blocked pattern found in generated code: {pattern}"

    return None


def execute_query(df: pd.DataFrame, question: str) -> str:
    """
    Execute a pandas query on a DataFrame using Gemini-generated code.

    Args:
        df: The pandas DataFrame to query
        question: Natural language question about the data

    Returns:
        Formatted result string with the answer and generated code
    """
    schema = _build_schema_info(df)

    # Generate pandas code via Gemini
    try:
        client = genai.Client(api_key=settings.google_api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=CODE_GEN_PROMPT.format(schema=schema, question=question),
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=512,
            ),
        )
        code = response.text.strip()
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        return f"Error generating query code: {e}"

    # Strip markdown code fences if present
    if code.startswith("```"):
        lines = code.split("\n")
        # Remove first and last lines (```python and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        code = "\n".join(lines)

    # Validate
    error = _validate_code(code)
    if error:
        logger.warning(f"Code validation failed: {error}")
        return f"Query blocked for safety: {error}"

    # Execute in sandbox
    sandbox_globals = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "df": df,
    }

    try:
        exec(code, sandbox_globals)
        result = sandbox_globals.get("result", "No 'result' variable was set by the generated code.")
    except Exception as e:
        logger.error(f"Code execution failed: {e}\nCode:\n{code}")
        return f"Query execution error: {e}\n\nGenerated code:\n```python\n{code}\n```"

    # Format result
    result_str = str(result)
    if len(result_str) > MAX_RESULT_LENGTH:
        result_str = result_str[:MAX_RESULT_LENGTH] + "\n... (truncated)"

    return f"**Query Result:**\n{result_str}\n\n**Generated code:**\n```python\n{code}\n```"
