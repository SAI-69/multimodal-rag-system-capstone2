import os
import re
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from typing import List, Dict

# ============================
# CONFIG
# ============================

load_dotenv(override=True)
POSTGRES_DSN = os.getenv("PG_CONNECTION_STRING_AGENT")

# ============================
# NORMALIZATION HELPERS
# ============================

def sanitize_llm_sql(sql: str) -> str:
    """
    Remove markdown, comments, trailing semicolons.
    """
    if not sql or not isinstance(sql, str):
        raise ValueError("Empty SQL")

    sql = re.sub(r"```sql|```", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    sql = sql.strip()

    if sql.endswith(";"):
        sql = sql[:-1]

    return sql


def normalize_string_comparisons(sql: str) -> str:
    """
    account_id = 12345  →  account_id = '12345'
    """
    pattern = re.compile(
        r"\b(account_id|card_id|loan_id)\s*=\s*(\d+)\b",
        re.IGNORECASE,
    )
    return pattern.sub(r"\1 = '\2'", sql)


def normalize_column_names(sql: str) -> str:
    """
    Fix common hallucinated column names.
    """
    replacements = {
        "transaction_date": "txn_date",
        "transaction_type": "txn_type",
        "transaction_amount": "amount",
    }

    for wrong, correct in replacements.items():
        sql = re.sub(rf"\b{wrong}\b", correct, sql, flags=re.IGNORECASE)

    return sql


# ============================
# SECURITY VALIDATION
# ============================

FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|"
    r"grant|revoke|commit|rollback|merge)\b",
    re.IGNORECASE,
)

def _validate_readonly_sql(sql: str) -> None:
    """
    FINAL validator:
    ✅ Allows SELECT / WITH / JOIN / GROUP BY / CTEs
    ❌ Blocks ALL write & admin operations
    """
    sql_clean = sql.strip().lower()

    if not (sql_clean.startswith("select") or sql_clean.startswith("with")):
        raise ValueError("Unsafe SQL detected: not SELECT/WITH")

    if FORBIDDEN_KEYWORDS.search(sql_clean):
        raise ValueError("Unsafe SQL detected: write/admin keyword found")


# ============================
# EXECUTOR
# ============================

# def execute_readonly_sql(sql: str, max_rows: int = 100) -> List[Dict]:
#     if not POSTGRES_DSN:
#         raise RuntimeError("PG_CONNECTION_STRING is not set")

#     # ✅ Normalize first
#     sql = sanitize_llm_sql(sql)
#     sql = normalize_string_comparisons(sql)
#     sql = normalize_column_names(sql)

#     # ✅ Validate final SQL
#     _validate_readonly_sql(sql)

#     # ✅ Robust LIMIT handling
#     has_limit = re.search(r"\blimit\s+\d+\b", sql, re.IGNORECASE) is not None

#     sql_final = sql.strip()
#     if not has_limit:
#         sql_final = f"{sql_final} LIMIT {max_rows}"

#     try:
#         conn = psycopg2.connect(
#             POSTGRES_DSN,
#             cursor_factory=RealDictCursor,
#         )
#         conn.set_session(readonly=True, autocommit=True)

#         with conn.cursor() as cursor:
#             cursor.execute(sql_final)
#             rows = cursor.fetchall()

#         return rows

#     except psycopg2.Error as e:
#         raise RuntimeError(f"Database error: {str(e)}") from e

#     finally:
#         if conn:
#             conn.close() 

def execute_readonly_sql(sql: str, max_rows: int = 100) -> List[Dict]:
    if not POSTGRES_DSN:
        raise RuntimeError("PG_CONNECTION_STRING is not set")

    # Initialize conn so 'finally' can see it even if connection fails
    conn = None 

    # ✅ Normalize and Validate...
    sql = sanitize_llm_sql(sql)
    sql = normalize_string_comparisons(sql)
    sql = normalize_column_names(sql)
    _validate_readonly_sql(sql)

    # ✅ Handle Limit...
    has_limit = re.search(r"\blimit\s+\d+\b", sql, re.IGNORECASE) is not None
    sql_final = sql.strip() if has_limit else f"{sql.strip()} LIMIT {max_rows}"

    try:
        conn = psycopg2.connect(
            POSTGRES_DSN,
            cursor_factory=RealDictCursor,
        )
        conn.set_session(readonly=True, autocommit=True)

        with conn.cursor() as cursor:
            cursor.execute(sql_final)
            return cursor.fetchall()

    except psycopg2.Error as e:
        raise RuntimeError(f"Database error: {str(e)}") from e

    finally:
        # Now 'conn' is guaranteed to exist as either None or a connection object
        if conn is not None:
            conn.close()