import os
import ast
from typing import TypedDict, List, Optional, cast

import cohere
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from src.api.v1.tools.fts_search import fts_search
from src.api.v1.tools.hybrid_search import hybrid_search
from src.api.v1.tools.vector_search import vector_search

from src.core.readonly_executor import execute_readonly_sql
from src.api.v1.schemas.query_schema import FeedBack

load_dotenv(override=True)

def extract_llm_text(response) -> str:
    if not response:
        return ""
    if hasattr(response, "content") and isinstance(response.content, list):
        first = response.content[0] if response.content else None
        if isinstance(first, dict) and "text" in first:
            return first["text"] or ""
    return ""

@tool
def fts_search_tool(query: str) -> list[Document]:
    """Performs full-text keyword search using PostgreSQL tsvector for exact matches."""
    return fts_search(query)


@tool
def vector_search_tool(query: str) -> list[Document]:
    """Performs pure semantic vector similarity search using pgvector embeddings."""
    return vector_search(query)


@tool
def hybrid_search_tool(query: str) -> list[Document]:
    """Combines vector semantic search + full-text keyword search using Reciprocal Rank Fusion."""
    return hybrid_search(query)


class RAGState(TypedDict):
    query: str
    route: str

    vector_query: Optional[str]
    sql_query: Optional[str]

    retrived_docs: List[Document]
    reranked_docs: List[Document]

    sql: Optional[str]
    sql_result: Optional[list]

    vector_response: Optional[str]
    sql_response: Optional[str]

    response: dict
    is_valid: str
    tool: str
    feedback: Optional[FeedBack]
    answer_attempts: int
    query_history: List[str]
    rewrite_count: int

    
def query_router_node(state: RAGState) -> RAGState:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    prompt = f"""
You are a query routing assistant for a banking system.

Classify the user query into exactly ONE of the following routes:

- vector → for conceptual questions such as definitions, explanations, policies, eligibility, or product details
- rdbms → for factual, numeric, or record-based questions such as balances, transactions, amounts, dates, or account-specific data
- hybrid → for questions that require BOTH explanation/context AND exact numbers or records

User Query: "{state['query']}"

Respond with ONLY ONE word (no extra text):
vector | rdbms | hybrid
"""

    route = llm.invoke(prompt).content[0]["text"].strip().lower()
    print(f"ROUTE: {route}")
    return {**state ,"route": route}

def query_splitter_node(state: RAGState) -> RAGState:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    prompt = f"""
Split the banking query into:
1) Vector (conceptual)
2) SQL (numeric / factual)

Query: "{state['query']}"

Reply ONLY as:
["vector query", "sql query"]
"""

    raw = llm.invoke(prompt).content[0]["text"].strip()
    try:
        vector_q, sql_q = ast.literal_eval(raw)
    except Exception:
        vector_q = state["query"]
        sql_q = state["query"]

    return {
        **state,
        "vector_query": vector_q,
        "sql_query": sql_q,
    }

def tool_call_agent_node(state: RAGState) -> RAGState:
    query = state.get("vector_query") or state["query"]

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    agent = llm.bind_tools(
        [vector_search_tool, fts_search_tool, hybrid_search_tool]
    )

    prompt = f"""
Choose ONE tool:
fts_search_tool | vector_search_tool | hybrid_search_tool

Query: {query}
"""

    response = agent.invoke(prompt)
    chosen = extract_llm_text(response).lower()

    if "fts" in chosen:
        docs = fts_search(query)
        tool = "fts"
    elif "vector" in chosen:
        docs = vector_search(query)
        tool = "vector"
    else:
        docs = hybrid_search(query)
        tool = "hybrid"

    return {"retrived_docs": docs, "tool": tool}

def rerank_node(state: RAGState) -> RAGState:
    docs = state["retrived_docs"]
    if not docs:
        return { "reranked_docs": []}

    try:
        co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        result = co.rerank(
            model="rerank-v3.5",
            query=state.get("vector_query") or state["query"],
            documents=[d.page_content for d in docs],
            top_n=5,
        )
        reranked = [docs[r.index] for r in result.results]
    except Exception:
        reranked = docs[:5]

    return {"reranked_docs": reranked}

def validate_node(state: RAGState) -> RAGState:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    context = "\n".join(d.page_content for d in state["reranked_docs"])
    prompt = f"""
Query: {state.get("vector_query") or state["query"]}

Context:
{context}

Answerable? Reply yes/no
"""

    ans = llm.invoke(prompt).content[0]["text"].lower()
    return { "is_valid": "yes" if ans.startswith("yes") else "no"}

def generate_answer_node(state: RAGState) -> RAGState:
    docs = state["reranked_docs"]
    if not docs:
        return { "vector_response": "No information found."}

    context = "\n".join(d.page_content for d in docs)

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    prompt = f"""
Context:
{context}

Question:
{state.get("vector_query") or state["query"]}
"""

    answer = llm.invoke(prompt).content[0]["text"].strip()
    return {"vector_response": answer}

def sql_planner_node(state: RAGState) -> RAGState:
    query = state.get("sql_query") or state["query"]

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    prompt = f"""
You are an expert PostgreSQL query generator for a banking system.

You MUST generate exactly ONE safe, read-only SQL query.

========================
DATABASE SCHEMA (STRICT)
========================

TABLE: accounts
- account_id (VARCHAR, PRIMARY KEY)
- customer_name (VARCHAR)
- account_type (VARCHAR: savings | current | salary)
- branch_code (VARCHAR)
- ifsc_code (VARCHAR)
- mobile (VARCHAR, masked)
- email (VARCHAR)
- kyc_status (VARCHAR)
- created_at (TIMESTAMP)

TABLE: transactions
- txn_id (UUID, PRIMARY KEY)
- account_id (VARCHAR, FK → accounts.account_id)
- txn_date (DATE)
- txn_type (VARCHAR: debit | credit)
- amount (NUMERIC)
- balance_after (NUMERIC)
- description (VARCHAR)
- channel (VARCHAR: ATM | UPI | NEFT | RTGS | IMPS | branch | online | POS)
- merchant_name (VARCHAR)
- category (VARCHAR)
- created_at (TIMESTAMP)

TABLE: loan_accounts
- loan_id (VARCHAR, PRIMARY KEY)
- account_id (VARCHAR, FK → accounts.account_id)
- loan_type (VARCHAR: home_loan | personal_loan | auto_loan | gold_loan)
- principal (NUMERIC)
- outstanding (NUMERIC)
- disbursed_date (DATE)
- emi_amount (NUMERIC)
- next_emi_date (DATE)
- interest_rate (NUMERIC)
- tenure_months (INT)
- emi_paid (INT)
- status (VARCHAR)
- created_at (TIMESTAMP)

TABLE: fixed_deposits
- fd_id (VARCHAR, PRIMARY KEY)
- account_id (VARCHAR, FK → accounts.account_id)
- principal (NUMERIC)
- interest_rate (NUMERIC)
- tenure_days (INT)
- start_date (DATE)
- maturity_date (DATE)
- maturity_amount (NUMERIC)
- interest_payout (VARCHAR)
- status (VARCHAR)
- created_at (TIMESTAMP)

TABLE: credit_cards
- card_id (VARCHAR, PRIMARY KEY)
- account_id (VARCHAR, FK → accounts.account_id)
- card_variant (VARCHAR)
- credit_limit (NUMERIC)
- available_limit (NUMERIC)
- outstanding_amt (NUMERIC)
- due_date (DATE)
- min_due (NUMERIC)
- status (VARCHAR)
- issued_date (DATE)
- created_at (TIMESTAMP)

TABLE: card_transactions
- txn_id (UUID, PRIMARY KEY)
- card_id (VARCHAR, FK → credit_cards.card_id)
- txn_date (DATE)
- txn_type (VARCHAR: purchase | cashadvance | payment | refund | fee)
- amount (NUMERIC)
- merchant_name (VARCHAR)
- category (VARCHAR)
- is_international (BOOLEAN)
- currency (VARCHAR)
- created_at (TIMESTAMP)

========================
CRITICAL RULES (MANDATORY)
========================

1.  Use ONLY the tables and columns listed above
2.  DO NOT invent or rename columns
3.  account_id, loan_id, fd_id, card_id are VARCHAR → ALWAYS quote them Example: account_id = '1345367'
4.  Use txn_date (NOT transaction_date)
5.  Use txn_type exactly as defined in schema
6.  Use SELECT or WITH only (read-only)
7.  Use JOINs only when logically required
8.  Prefer ORDER BY txn_date DESC for “latest” queries
9.  Apply LIMIT when returning lists (default ≤ 100)
10.  No INSERT, UPDATE, DELETE, DROP, ALTER
11.  No markdown, no comments, no explanation
12.  Output ONLY raw SQL

========================
USER QUERY
========================
{state['query']}

========================
SQL (OUTPUT ONLY)
========================
"""

    sql = llm.invoke(prompt).content[0]["text"].strip()
    return {"sql": sql}

def sql_executor_node(state: RAGState) -> RAGState:
    rows = execute_readonly_sql(state["sql"])
    return { "sql_result": rows}

def sql_summarizer_node(state: RAGState) -> RAGState:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

def sql_summarizer_node(state: RAGState) -> RAGState:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    prompt = f"""
You are a professional NorthStar Bank data summarizer.

Query:
{state.get("sql_query") or state["query"]}

Raw SQL Result (Python list of rows):
{state["sql_result"]}

Rules (MANDATORY):
- If the result contains transactions/purchases/history → output a clean Markdown table.
  Columns (use only what exists): Date | Merchant/Description | Category | Amount (INR) | Channel/Type
  Sort by date DESC.
- If single-value result (balance, EMI, outstanding, etc.) → output a clear, concise sentence.
- Professional banking tone. No SQL, no technical jargon.
- Output ONLY the Markdown/table or summary text. No extra explanation.

Final customer-facing summary:
"""

    answer = llm.invoke(prompt).content[0]["text"].strip()
    return { "sql_response": answer}

def hybrid_merger_node(state: RAGState) -> RAGState:
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    route = state.get("route", "")
    vector_ans = state.get("vector_response", "")
    sql_ans = state.get("sql_response", "")

    # 1. Build sources from reranked_docs (exactly as in test reports)
    sources: list[dict] = []
    if state.get("reranked_docs"):
        for idx, doc in enumerate(state["reranked_docs"]):
            meta = doc.metadata or {}
            sources.append({
                "rank": idx + 1,
                "page_number": meta.get("page_number", 0),
                "section": meta.get("section", ""),
                "chunk_type": meta.get("chunk_type", "text"),
                "content_preview": (doc.page_content[:300] + "...") if len(doc.page_content) > 300 else doc.page_content
            })

    # 2. Build sql_metadata
    sql_metadata: dict | None = None
    if state.get("sql") and state.get("sql_result") is not None:
        rows = state["sql_result"]
        sql_metadata = {
            "sql_query": state.get("sql_query") or state["query"],
            "executed_sql": state["sql"],
            "row_count": len(rows) if isinstance(rows, list) else 0,
            "result_preview": str(rows[:3]) if rows else "No rows returned"
        }

    # 3. Synthesize final answer (pure vs hybrid)
    if route != "hybrid":
        # Pure paths – keep speed
        final_answer = vector_ans if route == "vector" else sql_ans
    else:
        # Hybrid – true cross-validation + attributions
        vector_source_refs = "\n".join(
            [f"Page {d.metadata.get('page_number', 'N/A')} - {d.metadata.get('section', 'General')}"
             for d in state.get("reranked_docs", [])]
        ) or "None"

        synth_prompt = f"""
You are the final Answer Synthesizer for NorthStar Bank.

Vector/PDF Knowledge (from product knowledge base):
{vector_ans or "No PDF information available"}

SQL/Banking Records Summary (already formatted):
{sql_ans or "No SQL records found"}

Source references (use these for citations):
PDF Sources:
{vector_source_refs}

SQL Records: {sql_metadata['row_count'] if sql_metadata else 0} rows

Task:
- Produce ONE coherent, professional customer response.
- Keep the SQL Markdown table exactly as provided (do not reformat).
- Add inline attributions: [PDF: Page X - Section Y] for vector facts and [SQL: {sql_metadata['row_count'] if sql_metadata else 0} records] for SQL facts.
- Cross-validate both sources.
- Professional banking tone, concise, actionable.

Reply ONLY with the final customer-facing answer (no extra text, no explanations).
"""

    final_answer = llm.invoke(synth_prompt).content[0]["text"].strip()

    # 4. Full compliant output
    full_response = {
        "answer": final_answer,
        "sources": sources,
        "sql_metadata": sql_metadata,
        "warnings": ["Some vector sources could not be validated"] if state.get("is_valid") == "no" else []
    }

    return {"response": full_response}

def smart_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("route", query_router_node)
    graph.add_node("query_split", query_splitter_node)

    graph.add_node("tool_call", tool_call_agent_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("validate", validate_node)
    graph.add_node("generate", generate_answer_node)

    graph.add_node("sql_plan", sql_planner_node)
    graph.add_node("sql_exec", sql_executor_node)
    graph.add_node("sql_sum", sql_summarizer_node)

    graph.add_node("hybrid_merge", hybrid_merger_node)

    graph.set_entry_point("route")

    graph.add_conditional_edges(
        "route",
        lambda s: s["route"],
        {
            "vector": "tool_call",
            "rdbms": "sql_plan",
            "hybrid": "query_split",
        },
    )

    graph.add_edge("query_split", "tool_call")
    graph.add_edge("query_split", "sql_plan")

    graph.add_edge("tool_call", "rerank")
    graph.add_edge("rerank", "validate")
    graph.add_edge("validate", "generate")

    graph.add_edge("sql_plan", "sql_exec")
    graph.add_edge("sql_exec", "sql_sum")

    graph.add_edge("generate", "hybrid_merge")
    graph.add_edge("sql_sum", "hybrid_merge")

    graph.add_edge("hybrid_merge", END)

    return graph.compile()

rag_graph = smart_rag_graph()

image = rag_graph.get_graph().draw_mermaid_png() 

with open("diagram\\graph.png","wb") as f:
    f.write(image)

def run_agent(query: str) -> dict:
    state: RAGState = {
        "query": query,
        "route": "",
        "vector_query": None,
        "sql_query": None,
        "retrived_docs": [],
        "reranked_docs": [],
        "sql": None,
        "sql_result": None,
        "vector_response": None,
        "sql_response": None,
        "response": {},
        "is_valid": "",
        "tool": "",
        "feedback": None,
        "answer_attempts": 0,
        "query_history": [],
        "rewrite_count": 0,
    }

    final_state = rag_graph.invoke(state)
    response_data = final_state.get("response", {})

    # Explicitly return answer + sources (plus metadata/warnings if needed)
    return {
        "answer": response_data.get("answer", ""),
        "sources": response_data.get("sources", []),
        "sql_metadata": response_data.get("sql_metadata"),
        "warnings": response_data.get("warnings", [])
    }
