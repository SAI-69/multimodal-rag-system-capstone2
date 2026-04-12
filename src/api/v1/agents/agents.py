import os 
from typing import TypedDict, List, Optional 

import cohere 
from dotenv import load_dotenv 
from langchain_core.documents import Document 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langgraph.graph import StateGraph, END 
from langchain_core.tools import tool 
from langchain_core.prompts import ChatPromptTemplate

from src.api.v1.tools.fts_search import fts_search
from src.api.v1.tools.hybrid_search import hybrid_search
from src.api.v1.tools.vector_search import vector_search
from src.api.v1.schemas.query_schema import AIResponse, FeedBack
# from src.core.db import get_vector_store 

load_dotenv(override=True) 

# ============= TOOLS ============= 
@tool
def fts_search_tool(query: str):
    """Performs full text search using keyword matching over indexed content, supporting exact terms, phrase queries, and text relevance scoring for precise lookup."""
    return fts_search(query)

@tool
def hybrid_search_tool(query: str):
    """Combines full text keyword search with vector based semantic search to balance precision and contextual relevance in retrieval results."""
    return hybrid_search(query) 

@tool
def vector_search_tool(query: str):
    """Executes semantic search by comparing embedding vectors to find conceptually similar content, even when exact keywords are not present."""
    return vector_search(query) 

# ============= STATE ============= 

class RAGState(TypedDict):
    query: str 
    retrived_docs: List[Document] 
    reranked_docs: List[Document] 
    response: dict 
    is_valid: str 
    tool: str
    feedback: Optional[FeedBack]


# ============= Tool Call Agent Node =============

def tool_call_agent_node(state: RAGState) -> RAGState:
    tools = [vector_search_tool,fts_search_tool,hybrid_search_tool]
    agent1 = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview")
    agent1.bind_tools(tools) 

    prompt = f"""You are an tool call agent node in an RAG System.
                Your only job is to decide which tool to call for an user query.
                Your are provide with three tools.
                fts_search_tool : Performs fast keyword-based full text search to retrieve exact or near exact matches from indexed documents.
                vector_search_tool: Retrieves semantically similar results by comparing embedding vectors using similarity scoring.
                hybrid_search_tool: Combines full text keyword matching with vector similarity search to deliver both precise and contextually relevant results.
                Selecte the appropriate tool to call based on the user query to search my vector db.
                Produce the output in one word (i.e) tool name alone. Like 'fts_search_tool' or 'vector_search_tool' or 'hybrid_search_tool'. 
                User query : {state["query"]}
                """ 
    response = agent1.invoke(prompt).content[0]['text'].lower() 

    if response == 'fts_search_tool':
        retrived_docs = fts_search_tool(state['query'])

    elif response == 'vector_search_tool':
        retrived_docs = vector_search(state['query'])

    elif response == 'hybrid_search_tool':
        retrived_docs = hybrid_search_tool(state['query']) 

    return{
        **state,
        'retrived_docs': retrived_docs,
        'tool': response
    }

# ============= RE-RANK NODE ============= 

def rerank_node(state: RAGState) -> RAGState:
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    docs = state["retrieved_docs"] 

    rerank_response = co.rerank(
        model="rerenak-english-v3.0",
        query=state["query"],
        documents=[doc.page_content for doc in docs],
        top_n=5
    )

    reranked_docs = [docs[r.index] for r in rerank_response.results]

    return {
        **state,
        "reranked_docs": reranked_docs
    }

# ============= VALIDATE NODE ============= 

def validate_node(state: RAGState) -> RAGState:
    print("\n=============Inside Validate Node=================\n")
    agent2 = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_LLM_MODEL"),
                                google_api_key=os.getenv("GOOGLE_API_KEY"))
    prompt = f"""Your only job is to validate whether reranked chunks from the rerank node is realted to the query or not.
                Note that respond with yes if and only if we can answer the query with the reranked chunks
                Respond with "yes" if it is realted or else "no"
                Query : {state["query"]}
                Re-Ranked Chunks : {state["reranked_docs"]}""" 
   
    response = agent2.invoke(prompt)
    print(f"\n==================== RESPONSE ==================== \n {response.content[0]['text']}\n")
    temp = ""
    if response.content[0]['text'].lower() == "yes":
       temp = "yes" 
    else:
       temp = "no"
    print("\n=============Outside Validate Node=================\n")
    return{
       **state,
       "is_valid" : temp
    }

# ============= REWRITTER NODE ============= 

def rewriter_node(state: RAGState) -> RAGState:
   print("\n=============Inside Writer Node=================\n")
   agent3 = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_LLM_MODEL"),
                                google_api_key=os.getenv("GOOGLE_API_KEY")) 
   prompt = f"""Your only job is to rewrite thr user query. 
                You are called only when in my RAG system if the user given query is too vague to fetch results.
                So your job is to re-write the query so that we can able to fetch the better chunks 
                Old Query : {state.query}""" 
   
   response = agent3.invoke(prompt) 

   print("\n=============Outside Writer Node=================\n")
   return {
      **state,
      "query" : response.content[0]['text']
   }

# ============= GENERATE ANSWER NODE ============= 

def generate_answer_node(state: RAGState) -> RAGState:
   print("\n ============= Output Generation started================= \n")
   agent4 = ChatGoogleGenerativeAI(
       model=os.getenv("GOOGLE_LLM_MODEL"),
       google_api_key=os.getenv("GOOGLE_API_KEY")
   )
   #structured_llm = agent4.with_structured_output(AIResponse)

   context = "\n\n".join([
       f"[Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
       for doc in state["reranked_docs"]
   ])

   prompt = f"""You are helpful AI Assistant your job is to generate answer for the user query only using the provided content.
                Check for any feedback if provided and regenerate the answer based on the feedback provided.
                User Query: {state['query']}
                Context: {context}
                Feedback: {state['feeback']}"""

   result = agent4.invoke(prompt)


   print(f"[generate_answer_node] Answer generated.")
   print("\n ============= Generator Generated output ================= \n")
   return {**state, "response": result.model_dump()}

# ============= EVALUATOR NODE ============= 
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def evaluator_node(state: RAGState) -> RAGState:
    print("\n============= Evaluation started =============\n")

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_LLM_MODEL"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0
    )

    # Enforce structured output
    evaluator = llm.with_structured_output(FeedBack)

    prompt = f"""
                You are an evaluator agent.

                Evaluate the generator's response using the following criteria:
                1. Conciseness – Is the answer clear and to the point?
                2. Hallucination – Does the answer contain false or made-up information?
                3. Relevancy – Does the answer directly address the user query?

                User Query: {state['query']}
                Generated Response:{state['response']}

                Rules:
                - If the answer is fully satisfactory, respond with:
                    response="Yes"
                    feedback fields should briefly confirm quality.
                - If not satisfactory, respond with:
                    response="No"
                    feedback fields should clearly explain issues.
                """

    evaluation: FeedBack = evaluator.invoke(prompt)

    print("\n============= Evaluation ended =============\n")

    return {
        **state,
        "feedback": evaluation.feedback
    }
# ============= ROUTE's =============

def validate_route(state: RAGState) -> str:
    
    print("\n ============= Validate Router Invoked ============= \n") 
    if state["is_valid"] == "yes":
       print("=============== Validate Route -> Generator ===============")
    else:
       print("=============== Validate Route -> Writer ===============")

    return "yes" if state["is_valid"] == "yes" else "no" 

def feedback_route(state: RAGState) -> str:

    print("\n ============= Feedback Router Invoked ============= \n") 
    if state["feedback"].response == "yes":
       print("=============== Feedback Route -> END ===============")
    else:
       print("=============== Feedback Route -> Generator ===============")

    return "yes" if state['feedback'].response == "yes" else "no" 

# ============= LANG-GRAPH ============= 

def multimodal_rag_graph():
    graph = StateGraph(RAGState) 

    graph.add_node("tool_call",tool_call_agent_node)
    graph.add_node("rerank",rerank_node) 
    graph.add_node("generate_answer",generate_answer_node) 
    graph.add_node("validate",validate_node) 
    graph.add_node("rewriter",rewriter_node) 
    graph.add_node("evaluator", evaluator_node)

    graph.set_entry_point("tool_call") 
    graph.add_edge("tool_call","rerank")
    graph.add_edge("rerank","validate") 
    graph.add_conditional_edges("validate",
                                validate_route,
                                {
                                    "yes": "generate_answer",
                                    "no": "rewriter"
                                })
    graph.add_edge("rewriter","tool_call")
    graph.add_edge("generate_answer","evaluator") 
    graph.add_conditional_edges("evaluator",
                                feedback_route,
                                {
                                    "yes": END,
                                    "no": "generate_answer"
                                })
    graph.add_edge("evaluator","generate_answer") 
    graph.add_edge("evaluator", END)

    return graph.compile() 

rag_graph = multimodal_rag_graph() 

image = rag_graph.get_graph().draw_mermaid_png() 

with open("diagram\\graph.png","wb") as f:
    f.write(image) 

def run_agent(query: str) -> dict:
   initial_state: RAGState = {
       "query": query,
       "retrieved_docs": [],
       "reranked_docs": [],
       "response": {},
       "is_valid": "",
       "tool": ""
   }
   final_state = rag_graph.invoke(initial_state)
   return final_state["response"] 

print("------ DONE ------")