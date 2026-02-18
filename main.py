import os
from dotenv import load_dotenv
import psycopg2
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import operator
import time

load_dotenv()

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="postgres", 
    user="postgres",              
    password=os.getenv("PG_PASSWORD") 
)
cur = conn.cursor()

client = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
TOOL_TIMEOUT = 5

@tool
def calculator(a, b, operation):
    """
    Performs arithmetic.
    Args:
        a: The first number.
        b: The second number.
        operation: Must be 'add', 'subtract', 'multiply', or 'divide'.
    """    
    try:
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                return "Error: Division by zero"
            return a / b
        else:
            return "Error: Unknown operation"
    except Exception as e:
        return f"Error occurred: {e}"

@tool    
def query_knowledge_base(query):
    """
        Multi-document retrieval:
        1. Embed query
        2. Retrieve top 5 documents
        3. Retrieve top 8 chunks from those documents
        4. Format with citations
    """
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="retrieval_query",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    query_embedding = embeddings_model.embed_query(query)

    cur.execute("""
        SELECT chunk_text, document_name, page_number, chunk_id
        FROM pdf_chunks
        JOIN documents USING(document_id)
        ORDER BY embedding <-> %s::vector
        LIMIT 10
    """, (query_embedding,))

    results = cur.fetchall()

    if not results:
        return "No relevant content found"

    formatted_results = []
    for text, doc, page, chunk_id in results:
        formatted_results.append(
            f"{text}\n\n(Source: {doc}, Page {page}, Chunk ID: {chunk_id})"
        )

    return "\n\n---\n\n".join(formatted_results)

@tool
def reranker(chunks, query):
    """
    Re-rank top-N chunks using LLM to find the most relevant answer.
    """
    prompt = f"Query: {query}\n\nChunks:\n"
    for i, c in enumerate(chunks):
        snippet = c["text"][:300].replace("\n", " ")
        prompt += f"{i+1}. {snippet}...\n"
    
    response = client.invoke([{"role": "user", "content": prompt}])

    ranked_chunk_ids = response.messages[0].content.split("\n")

    ranked_chunks = []
    for cid in ranked_chunk_ids:
        for c in chunks:
            if c["chunk_id"] in cid:
                ranked_chunks.append(c)

    return ranked_chunks

TOOL_PERMISSIONS = {
    "calculator": {"can_run": True},
    "query_knowledge_base": {"can_run": True},
    "reranker": {"can_run": True}
}

def safe_tool_call(tool_fn, **kwargs):
    start = time.time()
    result = tool_fn.invoke(kwargs)
    if time.time() - start > TOOL_TIMEOUT:
        return "Error: Tool timed out"
    return result

def safe_execute_tool(tool_name, *args, **kwargs):
    perms = TOOL_PERMISSIONS.get(tool_name, {})
    if not perms.get("can_run", True):
        return f"Permission denied for {tool_name}"
    return safe_tool_call(tools_dict[tool_name], *args, **kwargs)


system_prompt = """
    You are a helpful assistant with access to a document database.
    When a user asks a question:
    1. First, search the 'query_knowledge_base'.
    2. CRITICALLY EVALUATE the results:
    - If the tool returns actual explanations, use them.
    - If the tool returns ONLY citations (like "J. Chem Phys..."), references, or says no information found, DISREGARD those results. 
    3. If the document doesn't have a real explanation, use your OWN internal general knowledge to answer the user's question directly.
    4. Always be honest: if you are answering from your own knowledge because the document was unhelpful, you can briefly mention that.
    If retrieval fails or returns no meaningful content,
    DO NOT answer from your own knowledge.
    Instead say:
    "I could not find the answer in the document."

"""

tools_dict = {
    "calculator": calculator,
    "query_knowledge_base": query_knowledge_base,
    "reranker": reranker
}

tool_node = ToolNode(list(tools_dict.values()))

client_with_tools = client.bind_tools(tools=list(tools_dict.values()))

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    tool_calls_count: int

def call_model(state: AgentState):
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = client_with_tools.invoke(messages)
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            args = tool_call['args']
            result = safe_execute_tool(tool_name, **args)
            print(f"\n[AI DECISION]: Calling tool '{tool_name}' with args: {args}")
            print(f"[TOOL RESULT]: {result}")
    else:
        print("\n[AI DECISION]: Generating final text response...")
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tool", tool_node)

workflow.add_edge(START,"agent")
workflow.add_conditional_edges(
    "agent", 
    should_continue,
    {
        "tools": "tool",  
        END: END       
    }
)
workflow.add_edge("tool", "agent")

app = workflow.compile()

state = {"messages" : [], "tool_calls_count" : 0}

while True:
    user_input = input("User : ")
    
    if user_input.lower() == "exit":
        break

    state = app.invoke({
        "messages": state["messages"] + [HumanMessage(content=user_input)],
        "tool_calls_count": state["tool_calls_count"]
    })

    print("Assistant : ", state["messages"][-1].content)