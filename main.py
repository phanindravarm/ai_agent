import os
from dotenv import load_dotenv
from google import genai
import psycopg2
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import operator


load_dotenv()

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="postgres", 
    user="postgres",              
    password=os.getenv("PG_PASSWORD") 
)
cur = conn.cursor()

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
        Search the internal knowledge base for information from uploaded PDF document

        Args:
        query: A specific search string or question to look up in the document database.
    
    Returns:
        A string containing the most relevant passages from the documents, 
        including source details like document name and page numbers.
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
        ORDER BY embedding <-> %s::vector  -- The fix: add ::vector
        LIMIT 3
    """, (query_embedding,))

    results = cur.fetchall()

    formatted_results = []
    for text, doc, page, chunk_id in results:
        formatted_results.append(
            f"{text}\n\n(Source: {doc}, Page {page}, Chunk ID: {chunk_id})"
        )

    return "\n\n---\n\n".join(formatted_results)

genai_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=genai_api_key)

system_prompt = """
    You are a helpful assistant with access to a document database.
    When a user asks a question:
    1. First, search the 'query_knowledge_base'.
    2. CRITICALLY EVALUATE the results:
    - If the tool returns actual explanations, use them.
    - If the tool returns ONLY citations (like "J. Chem Phys..."), references, or says no information found, DISREGARD those results. 
    3. If the document doesn't have a real explanation, use your OWN internal general knowledge to answer the user's question directly.
    4. Always be honest: if you are answering from your own knowledge because the document was unhelpful, you can briefly mention that.
"""

tools = [calculator, query_knowledge_base]
tool_node = ToolNode(tools)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

client = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
client_with_tools = client.bind_tools(tools)

def call_model(state: AgentState):
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = client_with_tools.invoke(messages)
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"\n[AI DECISION]: Calling tool '{tool_call['name']}' with args: {tool_call['args']}")
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

state = app.invoke({"messages": [HumanMessage(content="Find the yield percentage or concentration of cobalt mentioned in the report, and then multiply that number by 1.5 to estimate the scaled-up requirement.")]})


print("Result")
final_response = state["messages"][-1]

print(final_response.content)