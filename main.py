import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from duckduckgo_search import DDGS
from pypdf import PdfReader
import psycopg2

load_dotenv()

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="postgres", 
    user="postgres",              
    password=os.getenv("PG_PASSWORD") 
)
cur = conn.cursor()

def calculator(a, b, operation):
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
    
def search_web(query):
    """Searches the web for current information using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
            return results if results else "No results found."
    except Exception as e:
        return f"Search error: {e}"
    
def query_knowledge_base(query):
    query_embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    ).embeddings[0].values

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

calculator_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="calculator",
            description="Performs basic arithmetic operations: addition, subtraction, multiplication, division.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "a": {"type": "NUMBER", "description": "The first number"},
                    "b": {"type": "NUMBER", "description": "The second number"},
                    "operation": {
                        "type": "STRING", 
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The arithmetic operation to perform"
                    }
                },
                "required": ["a", "b", "operation"]
            }
        )
    ]
)

search_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name = "search_web",
            description="Search the web for real-time information, news, or facts.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "query": {"type": "STRING", "description": "The search query"}
                },
                "required": ["query"]
            }
        )
    ]
)

query_knowledge_base_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name = "query_knowledge_base",
            description="Search your internal memory for info from previously loaded PDFs.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "query": {"type": "STRING"}
                },
                "required": ["query"]
            }
        )
    ]
)

system_prompt = """
You are a document QA agent.

Tool priority rules:

- For questions about the uploaded report/document, ALWAYS call query_knowledge_base first.
- Only call search_web if query_knowledge_base returns empty or irrelevant context.
- Never call search_web repeatedly.
- After retrieving context, answer immediately.

Do not loop.
"""

messages = [
    types.Content(role="user", parts=[types.Part.from_text(text="What about xenon")])
]


config = types.GenerateContentConfig(
    system_instruction = system_prompt,
    tools=[calculator_tool, search_tool, query_knowledge_base_tool],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
)


available_functions = {
    "calculator": calculator,
    "search_web": search_web,
    "query_knowledge_base": query_knowledge_base
}

max_steps = 3
current_step = 0

while current_step < max_steps:
    response = client.models.generate_content(
        model="gemini-3-flash-preview", 
        contents=messages,
        config=config,
    )

    model_turn = response.candidates[0].content
    messages.append(model_turn)

    function_calls = [part.function_call for part in model_turn.parts if part.function_call]

    if function_calls:
        current_step += 1
        response_parts = []
        
        for fn_call in function_calls:
            print(f"Model requested: {fn_call.name}({fn_call.args})")
            function_to_call = available_functions[fn_call.name]
            result = function_to_call(**fn_call.args)
            
            print(f"Result: {result}")

            response_parts.append(
                types.Part.from_function_response(
                    name=fn_call.name,
                    response={"result": result}
                )
            )

        messages.append(types.Content(role="tool", parts=response_parts))
    else:
        print(f"\nFinal Response: {response.text}")
        break