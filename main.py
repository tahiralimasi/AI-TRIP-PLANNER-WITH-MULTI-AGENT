
import os
import operator
from typing import TypedDict, Annotated

import psycopg
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is missing.")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is missing.")


# ==========================================
# 2. LLM CONFIGURATION
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
    temperature=0.3
)


# ==========================================
# 3. TRAVEL STATE
# ==========================================

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


# ==========================================
# 4. FLIGHT AGENT
# ==========================================

def flight_agent(state: TravelState):

    query = state["user_query"]

    try:
        flight_data = search_flights(query)
    except Exception as e:
        flight_data = f"Flight search failed: {str(e)}"

    return {
        "flight_results": str(flight_data),
        "messages": [
            AIMessage(content="Flight search completed.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ==========================================
# 5. HOTEL AGENT
# ==========================================

def hotel_agent(state: TravelState):

    query = f"Best hotels for {state['user_query']}"

    try:
        hotel_results = tavily_search(query)
    except Exception as e:
        hotel_results = f"Hotel search failed: {str(e)}"

    return {
        "hotel_results": str(hotel_results),
        "messages": [
            AIMessage(content="Hotel search completed.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ==========================================
# 6. ITINERARY AGENT
# ==========================================

def itinerary_agent(state: TravelState):

    prompt = f"""
Create a detailed travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Include:
- Travel schedule
- Hotel recommendations
- Daily activities
- Estimated travel expenses

Clearly mention when flight or hotel information is unavailable.
Do not invent confirmed flight bookings or hotel availability.
"""

    response = llm.invoke([
        SystemMessage(
            content="You are an expert travel planner. "
                    "Create practical and well-organized travel plans."
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": str(response.content),
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ==========================================
# 7. FINAL RESPONSE AGENT
# ==========================================

def final_agent(state: TravelState):

    final_prompt = f"""
Generate the final travel planning response.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Itinerary:
{state['itinerary']}

Present the result in a clear and structured format.

Use headings and bullet points.
Mention estimated costs clearly.
Do not claim that bookings have been made.
"""

    response = llm.invoke([
        SystemMessage(
            content="You are a professional travel planning assistant. "
                    "Give clear, helpful and organized responses."
        ),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


# ==========================================
# 8. BUILD LANGGRAPH WORKFLOW
# ==========================================

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# ==========================================
# 9. SUPABASE POSTGRESQL CONNECTION
# ==========================================

# Use Supabase Session Pooler connection URL.
# Keep the connection open for LangGraph checkpointing.

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    prepare_threshold=0,
    connect_timeout=15
)

# Create checkpoint tables if they do not exist.
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

# Compile graph with persistent memory.
app = graph.compile(
    checkpointer=checkpointer
)


# ==========================================
# 10. CLI EXECUTION
# ==========================================

if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "user_aarohi"
        }
    }

    user_input = input("Enter your travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")

    print(result["messages"][-1].content)

    print("\nTotal agent steps:", result["llm_calls"])
