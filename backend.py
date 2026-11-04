import os
import requests
import certifi
from dotenv import load_dotenv


load_dotenv()

os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

from typing import TypedDict, Annotated
import operator
import uuid

import psycopg
from psycopg.rows import dict_row

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "缺少数据库地址，请在 .env 中配置 DATABASE_URL。"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("缺少大模型密钥，请在 .env 中配置 GEMINI_API_KEY 或 GOOGLE_API_KEY。")


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=GEMINI_API_KEY
)

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int


def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query, llm=llm)

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="已完成航班查询。")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def hotel_agent(state: TravelState):
    query = f"酒店推荐、住宿价格与交通便利性：{state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="已获取酒店信息。")
        ],
        "llm_calls": state.get("llm_calls", 0)
    }


def itinerary_agent(state: TravelState):
    prompt = f"""
请用简体中文制定完整、实用的旅行行程。

用户需求：
{state['user_query']}

航班查询结果：
{state['flight_results']}

酒店搜索结果：
{state['hotel_results']}

行程应易于执行，兼顾用户预算、交通时间与出行节奏。
标题、说明、酒店推荐和建议均使用简体中文，专有名称可保留原文并附中文说明。
保留原始机场代号、航班号、时间、链接及金额币种，不编造航班、酒店报价或实时数据。
航班查询失败、缺少地点或数据不完整时，请如实说明并提示补充，不要当作已成功查到。
实时航班状态不能当作用户未来出行日期的可预订航班。
"""

    response = llm.invoke([
        SystemMessage(content="你是专业旅行规划师，始终使用简体中文回复。"),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.text,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def final_agent(state: TravelState):
    final_prompt = f"""
请用简体中文为用户生成最终旅行方案。

用户需求：
{state['user_query']}

航班查询结果：
{state['flight_results']}

酒店搜索结果：
{state['hotel_results']}

行程草案：
{state['itinerary']}

请使用以下中文章节，以清晰的 Markdown 格式输出：

1. 行程概览
2. 航班信息
3. 酒店推荐
4. 每日行程
5. 预算估算
6. 出行建议

要求：
- 所有标题、说明、提示均使用简体中文；机场和酒店原文名称可附中文说明。
- 保留原始机场代号、航班号、时间、链接及币种，不能编造航班或价格。
- 展示本次识别的出发与到达机场列表；多机场查询有失败时必须说明结果不完整。
- 若航线不明确或查询失败，明确提示补充信息或重试，不声称已查到可用航班。
- 航班接口提供实时状态，不提供机票报价，不代表未来日期的可预订航班。
- 预算估算需注明币种与估算性质，方便实际出行参考。
"""

    response = llm.invoke([
        SystemMessage(content="你是专业的中文旅行规划助手，始终使用简体中文回复。"),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


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

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
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

    final_answer = result["messages"][-1].text

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }
