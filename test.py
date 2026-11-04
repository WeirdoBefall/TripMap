from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from backend import run_travel_agent

res = run_travel_agent(user_input="我想在七月从中国去巴黎旅行五天，请推荐经济实惠的航班、酒店和行程。")
print(res)
