from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

res = search_flights("Plan a 7 days Japan trip from China")
print(res)