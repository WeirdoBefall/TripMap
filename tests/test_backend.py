import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from tools import flight_tool


class LocalSaver(InMemorySaver):
    def setup(self):
        pass


class TravelGraphTests(unittest.TestCase):
    def test_real_graph_connects_extraction_search_and_chinese_generation(self):
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = {
            "origin": {"source_text": "上海", "name": "上海", "kind": "city", "country_code": "CN",
                       "city_names": ["Shanghai"], "airport_iatas": ["PVG"]},
            "destination": {"source_text": "东京", "name": "东京", "kind": "city", "country_code": "JP",
                            "city_names": ["Tokyo"], "airport_iatas": ["HND"]},
        }
        model.invoke.side_effect = [
            AIMessage(content=[{"type": "text", "text": "中文每日行程"}]),
            AIMessage(content=[{"type": "text", "text": "中文完整旅行方案"}]),
        ]
        path = Path(__file__).resolve().parents[1] / "backend.py"
        spec = importlib.util.spec_from_file_location("tripmap_test_backend", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key", "GEMINI_API_KEY": "", "TAVILY_API_KEY": "test-key",
                                     "DATABASE_URL": "postgresql://test:test@localhost/test"}), \
                patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=model) as create_model, \
                patch("psycopg.connect") as connect, \
                patch("langgraph.checkpoint.postgres.PostgresSaver", return_value=LocalSaver()):
            spec.loader.exec_module(module)

        create_model.assert_called_once_with(model="gemini-3.5-flash-lite", api_key="test-key")

        response = Mock()
        response.json.return_value = {"data": []}
        module.tavily_search = Mock(return_value="中文酒店资料")
        with patch.object(flight_tool, "API_KEY", "test-key"), \
                patch.object(flight_tool.requests, "get", return_value=response) as get:
            result = module.run_travel_agent("从上海去东京玩五天", thread_id="test-chinese-graph")

        self.assertEqual(result["answer"], "中文完整旅行方案")
        self.assertEqual(result["itinerary"], "中文每日行程")
        self.assertEqual(result["llm_calls"], 3)
        self.assertEqual(result["thread_id"], "test-chinese-graph")
        self.assertIn("出发机场：PVG、SHA", result["flight_results"])
        self.assertIn("到达机场：HND、NRT", result["flight_results"])
        self.assertEqual(get.call_count, 4)
        self.assertIn("酒店推荐", module.tavily_search.call_args.args[0])
        for call in model.invoke.call_args_list:
            self.assertIn("简体中文", call.args[0][0].content)
        self.assertIn("预算估算", model.invoke.call_args_list[-1].args[0][1].content)
        connect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
