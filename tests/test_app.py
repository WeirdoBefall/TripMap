import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


class ChineseAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 隔离外部服务，直接验证真实 FastAPI 路由和页面模板。
        backend = types.ModuleType("backend")
        cls.run_agent = Mock()
        backend.run_travel_agent = cls.run_agent
        path = Path(__file__).resolve().parents[1] / "app.py"
        spec = importlib.util.spec_from_file_location("tripmap_test_app", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"backend": backend}):
            spec.loader.exec_module(module)
        cls.client = TestClient(module.app)

    def setUp(self):
        self.run_agent.reset_mock(side_effect=True, return_value=True)

    def test_chinese_homepage_and_quick_prompts(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        for text in ('lang="zh-CN"', "TripMap 智能旅行规划", "生成旅行方案", "东京七日游", "会话编号"):
            self.assertIn(text, response.text)
        for text in ("Plan Your Perfect Trip", "Generate Plan", "TripMate", "Thread ID"):
            self.assertNotIn(text, response.text)

    def test_empty_and_invalid_requests_return_chinese_errors(self):
        for payload, status in (({"message": "  "}, 400), ({}, 422), ({"message": 123}, 422)):
            response = self.client.post("/api/travel", json=payload)
            self.assertEqual(response.status_code, status)
            self.assertFalse(response.json()["success"])
            self.assertTrue(any("\u4e00" <= char <= "\u9fff" for char in response.json()["error"]))
        self.run_agent.assert_not_called()

    def test_chinese_query_and_thread_reach_backend(self):
        self.run_agent.return_value = {
            "thread_id": "test-thread", "answer": "中文旅行方案", "flight_results": "航班信息",
            "hotel_results": "酒店推荐", "itinerary": "每日行程", "llm_calls": 3,
        }
        response = self.client.post("/api/travel", json={"message": " 从北京去东京 ", "thread_id": "test-thread"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "中文旅行方案")
        self.run_agent.assert_called_once_with(user_input="从北京去东京", thread_id="test-thread")

    def test_provider_exception_is_localized_and_does_not_leak_details(self):
        self.run_agent.side_effect = RuntimeError("English error: secret-api-key")
        with patch("traceback.print_exc"), patch("builtins.print"):
            response = self.client.post("/api/travel", json={"message": "北京去东京"})
        self.assertEqual(response.status_code, 500)
        self.assertIn("暂时无法生成旅行方案", response.json()["error"])
        self.assertNotIn("secret-api-key", response.text)


if __name__ == "__main__":
    unittest.main()
