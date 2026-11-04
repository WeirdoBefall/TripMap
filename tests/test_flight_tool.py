import unittest
from itertools import product
from unittest.mock import Mock, patch

import requests

from tools import flight_tool as flights


def location(name, kind, country, cities, codes):
    return {"source_text": name, "name": name, "kind": kind, "country_code": country,
            "city_names": cities, "airport_iatas": codes}


BEIJING = location("北京", "city", "CN", ["Beijing"], ["PEK", "PKX"])
TOKYO = location("东京", "city", "JP", ["Tokyo"], ["HND", "NRT"])


def model_for(**route):
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = route
    return model


def api_response(data):
    response = Mock()
    response.json.return_value = data
    return response


def flight(number="CA001", dep="PEK", arr="HND", date="2026-11-04"):
    return {"flight_date": date, "flight_status": "scheduled",
            "flight": {"iata": number}, "airline": {"name": "测试航空"},
            "departure": {"iata": dep, "scheduled": f"{date}T08:00:00+08:00"},
            "arrival": {"iata": arr}}


class FlightSearchTests(unittest.TestCase):
    def setUp(self):
        key_patch = patch.object(flights, "API_KEY", "test-key")
        origin_patch = patch.object(flights, "DEFAULT_ORIGIN_IATA", "")
        key_patch.start()
        origin_patch.start()
        self.addCleanup(key_patch.stop)
        self.addCleanup(origin_patch.stop)

    def test_chinese_and_reversed_order_queries_reach_model_unchanged(self):
        for query in ("从北京到东京的航班", "去东京，从北京出发", "Beijing 到 Tokyo 的机票"):
            with self.subTest(query=query):
                model = model_for(origin=BEIJING, destination=TOKYO)
                origins, destinations = flights.parse_route(query, llm=model)
                self.assertTrue({"PEK", "PKX"}.issubset(origins))
                self.assertEqual(set(destinations), {"HND", "NRT"})
                messages = model.with_structured_output.return_value.invoke.call_args.args[0]
                self.assertEqual(messages[-1].content, query)

    def test_database_completes_city_airports_omitted_by_model(self):
        shanghai = location("上海", "city", "CN", ["Shanghai"], ["PVG"])
        model = model_for(origin=shanghai, destination=TOKYO)
        self.assertEqual(flights.parse_route("从上海到东京", llm=model)[0], ["PVG", "SHA"])

    def test_country_uses_capital_instead_of_largest_city(self):
        cases = [("US", "New York", "JFK", "IAD"), ("AU", "Sydney", "SYD", "CBR"),
                 ("CA", "Toronto", "YYZ", "YOW"), ("AE", "Dubai", "DXB", "AUH"),
                 ("TR", "Istanbul", "IST", "ESB"), ("CN", "Shanghai", "PVG", "PEK")]
        for country, city, incorrect, expected in cases:
            with self.subTest(country=country):
                country_location = location(country, "country", country, [city], [incorrect])
                model = model_for(destination=country_location)
                self.assertEqual(flights.parse_route("去这个国家", llm=model), ([], [expected]))

    def test_unlisted_country_requires_capital_airport_match(self):
        ireland = location("爱尔兰", "country", "IE", ["Dublin"], ["DUB"])
        self.assertEqual(flights.resolve_location_to_iatas(flights.Location(**ireland)), ["DUB"])
        ireland["airport_iatas"] = ["ORK"]
        with self.assertRaisesRegex(ValueError, "首都机场"):
            flights.resolve_location_to_iatas(flights.Location(**ireland))

    def test_metropolitan_airports_and_foreign_country_validation(self):
        new_york = location("纽约", "city", "US", ["New York"], ["JFK", "EWR", "PEK", "ZZZ"])
        codes = flights.resolve_location_to_iatas(flights.Location(**new_york))
        self.assertTrue({"JFK", "LGA", "EWR"}.issubset(codes))
        self.assertNotIn("PEK", codes)
        self.assertNotIn("ZZZ", codes)

    def test_explicit_airport_does_not_expand_to_other_city_airports(self):
        airport = location("首都机场", "airport", "CN", ["Beijing"], ["PEK"])
        self.assertEqual(flights.parse_route("从首都机场出发", llm=model_for(origin=airport)), (["PEK"], []))
        airport["airport_iatas"] = ["ZZZ"]
        with self.assertRaisesRegex(ValueError, "机场代号"):
            flights.parse_route("从未知机场出发", llm=model_for(origin=airport))

    def test_destination_only_has_no_unrequested_default_origin(self):
        self.assertEqual(flights.parse_route("去东京", llm=model_for(destination=TOKYO)), ([], ["HND", "NRT"]))
        with patch.object(flights, "DEFAULT_ORIGIN_IATA", "CAN"):
            self.assertEqual(flights.parse_route("去东京", llm=model_for(destination=TOKYO))[0], ["CAN"])

    def test_standalone_search_uses_gemini_with_google_api_key(self):
        model = model_for(destination=TOKYO)
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-google-key", "GEMINI_API_KEY": ""}), \
                patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=model) as create_model:
            self.assertEqual(flights.parse_route("去东京"), ([], ["HND", "NRT"]))
        create_model.assert_called_once_with(
            model="gemini-3.5-flash-lite", api_key="test-google-key",
        )

    def test_explicit_iata_is_not_expanded_even_if_model_marks_it_as_city(self):
        airport = {**BEIJING, "source_text": "PEK"}
        self.assertEqual(flights.parse_route("从 PEK 出发", llm=model_for(origin=airport)), (["PEK"], []))

    @patch.object(flights.requests, "get")
    def test_unknown_or_ambiguous_input_never_requests_global_flights(self, get):
        routes = [{}, {"clarification": "请说明是哪个国家的同名城市。"},
                  {"destination": location("不存在的城市", "city", "CN", ["Unknown city"], ["ZZZ"])},
                  {"destination": {"kind": "city"}}]
        for route in routes:
            result = flights.search_flights("帮我安排一下", llm=model_for(**route))
            self.assertTrue("请" in result)
        get.assert_not_called()

    @patch.object(flights.requests, "get")
    def test_model_outage_does_not_fall_back_to_global_query(self, get):
        model = model_for()
        model.with_structured_output.return_value.invoke.side_effect = RuntimeError("provider down")
        self.assertIn("暂时无法识别航线", flights.search_flights("北京到东京", llm=model))
        get.assert_not_called()

    @patch.object(flights.requests, "get")
    def test_every_airport_combination_is_queried_even_with_one_result_limit(self, get):
        get.return_value = api_response({"data": [flight()]})
        model = model_for(origin=BEIJING, destination=TOKYO)
        origins = flights.resolve_location_to_iatas(flights.Location(**BEIJING))
        result = flights.search_flights("从北京到东京", limit=1, llm=model)
        pairs = {(call.kwargs["params"]["dep_iata"], call.kwargs["params"]["arr_iata"])
                 for call in get.call_args_list}
        self.assertEqual(pairs, set(product(origins, ["HND", "NRT"])))
        self.assertEqual(result.count("航班号：CA001"), 1)
        self.assertIn("不提供机票报价", result)

    @patch.object(flights.requests, "get")
    def test_only_explicit_worldwide_request_omits_both_filters(self, get):
        get.return_value = api_response({"data": []})
        result = flights.search_flights("全球航班", llm=model_for(worldwide=True))
        params = get.call_args.kwargs["params"]
        self.assertNotIn("dep_iata", params)
        self.assertNotIn("arr_iata", params)
        self.assertIn("无数据不代表该航线不存在", result)

    @patch.object(flights.requests, "get")
    def test_partial_failure_retains_success_and_warns_in_chinese(self, get):
        def fetch(url, *, params, timeout):
            if params["arr_iata"] == "NRT":
                return api_response({"error": {"code": "usage_limit_reached", "message": "quota exceeded"}})
            return api_response({"data": [flight()]})
        get.side_effect = fetch
        result = flights.search_flights("去东京", llm=model_for(destination=TOKYO))
        self.assertIn("航班号：CA001", result)
        self.assertIn("结果可能不完整", result)
        self.assertIn("调用额度已用尽", result)
        self.assertNotIn("quota exceeded", result)

    @patch.object(flights.requests, "get")
    def test_network_and_malformed_api_errors_are_chinese_and_hide_credentials(self, get):
        get.side_effect = requests.exceptions.ConnectionError("https://example.com?access_key=secret")
        result = flights.search_flights("去东京", llm=model_for(destination=TOKYO))
        self.assertIn("请求失败", result)
        self.assertNotIn("secret", result)
        self.assertNotIn("未找到匹配", result)
        get.side_effect = None
        for payload in ([1], {"data": "invalid"}, {"error": "invalid"}):
            get.return_value = api_response(payload)
            result = flights.search_flights("去东京", llm=model_for(destination=TOKYO))
            self.assertIn("检索失败", result)

    @patch.object(flights.requests, "get")
    def test_same_flight_number_on_different_dates_is_retained(self, get):
        get.return_value = api_response({"data": [flight(), flight(), flight(date="2026-11-05")]})
        result = flights.search_flights("全球航班", llm=model_for(worldwide=True))
        self.assertEqual(result.count("航班号：CA001"), 2)

    def test_missing_api_key_and_invalid_limit_fail_before_using_model(self):
        model = Mock()
        with patch.object(flights, "API_KEY", ""):
            self.assertIn("缺少航班接口密钥", flights.search_flights("北京到东京", llm=model))
        self.assertIn("1 到 100", flights.search_flights("北京到东京", limit=0, llm=model))
        model.with_structured_output.assert_not_called()

    def test_flight_fields_status_and_null_fallbacks_are_chinese(self):
        result = flights.format_flight({"airline": None, "flight": None, "departure": None,
                                       "arrival": {"delay": 0}, "flight_status": "active"})
        self.assertIn("未知航空公司", result)
        self.assertIn("状态：飞行中", result)
        self.assertIn("0 分钟", result)
        self.assertNotIn("Unknown", result)


if __name__ == "__main__":
    unittest.main()
