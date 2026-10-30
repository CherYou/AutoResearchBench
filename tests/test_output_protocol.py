import json
import re
import unittest
from unittest.mock import AsyncMock, patch

from inference import AgentService
from prompts import AGENT_SYSTEM_PROMPT, SUMMARY_MODEL_PROMPT, TOOL_ARGS
from utils import build_agent_messages, parse_agent_output


CALL = {"name": "search", "arguments": {"query": "retrieval-augmented generation"}}


class PromptTests(unittest.TestCase):
    def test_tool_schema_is_valid_json(self):
        body = re.search(r"<tools>(.*?)</tools>", TOOL_ARGS, re.DOTALL).group(1)
        schema = json.loads(body)
        self.assertEqual(schema["function"]["name"], "search")
        self.assertEqual(schema["function"]["parameters"]["required"], ["query"])

    def test_final_prompt_includes_valid_schema(self):
        body = re.search(r"<tools>\s*(\{.*?)</tools>", AGENT_SYSTEM_PROMPT, re.DOTALL).group(1)
        self.assertEqual(json.loads(body)["type"], "function")

    def test_rendered_agent_prompt_has_no_escaped_braces(self):
        self.assertNotIn("{{", AGENT_SYSTEM_PROMPT)

    def test_short_tool_example_can_be_followed_literally(self):
        body = re.search(r"<tool_call>\n(.*?)\n</tool_call>", AGENT_SYSTEM_PROMPT, re.DOTALL).group(1)
        # Fill the original abstract placeholders, if present, without repairing braces.
        body = body.replace("<function-name>", '"search"').replace("<args-json-object>", '{"query": "test"}')
        parsed = parse_agent_output(f"<tool_call>{body}</tool_call>")
        self.assertEqual(parsed["action"]["type"], "tool")
        self.assertEqual(parsed["action"]["content"]["name"], "search")

    def test_complete_search_example(self):
        match = re.search(r"Example search turn:\n```text\n(.*?)\n```", AGENT_SYSTEM_PROMPT, re.DOTALL)
        self.assertIsNotNone(match)
        parsed = parse_agent_output(match.group(1))
        self.assertEqual(parsed["action"]["type"], "tool")
        self.assertEqual(parsed["candidate_state"], "empty")

    def test_complete_finish_example(self):
        match = re.search(r"Example finish turn[^\n]*:\n```text\n(.*?)\n```", AGENT_SYSTEM_PROMPT, re.DOTALL)
        self.assertIsNotNone(match)
        parsed = parse_agent_output(match.group(1))
        self.assertEqual(parsed["action"]["type"], "finish")
        self.assertEqual(parsed["candidate_state"], "none")

    def test_summary_format_preserves_json_and_substitutions(self):
        rendered = SUMMARY_MODEL_PROMPT.format(query="test {query}", paper_content="paper {data}")
        body = re.search(r"```json\n(.*?)\n```", rendered, re.DOTALL).group(1)
        self.assertIsInstance(json.loads(body)["summary"], str)
        self.assertIn("test {query}", rendered)
        self.assertIn("paper {data}", rendered)


class ParserTests(unittest.TestCase):
    def test_normal_search_and_candidate_ids(self):
        parsed = parse_agent_output('<candidates>[0], [3]</candidates><tool_call>' + json.dumps(CALL) + '</tool_call>')
        self.assertEqual(parsed["action"], {"type": "tool", "content": CALL})
        self.assertEqual(parsed["candidate_ids"], [0, 3])

    def test_json_code_fence(self):
        parsed = parse_agent_output('<tool_call>```json\n' + json.dumps(CALL) + '\n```</tool_call>')
        self.assertEqual(parsed["action"], {"type": "tool", "content": CALL})

    def test_candidate_states(self):
        for content, state in [("None", "none"), ("[]", "empty"), ("", "empty")]:
            with self.subTest(content=content):
                parsed = parse_agent_output(f'<candidates>{content}</candidates><answer>Done</answer>')
                self.assertEqual(parsed["candidate_state"], state)
                self.assertEqual(parsed["action"]["type"], "finish")

    def test_nested_candidates_remain_an_error_with_correction(self):
        parsed = parse_agent_output('<tool_call><candidates>None</candidates></tool_call>')
        self.assertEqual(parsed["action"]["type"], "error")
        self.assertEqual(parsed["candidate_state"], "none")
        self.assertIn("outside", parsed["action"]["content"])
        self.assertIn('<answer>Done</answer>', parsed["action"]["content"])

    def test_bare_none_remains_an_error_with_correction(self):
        parsed = parse_agent_output('<tool_call>None</tool_call>')
        self.assertEqual(parsed["action"]["type"], "error")
        self.assertIn('<answer>Done</answer>', parsed["action"]["content"])

    def test_nested_candidates_with_json_are_not_silently_repaired(self):
        parsed = parse_agent_output('<tool_call><candidates>[0]</candidates>' + json.dumps(CALL) + '</tool_call>')
        self.assertEqual(parsed["action"]["type"], "error")
        self.assertEqual(parsed["candidate_ids"], [0])

    def test_invalid_json_and_missing_action_are_errors(self):
        for output in ['<tool_call>{{"name": "search"}}</tool_call>', '<candidates>None</candidates>']:
            with self.subTest(output=output):
                self.assertEqual(parse_agent_output(output)["action"]["type"], "error")

    def test_candidate_tag_inside_query_is_not_removed(self):
        call = {"name": "search", "arguments": {"query": "literal <candidates>None</candidates>"}}
        parsed = parse_agent_output('<tool_call>' + json.dumps(call) + '</tool_call>')
        self.assertEqual(parsed["action"], {"type": "tool", "content": call})

    def test_error_is_replayed_as_user_feedback(self):
        output = '<tool_call><candidates>None</candidates></tool_call>'
        error = parse_agent_output(output)["action"]["content"]
        history = [{"assistant_output": f'{output}\n<error>Parsing Error: {error}</error>', "paper_list": []}]
        messages = build_agent_messages("system", "query", history, 10)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("Parsing Error:", messages[-1]["content"])
        self.assertIn("outside", messages[-1]["content"])


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    def make_service(self, outputs, max_turns):
        service = AgentService.__new__(AgentService)
        service.model = "offline-test"
        service.max_turns = max_turns
        service.verbose = False
        service.visible_paper_number = 10
        service.max_context_tokens = 110000
        service.planner_agent = AsyncMock()
        service.planner_agent.send_message.side_effect = outputs
        service.search_engine = AsyncMock()
        service.search_engine.search.return_value = []
        return service

    async def test_format_error_recovers_on_next_existing_turn(self):
        service = self.make_service([
            '<tool_call><candidates>None</candidates></tool_call>',
            '<candidates>[]</candidates><tool_call>' + json.dumps(CALL) + '</tool_call>',
            '<candidates>None</candidates><answer>Done</answer>',
        ], 4)
        with patch('inference.count_tokens', return_value=1):
            result = await service._run_single_pass('test')
        self.assertEqual([turn['action'] for turn in result['turn_details']], ['error', 'tool', 'finish'])
        self.assertEqual(result['status'], 'finished')
        service.search_engine.search.assert_awaited_once_with(query=CALL['arguments']['query'], top_k=10)
        feedback = service.planner_agent.send_message.call_args_list[1].kwargs['messages']
        self.assertTrue(any(m['role'] == 'user' and 'outside' in m['content'] for m in feedback))

    async def test_invalid_output_does_not_get_free_retries(self):
        output = '<tool_call><candidates>None</candidates></tool_call>'
        service = self.make_service([output, output], 2)
        with patch('inference.count_tokens', return_value=1):
            result = await service._run_single_pass('test')
        self.assertEqual(service.planner_agent.send_message.await_count, 2)
        self.assertEqual(result['status'], 'max_turns_reached')
        self.assertEqual([t['action'] for t in result['turn_details']], ['error', 'error'])
        service.search_engine.search.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
