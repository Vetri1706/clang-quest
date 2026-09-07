"""Local evidence/intent tests; no network access or external model is used."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import unittest
from urllib.parse import urlparse

from backend.mentor import BM25, MAX_TEXT, load_knowledge, mentor_reply


CHALLENGE = {
    "id": "energy", "title": "Player energy", "summary": "Use integer division to count complete energy packs.",
    "story": "Prepare a player for a journey.", "difficulty": "beginner", "track": "games",
    "concepts": [{"title": "Integer division", "body": "Count whole groups; a leftover part does not form another group."}],
    "objectives": ["Read two integers", "Count complete groups", "Print the count"],
    "prompt": "Count the complete groups.", "input_format": "Two positive integers: energy and group size.",
    "output_format": "One integer followed by a newline.", "constraints": ["Both inputs are positive."],
    "starter_code": "#include <iostream>\nint main() {\n}\n",
    "reference_solution": "#include <iostream>\nint main() { int energy, size; std::cin >> energy >> size; std::cout << energy / size << '\\n'; }",
    "hints": ["First identify the amount available and the amount needed for one group.",
              "Trace one input where a partial group remains.",
              "Think about the operator that returns the number of complete groups."],
    "tests": [{"input": "7 3\n", "output": "2\n", "hidden": False},
              {"input": "987654 7\n", "output": "141093\n", "hidden": True}],
    "tags": ["division", "input", "output"]
}


class MentorTests(unittest.TestCase):
    def answer(self, question, **kwargs):
        return mentor_reply(question, copy.deepcopy(CHALLENGE), kwargs.pop("source", CHALLENGE["starter_code"]), **kwargs)

    def test_exact_api_and_original_card_inventory(self):
        cards = load_knowledge()
        self.assertEqual(len(cards), 24)
        self.assertEqual(len({card["id"] for card in cards}), 24)
        result = self.answer("Give me a hint")
        self.assertEqual(set(result), {"text", "kind", "sources", "suggestions", "next_hint_level", "model"})
        self.assertEqual(result["model"], {"mode": "grounded", "label": "Local study mentor"})
        self.assertLessEqual(len(result["text"]), MAX_TEXT)
        json.dumps(result, allow_nan=False)

    def test_hints_progress_without_revealing_solution(self):
        level = 0
        for index in range(4):
            result = self.answer("Give me the next hint", hint_level=level)
            self.assertEqual(result["kind"], "hint")
            self.assertIn(CHALLENGE["hints"][min(index, 2)], result["text"])
            self.assertNotIn(CHALLENGE["reference_solution"], result["text"])
            level = result["next_hint_level"]
        self.assertEqual(level, 3)

    def test_only_direct_positive_solution_request_reveals_current_solution(self):
        for question in ("show solution", "Please show me the full reference solution", "Can you show me the solution?"):
            result = self.answer(question)
            self.assertEqual(result["kind"], "solution")
            self.assertIn(CHALLENGE["reference_solution"], result["text"])
        for question in ("Do not show solution", "Don't show solution; explain division", "Why do people say show solution?", "show solution for another challenge", "review my code"):
            result = self.answer(question)
            self.assertNotIn(CHALLENGE["reference_solution"], result["text"])

    def test_reference_like_hint_is_not_leaked(self):
        challenge = copy.deepcopy(CHALLENGE)
        challenge["hints"][0] = challenge["reference_solution"]
        result = mentor_reply("hint", challenge, "")
        self.assertNotIn(challenge["reference_solution"], result["text"])

    def test_unknown_question_does_not_claim_knowledge(self):
        for question in ("What is the weather in Pune?", "Explain quaternion rotation of an object", "How does quantum chromodynamics renormalization work?"):
            result = self.answer(question)
            self.assertEqual(result["kind"], "unknown", question)
            self.assertEqual(result["sources"], [])
            self.assertIn("do not have a grounded reference card", result["text"])

    def test_bm25_retrieves_specific_primary_reference(self):
        cases = [
            ("Why does std::vector reserve not change size?", "https://eel.is/c++draft/vector"),
            ("Explain LLVM phi nodes and SSA", "https://llvm.org/docs/LangRef.html"),
            ("How does integer division truncate?", "https://eel.is/c++draft/expr.mul"),
            ("How does SDL_PollEvent process the event queue?", "https://wiki.libsdl.org/SDL3/SDL_PollEvent")]
        for question, expected in cases:
            result = self.answer(question)
            self.assertEqual(result["kind"], "concept")
            self.assertIn(expected, [source["url"] for source in result["sources"]], question)
        self.assertEqual(BM25().search("weather in Pune", track="games"), [])

    def test_level_and_track_adaptation(self):
        beginner = self.answer("Explain vector reserve", profile={"level": "beginner", "track": "games"})
        advanced = self.answer("Explain vector reserve", profile={"level": "advanced", "track": "systems"})
        self.assertIn("spawning", beginner["text"])
        self.assertIn("amortized", advanced["text"])
        self.assertNotEqual(beginner["text"], advanced["text"])

    def test_profile_difficulty_overrides_xp_level(self):
        advanced = self.answer("Where should I start?", profile={"difficulty": "advanced", "level": 1})
        beginner = self.answer("Where should I start?", profile={"difficulty": "beginner", "level": 12})
        self.assertIn("correctness invariant", advanced["text"])
        self.assertIn("Read one value", beginner["text"])

    def test_all_track_uses_challenge_context_and_all_difficulty_keeps_progress(self):
        result = self.answer("Where should I start?", profile={"difficulty": "all", "level": 8, "track": "all"})
        self.assertIn("correctness invariant", result["text"])
        self.assertIn("game track", result["text"])
        explicit = self.answer("Where should I start?", profile={"difficulty": "beginner", "track": "systems"})
        self.assertNotIn("game track", explicit["text"])

    def test_explicit_case_identifier_overrides_display_index(self):
        run = {"mode": "submit", "results": [{"index": 1, "test_index": 0,
                "status": "wrong_answer", "passed": False, "hidden": False, "actual": "3\n"}]}
        result = self.answer("Why did this test fail?", last_run=run)
        self.assertIn("7 3", result["text"])
        self.assertIn("Expected:", result["text"])
        self.assertNotIn("hidden test", result["text"])

    def test_explicit_private_identifier_cannot_be_overridden_by_false_hidden_flag(self):
        run = {"mode": "submit", "results": [{"index": 2, "test_index": 1,
                "status": "wrong_answer", "passed": False, "hidden": False, "actual": "SECRET-ACTUAL"}]}
        result = self.answer("Why did this test fail?", last_run=run)
        self.assertIn("hidden test", result["text"])
        for secret in ("987654", "141093", "SECRET-ACTUAL"):
            self.assertNotIn(secret, result["text"])

    def test_persisted_dashboard_display_index_uses_one_based_fallback(self):
        run = {"mode": "submit", "results": [{"index": 1,
                "status": "wrong_answer", "passed": False, "hidden": False, "actual": "3\n"}]}
        result = self.answer("Why did this test fail?", last_run=run)
        self.assertIn("7 3", result["text"])
        self.assertNotIn("hidden test", result["text"])

    def test_custom_run_does_not_borrow_challenge_expected_output(self):
        run = {"mode": "custom", "results": [{"index": 2, "test_index": 1,
                "status": "runtime_error", "passed": False, "hidden": False, "input": "provided input\n"}]}
        result = self.answer("Why did this test fail?", last_run=run)
        self.assertIn("provided input", result["text"])
        self.assertNotIn("hidden test", result["text"])
        self.assertNotIn("987654", result["text"])

    def test_stale_diagnostic_omits_current_source_excerpt(self):
        source = "int main() {\n  int current_revision_marker = 42;\n}\n"
        run = {"source_changed": True, "compile_status": "compile_error",
               "diagnostics": "main.cpp:2:4: error: expected ';' at end of declaration", "results": []}
        result = self.answer("Explain this error", source=source, last_run=run)
        self.assertTrue(result["text"].startswith("This result is from an earlier draft. Run your current code for fresh feedback."))
        self.assertIn("line 2", result["text"])
        self.assertNotIn("current_revision_marker", result["text"])
        self.assertNotIn("Reported source line", result["text"])

    def test_stale_test_and_progress_feedback_clearly_mark_previous_draft(self):
        runs = [
            {"source_changed": True, "results": [{"index": 0, "passed": False, "status": "wrong_answer", "actual": "3\n"}]},
            {"source_changed": True, "results": [{"index": 1, "passed": False, "status": "wrong_answer", "actual": "PRIVATE_NEW_ACTUAL"}]},
            {"source_changed": True, "results": [{"passed": True, "status": "passed"}]},
            {"source_changed": True, "compile_status": "unavailable", "diagnostics": "Runner unavailable", "results": []},
            {"source_changed": True, "results": [{"index": 0, "passed": False, "status": "time_limit"}]},
        ]
        for run in runs:
            with self.subTest(run=run):
                result = self.answer("Explain the last run results", last_run=run)
                self.assertTrue(result["text"].startswith("This result is from an earlier draft."))
                self.assertNotIn("PRIVATE_NEW_ACTUAL", result["text"])
                self.assertLessEqual(len(result["text"]), MAX_TEXT)
                json.dumps(result, allow_nan=False)

    def test_stale_run_does_not_replace_current_review_or_hint_context(self):
        run = {"source_changed": True, "diagnostics": "main.cpp:2:4: error: expected ';'", "results": []}
        review = self.answer("Review my code", source="int main() { int x = 0; if (x = 1) {} }", last_run=run)
        self.assertEqual(review["kind"], "review")
        self.assertIn("assign", review["text"])
        self.assertNotIn("earlier draft", review["text"])
        hint = self.answer("Give me a hint", last_run=run)
        self.assertEqual(hint["kind"], "hint")
        self.assertNotIn("earlier draft", hint["text"])

    def test_matching_source_keeps_diagnostic_excerpt_and_does_not_mutate_evidence(self):
        run = {"source_changed": False, "diagnostics": "main.cpp:2:4: error: expected ';'", "results": []}
        before = copy.deepcopy(run)
        result = self.answer("Explain this diagnostic", source="int main() {\n  int answer = 7\n}\n", last_run=run)
        self.assertIn("Reported source line", result["text"])
        self.assertIn("int answer = 7", result["text"])
        self.assertNotIn("earlier draft", result["text"])
        self.assertEqual(run, before)

    def test_structured_diagnostic_uses_observed_location(self):
        result = self.answer("Explain this error", source="int main() {\nint x = 1\nreturn x;\n}",
                             last_run={"diagnostics": [{"severity": "error", "line": 2, "column": 10, "message": "expected ';' at end of declaration"}]})
        self.assertEqual(result["kind"], "diagnostic")
        self.assertIn("line 2, column 10", result["text"])
        self.assertIn("int x = 1", result["text"])
        self.assertIn("preceding declaration", result["text"])
        self.assertIn("https://clang.llvm.org/docs/UsersManual.html", [source["url"] for source in result["sources"]])

    def test_actual_runner_diagnostics_string_schema(self):
        run = {"compile_status": "compile_error", "diagnostics": "<workspace>/main.cpp:3:4: error: use of undeclared identifier 'energy'\n", "results": []}
        result = self.answer("Why does compile fail?", last_run=run)
        self.assertEqual(result["kind"], "diagnostic")
        self.assertIn("line 3, column 4", result["text"])
        self.assertIn("not visible", result["text"])

    @unittest.skipUnless(shutil.which("clang++"), "Clang is not installed")
    def test_real_clang_diagnostic_is_explained(self):
        code = "int main() {\n  int answer = 2\n  return answer;\n}\n"
        process = subprocess.run([shutil.which("clang++"), "-std=c++20", "-fsyntax-only", "-x", "c++", "-"], input=code, text=True, capture_output=True, timeout=10)
        self.assertNotEqual(process.returncode, 0)
        result = self.answer("Explain the compiler error", source=code,
                             last_run={"compile_status": "compile_error", "diagnostics": process.stderr, "results": []})
        self.assertEqual(result["kind"], "diagnostic")
        self.assertIn("semicolon", result["text"])
        self.assertIn("line 2", result["text"])

    def test_failed_output_contains_actual_input_expected_and_actual(self):
        run = {"compile_status": "ok", "results": [{"index": 0, "status": "wrong_answer", "passed": False,
                "input": "7 3\n", "expected": "2\n", "actual": "3\n", "stdout": "3\n"}]}
        result = self.answer("Why is this test failing?", last_run=run)
        self.assertEqual(result["kind"], "test_feedback")
        for text in ("7 3", "Expected:", "Actual:", "number 1"):
            self.assertIn(text, result["text"])
        self.assertNotIn(CHALLENGE["reference_solution"], result["text"])

    def test_hidden_case_data_is_never_disclosed(self):
        run = {"results": [{"index": 1, "status": "wrong_answer", "passed": False, "input": "987654 7\n", "expected": "141093\n", "actual": "SECRET-ACTUAL"}]}
        result = self.answer("Why did the test fail?", last_run=run)
        self.assertIn("hidden test", result["text"])
        for secret in ("987654", "141093", "SECRET-ACTUAL"):
            self.assertNotIn(secret, result["text"])

    def test_timeout_and_other_limits_do_not_invent_output_mismatch(self):
        for status in ("time_limit", "memory_limit", "output_limit", "process_limit"):
            run = {"results": [{"index": 0, "status": status, "passed": False, "input": "7 3\n", "actual": ""}]}
            result = self.answer("Explain this failed test", last_run=run)
            self.assertEqual(result["kind"], "test_feedback")
            self.assertNotIn("output differs", result["text"])
            self.assertIn("limit", result["text"])

    def test_runtime_failure_uses_actual_stderr(self):
        run = {"results": [{"index": 0, "status": "runtime_error", "passed": False, "exit_code": -11,
                           "input": "7 3\n", "stderr": "AddressSanitizer: heap-buffer-overflow"}]}
        result = self.answer("Why did this crash?", last_run=run)
        self.assertIn("heap-buffer-overflow", result["text"])
        self.assertIn("https://clang.llvm.org/docs/AddressSanitizer.html", [source["url"] for source in result["sources"]])

    def test_warning_does_not_displace_wrong_answer_evidence(self):
        run = {"diagnostics": "main.cpp:1:1: warning: unused variable 'unused'\n",
               "results": [{"index": 0, "status": "wrong_answer", "passed": False, "actual": "3\n"}]}
        self.assertEqual(self.answer("Why is my output wrong?", last_run=run)["kind"], "test_feedback")
        self.assertEqual(self.answer("Explain the warning", last_run=run)["kind"], "diagnostic")

    def test_passed_results_are_reported_without_claiming_universal_correctness(self):
        run = {"compile_status": "ok", "results": [{"passed": True, "status": "passed"}, {"passed": True, "status": "passed"}]}
        result = self.answer("Did my tests pass?", last_run=run)
        self.assertEqual(result["kind"], "progress")
        self.assertIn("All 2", result["text"])
        self.assertIn("not every possible input", result["text"])

    def test_unavailable_compiler_is_not_an_algorithm_diagnosis(self):
        result = self.answer("Why did compile fail?", last_run={"compile_status": "unavailable", "diagnostics": "Local compiler unavailable", "results": []})
        self.assertEqual(result["kind"], "diagnostic")
        self.assertIn("unavailable", result["text"])
        self.assertNotIn("output differs", result["text"])

    def test_no_run_evidence_is_not_fabricated(self):
        result = self.answer("Why does this fail?")
        self.assertIn("do not have a compiler or test result", result["text"])

    def test_conservative_review_marks_patterns_as_uncertain(self):
        source = "int main() {\nint x;\nif (x = 1) { }\n}\n"
        result = self.answer("Review my code", source=source)
        self.assertEqual(result["kind"], "review")
        self.assertIn("not confirmed defects", result["text"])
        self.assertIn("assignment is intentional", result["text"])
        self.assertIn("does not prove an uninitialized read", result["text"])

    def test_comments_and_strings_do_not_trigger_code_review_patterns(self):
        source = 'int main() {\n// if (x = 1) int score;\nconst char* s = "new Thing; <= v.size()";\n}\n'
        result = self.answer("Review my code", source=source)
        self.assertNotIn("raw allocation appears", result["text"])
        self.assertNotIn("condition appears to assign", result["text"])
        self.assertIn("not a correctness verdict", result["text"])

    def test_challenge_concepts_and_formats_use_supplied_content(self):
        result = self.answer("Explain the key concept")
        self.assertIn(CHALLENGE["concepts"][0]["body"], result["text"])
        self.assertEqual(result["sources"], [])
        self.assertIn(CHALLENGE["input_format"], self.answer("Explain the input format")["text"])

    def test_sources_are_primary_https_and_runtime_is_bounded(self):
        hosts = {"eel.is", "isocpp.github.io", "llvm.org", "clang.llvm.org", "cmake.org", "www.sfml-dev.org", "wiki.libsdl.org", "www.raylib.com", "github.com"}
        for card in load_knowledge():
            for source in card["sources"]:
                parsed = urlparse(source["url"])
                self.assertEqual(parsed.scheme, "https")
                self.assertIn(parsed.hostname, hosts)
        result = self.answer("Why did this error happen?", last_run={"diagnostics": "main.cpp:1:1: error: " + "x" * 100000})
        self.assertLessEqual(len(result["text"]), MAX_TEXT)
        self.assertLessEqual(len(result["suggestions"]), 3)

    def test_quiz_and_beginner_orientation_are_actionable(self):
        quiz = self.answer("Quiz me on this concept")
        self.assertEqual(quiz["kind"], "question")
        self.assertIn("?", quiz["text"])
        orientation = self.answer("Teach me C++ from scratch")
        self.assertEqual(orientation["kind"], "question")
        self.assertIn("input and output", orientation["text"])
        self.assertIn("game track", orientation["text"])

    def test_long_output_feedback_identifies_excerpt_limits(self):
        result = self.answer("Why did this test fail?", last_run={"results": [{"index": 0, "status": "wrong_answer", "passed": False, "expected": "1 " * 1500, "actual": "2 " * 1500}]})
        self.assertIn("excerpt truncated", result["text"])
        self.assertIn("bounded output excerpts", result["text"])
        self.assertNotIn("Expected 600", result["text"])
        self.assertLessEqual(len(result["text"]), MAX_TEXT)

    def test_deterministic_retrieval_and_no_input_mutation(self):
        challenge, profile, run = copy.deepcopy(CHALLENGE), {"level": "beginner"}, {"results": []}
        before = json.dumps([challenge, profile, run], sort_keys=True)
        first = mentor_reply("Explain vectors", challenge, "", run, profile)
        second = mentor_reply("Explain vectors", challenge, "", run, profile)
        self.assertEqual(first, second)
        self.assertEqual(json.dumps([challenge, profile, run], sort_keys=True), before)


if __name__ == "__main__":
    unittest.main()
