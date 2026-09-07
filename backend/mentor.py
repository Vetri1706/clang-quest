"""A local, grounded study mentor: curated retrieval and evidence-based feedback.

No network requests, learned language model, external service, source execution,
or package dependencies are used at runtime. Compiler/test evidence is supplied
by the caller. Responses describe observed evidence separately from suggestions.
The API only returns reference_solution when the question explicitly asks to
show it. Hidden tests are not disclosed through feedback or hints.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse


MAX_TEXT = 6000
MAX_QUESTION = 2000
MAX_SOURCE = 65536
MODEL = {"mode": "grounded", "label": "Local study mentor"}
STALE_NOTICE = "This result is from an earlier draft. Run your current code for fresh feedback."
_STOP = set("a an and are as at be by can could do does explain for from give help how i in into is it me my of on or please program programming question should that the their this to using was what when where which why will with would you your c++ cpp code".split())
_GENERIC_ANCHORS = {"object", "objects", "memory", "time", "type", "types", "value", "values", "game", "games", "input", "output", "library", "resource", "state", "system", "systems", "basic"}
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PRIMARY_HOSTS = {"eel.is", "isocpp.github.io", "llvm.org", "clang.llvm.org", "cmake.org", "www.sfml-dev.org", "wiki.libsdl.org", "www.raylib.com", "github.com"}


def _text(value: Any, maximum: int = 1000) -> str:
    if value is None:
        return ""
    if not isinstance(value, (str, int, float, bool)):
        return ""
    return _ANSI.sub("", str(value)).replace("\x00", "")[:maximum]


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z_][a-z0-9_:+]*|\d+", text.lower())
    expanded = []
    for word in words:
        if word not in _STOP:
            expanded.append(word)
        if "::" in word:
            expanded.extend(part for part in word.split("::") if part and part not in _STOP and part != "std")
    return expanded


@lru_cache(maxsize=1)
def load_knowledge() -> tuple[dict, ...]:
    path = Path(__file__).with_name("knowledge.json")
    if path.stat().st_size > 262144:
        raise ValueError("Mentor knowledge exceeds its local byte budget")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "local-study-cards" or payload.get("version") != 1:
        raise ValueError("Unsupported local mentor knowledge format")
    cards = payload.get("cards")
    if not isinstance(cards, list) or not 18 <= len(cards) <= 24:
        raise ValueError("Mentor needs 18..24 bounded reference cards")
    seen = set()
    for card in cards:
        if not isinstance(card, dict) or card.get("id") in seen:
            raise ValueError("Invalid or duplicate mentor card")
        seen.add(card["id"])
        for field in ("id", "title", "body", "advanced", "question"):
            if not isinstance(card.get(field), str) or not card[field] or len(card[field]) > 1600:
                raise ValueError("Mentor card text is missing or unbounded")
        for field in ("tags", "tracks", "sources"):
            if not isinstance(card.get(field), list) or not card[field]:
                raise ValueError("Mentor card metadata is missing")
        for source in card["sources"]:
            parsed = urlparse(source["url"])
            if parsed.scheme != "https" or parsed.hostname not in _PRIMARY_HOSTS or not isinstance(source.get("title"), str):
                raise ValueError("Mentor sources must use the curated primary-reference hosts")
    return tuple(cards)


class BM25:
    """Small deterministic Okapi BM25 index with an explicit topic-match gate."""
    def __init__(self, cards: tuple[dict, ...] | None = None, k1: float = 1.5, b: float = 0.75):
        self.cards = cards or load_knowledge()
        self.k1, self.b = k1, b
        self.documents = []
        self.anchors = []
        for card in self.cards:
            anchors = _tokens(card["title"] + " " + " ".join(card["tags"]))
            document = anchors * 3 + _tokens(card["body"] + " " + card["advanced"])
            self.documents.append(Counter(document))
            self.anchors.append(set(anchors))
        self.lengths = [sum(document.values()) for document in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths)
        frequency = Counter(token for document in self.documents for token in document)
        count = len(self.documents)
        self.idf = {token: math.log(1 + (count - df + 0.5) / (df + 0.5)) for token, df in frequency.items()}

    def search(self, query: str, track: str = "", limit: int = 2) -> list[dict]:
        terms = set(_tokens(query))
        matches = []
        for index, document in enumerate(self.documents):
            overlap = terms & self.anchors[index]
            # A track preference never manufactures relevance for an unknown query.
            if not overlap or (overlap <= _GENERIC_ANCHORS and len(terms - overlap) >= 2):
                continue
            score = 0.0
            for token in terms:
                frequency = document.get(token, 0)
                if frequency:
                    denominator = frequency + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                    score += self.idf[token] * frequency * (self.k1 + 1) / denominator
            if track in self.cards[index]["tracks"]:
                score *= 1.08
            matches.append((score, self.cards[index]["id"], self.cards[index]))
        matches.sort(key=lambda item: (-item[0], item[1]))
        return [card for _, _, card in matches[:max(0, min(limit, 3))]]


@lru_cache(maxsize=1)
def _index() -> BM25:
    return BM25()


def _track(challenge: dict, profile: dict) -> str:
    selected = _text(profile.get("track"), 80).strip().lower()
    track = selected if selected and selected != "all" else _text(challenge.get("track"), 80).strip().lower()
    if any(word in track for word in ("game", "graphics", "engine")):
        return "games"
    if any(word in track for word in ("llvm", "compiler")):
        return "compilers"
    if any(word in track for word in ("system", "performance", "concurr")):
        return "systems"
    return "fundamentals"


def _level(profile: dict, challenge: dict) -> str:
    selected = _text(profile.get("difficulty"), 80).strip().lower()
    if selected in {"beginner", "intermediate", "advanced"}:
        return selected
    value = profile.get("skill_level", profile.get("level", challenge.get("difficulty", "beginner")))
    if isinstance(value, (int, float)):
        return "advanced" if value >= 8 else "intermediate" if value >= 4 else "beginner"
    value = _text(value, 80).lower()
    if any(word in value for word in ("advanced", "expert", "hard", "senior")):
        return "advanced"
    return "intermediate" if any(word in value for word in ("intermediate", "medium")) else "beginner"


def _response(text: str, kind: str, cards: list[dict] | None = None, suggestions: list[str] | None = None,
              next_hint_level: int = 0, sources: list[dict] | None = None) -> dict:
    references, seen = [], set()
    candidates = list(sources or []) + [source for card in cards or [] for source in card["sources"]]
    for source in candidates:
        if source["url"] not in seen:
            references.append({"title": source["title"], "url": source["url"]})
            seen.add(source["url"])
    return {"text": text[:MAX_TEXT], "kind": kind, "sources": references[:4],
            "suggestions": [_text(item, 180) for item in (suggestions or [])[:3]],
            "next_hint_level": max(0, min(3, next_hint_level)), "model": dict(MODEL)}


def _run_response(reply: dict, last_run: dict) -> dict:
    if last_run.get("source_changed") is True:
        return {**reply, "text": (STALE_NOTICE + "\n\n" + reply["text"])[:MAX_TEXT]}
    return reply


def _card(card_id: str) -> dict:
    return next(card for card in load_knowledge() if card["id"] == card_id)


def _context_query(challenge: dict) -> str:
    fields = [_text(challenge.get("title")), _text(challenge.get("summary"))]
    fields.extend(_text(tag, 100) for tag in challenge.get("tags", []) if isinstance(tag, str))
    for concept in challenge.get("concepts", [])[:8]:
        if isinstance(concept, dict):
            fields.append(_text(concept.get("title"), 150))
    return " ".join(fields)


def _explicit_solution(question: str) -> bool:
    # Only a direct positive request authorizes revealing the current challenge.
    return bool(re.fullmatch(
        r"\s*(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?(?:show|reveal|give)(?:\s+me)?\s+(?:the\s+)?(?:full\s+)?(?:reference\s+)?solution(?:\s+(?:for|to)\s+(?:this|the current)\s+(?:challenge|quest|problem))?\s*(?:please)?[.!?]*\s*",
        question, re.IGNORECASE))


def _fenced(value: str, language: str = "text", maximum: int = 700) -> str:
    original_length = len(value) if isinstance(value, str) else 0
    value = _text(value, maximum)
    if original_length > maximum:
        value += "\n… [excerpt truncated]"
    largest = max((len(run) for run in re.findall(r"`+", value)), default=0)
    fence = "`" * max(3, largest + 1)
    return f"{fence}{language}\n{value}\n{fence}"


def _hint(challenge: dict, stage: int, level: str, track: str) -> dict:
    hints = [item.strip() for item in challenge.get("hints", [])[:3] if isinstance(item, str) and item.strip()]
    reference = _text(challenge.get("reference_solution"), MAX_SOURCE).strip()
    hints = [hint for hint in hints if not reference or len(reference) < 30 or reference not in hint]
    objectives = [_text(item, 250) for item in challenge.get("objectives", [])[:3]]
    fallback = [
        "Start with the input and output contract. " + (objectives[0] if objectives else "Name the value you need to compute."),
        "Trace one small permitted input by hand. Write the value of each changing variable after every step.",
        "Check the smallest allowed input and the boundary cases. Compare the exact output format with the specification."
    ]
    hints += fallback[len(hints):]
    index = min(stage, 2)
    prefix = f"Hint {index + 1} of 3"
    if stage >= 3:
        prefix = "You have reached the final hint"
    follow = "What would you try first?" if level == "beginner" else "Which invariant or boundary case would verify that step?"
    if track == "games":
        follow = "Which game-state value changes at this step, and what should remain unchanged?"
    cards = _index().search(_context_query(challenge), track, limit=1)
    return _response(f"{prefix}: {hints[index]}\n\n{follow}", "hint", cards,
                     ["Check my code", "Explain the key concept", "Give me the next hint"], min(3, stage + 1))


def _diagnostic(run: dict) -> tuple[str, int | None, int | None, str] | None:
    diagnostics = run.get("diagnostics", [])
    if isinstance(diagnostics, list):
        for item in sorted(diagnostics[:20], key=lambda entry: isinstance(entry, dict) and entry.get("severity") == "warning"):
            if isinstance(item, dict) and item.get("severity", "error") in ("error", "fatal error", "warning"):
                message = _text(item.get("message"), 1000)
                if message:
                    return message, item.get("line"), item.get("column"), item.get("severity", "error")
    compile_result = run.get("compile", {})
    if not isinstance(compile_result, dict):
        compile_result = {}
    raw = _text((diagnostics if isinstance(diagnostics, str) else "") or compile_result.get("stderr") or run.get("compiler_stderr") or run.get("compile_stderr") or run.get("stderr"), 8000)
    pattern = r"(?:^|\n)[^\n]*?:(\d+):(\d+):\s*(fatal error|error|warning):\s*([^\n]+)"
    matches = list(re.finditer(pattern, raw))
    for match in matches:
        if match.group(3) != "warning":
            return match.group(4), int(match.group(1)), int(match.group(2)), match.group(3)
    if matches:
        first = matches[0]
        return first.group(4), int(first.group(1)), int(first.group(2)), first.group(3)
    if any(token in raw.lower() for token in ("undefined reference", "undefined symbols", "linker command failed")):
        return raw.splitlines()[0], None, None, "linker error"
    generic = re.search(r"(?:fatal error|error):\s*([^\n]+)", raw)
    if generic:
        return generic.group(1), None, None, "error"
    return None


def _diagnostic_reply(diagnostic, source: str, stage: int) -> dict:
    message, line, column, severity = diagnostic
    lowered = message.lower()
    explanation = "Read this diagnostic together with any following compiler notes. Inspect the reported declaration and the line immediately before the location; the visible message alone may not identify the complete cause."
    suggestions = ["Fix the first reported issue, then compile again", "Explain the relevant concept"]
    if "expected ';'" in lowered or "expected semicolon" in lowered:
        explanation = "Clang expected a semicolon. Check whether the preceding declaration or statement is complete; the reported location can be after the actual omission."
    elif "undeclared identifier" in lowered or "not declared in this scope" in lowered:
        explanation = "This name is not visible at this use. Check its spelling, declaration order, scope, and whether a library name needs std:: qualification and its proper header."
    elif "no matching" in lowered or "candidate function" in lowered:
        explanation = "The call does not match an available overload. Compare the argument count and argument types with the function declaration and the compiler's candidate notes."
    elif "file not found" in lowered:
        explanation = "The compiler could not locate an included file. Check the header spelling and whether that dependency is available to this local build; changing the algorithm will not fix a missing header."
    elif "cannot initialize" in lowered or "no viable conversion" in lowered or "incompatible" in lowered:
        explanation = "The value's type cannot be used in the requested initialization or conversion. Compare the source type with the destination type before adding a cast."
    elif "non-void" in lowered and "return" in lowered:
        explanation = "Check whether every required path in this value-returning function produces the promised result. Reaching the end of main is a special case; do not generalize it to other functions."
    elif "uninitialized" in lowered:
        explanation = "There may be a path that reads this value before assigning it. Trace the branches leading to the read and give the value a meaningful initial state where required."
    elif "comparison" in lowered and ("signed" in lowered or "unsigned" in lowered):
        explanation = "The comparison mixes signed and unsigned types. Check for negative values and conversion effects, especially when comparing an index with a container's size."
    elif "division by zero" in lowered:
        explanation = "The divisor is zero in the reported expression. Identify which permitted inputs can reach this division and handle the denominator before evaluating it."
    elif "undefined reference" in lowered or "undefined symbols" in lowered or severity == "linker error":
        explanation = "A declaration was available, but linking could not find a required definition. Check the missing symbol, its matching definition, and the source files or libraries passed to the linker."
    location = ""
    if type(line) is int and line > 0:
        location = f" at line {line}"
        if type(column) is int and column > 0:
            location += f", column {column}"
    excerpt = ""
    lines = source.splitlines()
    if type(line) is int and 1 <= line <= len(lines):
        excerpt = "\n\nReported source line:\n" + _fenced(lines[line - 1], "cpp", 350)
    text = f"Clang reported {severity}{location}:\n{_fenced(message, maximum=1000)}\n\n{explanation}{excerpt}"
    return _response(text, "diagnostic", [_card("clang-diagnostics")], suggestions, stage)


def _failed_test(run: dict, challenge: dict) -> tuple[dict, bool] | None:
    results = run.get("tests", run.get("results", run.get("test_results", [])))
    if not isinstance(results, list):
        return None
    challenge_tests = challenge.get("tests", [])
    for index, result in enumerate(results[:100]):
        if not isinstance(result, dict):
            continue
        status = _text(result.get("status"), 80).lower()
        failed = result.get("passed") is False or result.get("ok") is False or status in {"failed", "wrong_answer", "wrong answer", "timeout", "time_limit", "output_limit", "memory_limit", "process_limit", "runtime_error", "error"}
        if not failed:
            continue
        # test_index is the zero-based source case identifier. Dashboard index
        # is one-based display text; historical runner-only records used a
        # zero-based index and have no dashboard mode. Keep both contracts.
        if "test_index" in result:
            test_index = result["test_index"]
        elif run.get("mode") in {"run", "submit", "custom"} and type(result.get("index")) is int:
            test_index = result["index"] - 1
        else:
            test_index = result.get("index", index)
        expected = challenge_tests[test_index] if run.get("mode") != "custom" and type(test_index) is int and 0 <= test_index < len(challenge_tests) and isinstance(challenge_tests[test_index], dict) else {}
        hidden = bool(result.get("hidden", False) or expected.get("hidden", False))
        merged = {"input": expected.get("input", ""), "expected": expected.get("output", ""), **result}
        if "expected_output" in result:
            merged["expected"] = result["expected_output"]
        if "actual_output" in result:
            merged["actual"] = result["actual_output"]
        elif "stdout" in result:
            merged["actual"] = result["stdout"]
        return merged, hidden
    return None


def _test_reply(result: dict, hidden: bool, stage: int) -> dict:
    status = _text(result.get("status"), 80).lower()
    if hidden:
        return _response("A hidden test failed. Its private input and expected output stay hidden. Check the smallest allowed input, equality boundaries, empty cases when allowed, and the largest intermediate arithmetic value.\n\nWhich boundary has not appeared in your own tests yet?",
                         "test_feedback", suggestions=["Give me a hint", "Review my boundary checks", "Explain integer overflow"], next_hint_level=stage)
    if status in {"timeout", "time_limit"} or result.get("timed_out"):
        return _response("This test exceeded the runner's time limit. That does not by itself prove an infinite loop. Check that each iteration makes progress and estimate how the work grows with the input size.\n\nInput:\n" + _fenced(result.get("input", "")),
                         "test_feedback", [_card("complexity")], ["Review my loop", "Explain time complexity"], stage)
    if status in {"output_limit", "memory_limit", "process_limit"}:
        descriptions = {"output_limit": "The runner stopped this test after its output limit was reached. Check repeated printing and whether your loop terminates.", "memory_limit": "The runner stopped this test after its memory limit was reached. Check allocation sizes and whether stored data grows each iteration.", "process_limit": "The runner stopped this test after its process limit was reached. This challenge expects one bounded local program."}
        return _response(descriptions[status] + "\n\nWhich operation can keep increasing this resource?", "test_feedback", suggestions=["Review my code", "Give me a hint"], next_hint_level=stage)
    stderr = _text(result.get("stderr"), 1000)
    if status == "runtime_error" or (type(result.get("returncode", result.get("exit_code"))) is int and result.get("returncode", result.get("exit_code")) != 0):
        description = "The compiled program exited unsuccessfully on this test."
        if "addresssanitizer" in stderr.lower() or "runtime error:" in stderr.lower():
            description += " The runtime diagnostic is evidence about this particular execution."
        return _response(description + "\n\nInput:\n" + _fenced(result.get("input", "")) + ("\n\nDiagnostic:\n" + _fenced(stderr) if stderr else "") + "\n\nWhich access, division, or lifetime assumption does this input exercise?",
                         "test_feedback", [_card("sanitizers")], ["Review my bounds", "Explain the diagnostic"], stage)
    expected, actual = _text(result.get("expected"), 1200), _text(result.get("actual", result.get("output", "")), 1200)
    text = "This test's output differs from its expected output.\n\nInput:\n" + _fenced(result.get("input", ""))
    text += "\n\nExpected:\n" + _fenced(expected) + "\n\nActual:\n" + _fenced(actual)
    raw_expected = result.get("expected", "")
    raw_actual = result.get("actual", result.get("output", ""))
    clipped = (isinstance(raw_expected, str) and len(raw_expected) > 1200) or (isinstance(raw_actual, str) and len(raw_actual) > 1200)
    if clipped:
        text += "\n\nThese are bounded output excerpts. Inspect the complete runner output before drawing conclusions about the first differing value or total value count."
    elif expected != actual and expected.split() == actual.split():
        text += "\n\nThe visible values match after splitting whitespace. Check the runner's exact comparison rule and inspect line breaks or extra spaces."
    else:
        expected_tokens, actual_tokens = expected.split(), actual.split()
        differing = next((i for i, pair in enumerate(zip(expected_tokens, actual_tokens)) if pair[0] != pair[1]), None)
        if differing is not None:
            text += f"\n\nThe first differing whitespace-delimited value is number {differing + 1}. Trace the calculation that produces it."
        elif len(expected_tokens) != len(actual_tokens):
            text += f"\n\nExpected {len(expected_tokens)} whitespace-delimited values but received {len(actual_tokens)}. Check loop counts and output labels."
    text += "\n\nWhat intermediate value first differs when you trace this exact input by hand?"
    return _response(text, "test_feedback", suggestions=["Give me a hint", "Review my code", "Explain the key concept"], next_hint_level=stage)


def _strip_comments_literals(source: str) -> str:
    # Replacing with spaces preserves lines and columns. This is a conservative
    # lexical screen, not a C++ parser or proof about preprocessor expansion.
    pattern = r'R"([^ ()\\\t\r\n]{0,16})\([\s\S]*?\)\1"|//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(pattern, lambda match: re.sub(r"[^\n]", " ", match.group(0)), source)


def _review(source: str, challenge: dict, stage: int, level: str) -> dict:
    if not source.strip():
        return _response("The editor is empty. Read the input format, then write the smallest input-reading step and compile it before adding the calculation.",
                         "review", [_card("streams")], ["Explain the input format", "Give me a hint"], stage)
    clean = _strip_comments_literals(source)
    findings, cards = [], []
    checks = [
        (r"\bif\s*\([^\n;)]*(?<![=!<>])=(?!=)[^\n;)]*\)", "A condition appears to assign a value. Confirm whether assignment is intentional or whether you meant an equality comparison.", "branches"),
        (r"\b(?:int|long|float|double)\s+[A-Za-z_]\w*\s*;", "A scalar declaration has no visible initializer. Check that every path assigns it before its first read; this pattern alone does not prove an uninitialized read.", "numeric-types"),
        (r"<=\s*[A-Za-z_]\w*\s*\.\s*size\s*\(\s*\)", "A loop or condition includes a container's size as an upper boundary. If the same value is used as an index, verify that you never access element size().", "vector"),
        (r"\bnew\s+[A-Za-z_]", "A raw allocation appears in this code. Identify its owner and cleanup path; a standard container or ownership handle may make that responsibility clearer.", "raii"),
        (r"\b(?:float|double)\s+\w+\s*=\s*\d+\s*/\s*\d+\s*;", "A floating-point variable is initialized from an integer-literal division. The division happens before the conversion; check whether a fractional result was intended.", "division")
    ]
    for pattern, message, card_id in checks:
        match = re.search(pattern, clean)
        if match:
            line = clean.count("\n", 0, match.start()) + 1
            findings.append(f"Line {line}: {message}")
            cards.append(_card(card_id))
        if len(findings) == 3:
            break
    if findings:
        text = "These are checks suggested by the visible source, not confirmed defects:\n\n" + "\n\n".join(findings)
    else:
        text = "I do not see a specific issue covered by these conservative source checks. That is not a correctness verdict. Compile the current code and test it against the challenge's input/output contract."
    follow = "Choose one permitted input and trace each variable." if level == "beginner" else "State the invariant and a boundary case that could falsify it."
    return _response(text + "\n\n" + follow, "review", cards, ["Run my code", "Explain the key concept", "Give me a hint"], stage)


def mentor_reply(question: str, challenge: dict, source: str, last_run: dict | None = None,
                 profile: dict | None = None, hint_level: int = 0) -> dict:
    """Return one bounded, JSON-serializable grounded mentor message."""
    if not isinstance(challenge, dict):
        raise TypeError("challenge must be a dictionary")
    question, source = _text(question, MAX_QUESTION).strip(), _text(source, MAX_SOURCE)
    profile = profile if isinstance(profile, dict) else {}
    last_run = last_run if isinstance(last_run, dict) else {}
    try:
        stage = max(0, min(3, int(hint_level)))
    except (TypeError, ValueError, OverflowError):
        stage = 0
    level, track = _level(profile, challenge), _track(challenge, profile)
    lowered = question.lower()
    if _explicit_solution(question):
        solution = _text(challenge.get("reference_solution"), MAX_SOURCE).strip()
        if not solution:
            return _response("This challenge has no reference solution available to show. We can work through its hints and observed test results.", "solution", suggestions=["Give me a hint"], next_hint_level=stage)
        if len(solution) > MAX_TEXT - 700:
            return _response("The reference solution is too long for this bounded mentor reply. Open this challenge's solution panel to inspect the complete source.", "solution", next_hint_level=stage)
        return _response("You explicitly requested this challenge's reference solution.\n\n" + _fenced(solution, "cpp", MAX_TEXT - 700) + "\n\nBefore reusing it, explain how it meets each objective and test its boundary cases.",
                         "solution", suggestions=["Explain the key concept", "Review my version"], next_hint_level=stage)
    wants_hint = bool(re.search(r"\bhint\b|\bstuck\b|\bnext step\b|^help[.!?]*$", lowered)) or not question
    if wants_hint:
        return _hint(challenge, stage, level, track)
    wants_debug = bool(re.search(r"\berror\b|\bwarning\b|\bdiagnostic\b|\bcompile\b|\bfail(?:ed|ing|s)?\b|\bwrong\b|\bcrash\b|\bdebug\b|\btests?\b|\bfix\b|\blast run\b|\brun results?\b|\bcheck (?:my )?output\b", lowered))
    diagnostic = _diagnostic(last_run)
    failure = _failed_test(last_run, challenge)
    observed_source = "" if last_run.get("source_changed") is True else source
    if wants_debug and diagnostic and (diagnostic[3] != "warning" or re.search(r"warning|compil|diagnostic", lowered)):
        return _run_response(_diagnostic_reply(diagnostic, observed_source, stage), last_run)
    if wants_debug and failure:
        return _run_response(_test_reply(*failure, stage), last_run)
    if wants_debug and diagnostic:
        return _run_response(_diagnostic_reply(diagnostic, observed_source, stage), last_run)
    if wants_debug and last_run.get("compile_status") in {"unavailable", "time_limit", "memory_limit", "output_limit", "process_limit", "compile_error"}:
        status = last_run["compile_status"]
        details = _text(last_run.get("diagnostics"), 1200)
        return _run_response(_response("The local compilation runner reported " + status.replace("_", " ") + ". This is the recorded execution status; it does not establish an algorithmic mistake." + ("\n\nRecorded details:\n" + _fenced(details) if details else ""),
                         "diagnostic", suggestions=["Review my code", "Give me a hint"], next_hint_level=stage), last_run)
    reported = last_run.get("results", last_run.get("tests", last_run.get("test_results", [])))
    if wants_debug and isinstance(reported, list) and reported and all(isinstance(item, dict) and (item.get("passed") is True or item.get("status") == "passed") for item in reported):
        return _run_response(_response(f"All {len(reported)} tests in the last recorded run passed. This confirms those tested executions, not every possible input.\n\nCan you explain why the algorithm also covers the challenge's boundary cases?",
                         "progress", suggestions=["Review my code", "Quiz me on this concept"], next_hint_level=stage), last_run)
    if re.search(r"\breview\b|\bcheck my\b|\blook at my\b", lowered):
        return _review(source, challenge, stage, level)
    if wants_debug and not last_run:
        return _response("I do not have a compiler or test result for this code yet. Run the current source, then I can explain its actual diagnostic or compare a failed test's input, expected output, and actual output.\n\nWhich small input would you test first?",
                         "question", suggestions=["Run my code", "Review my code", "Give me a hint"], next_hint_level=stage)
    if re.search(r"\b(?:key concept|this challenge|this quest|input format|output format|objective)\b", lowered):
        concepts = [concept for concept in challenge.get("concepts", [])[:3] if isinstance(concept, dict)]
        if "input format" in lowered or "output format" in lowered:
            field = "input_format" if "input format" in lowered else "output_format"
            text = _text(challenge.get(field), 1600) or "The challenge has not supplied that format."
        elif concepts:
            text = "\n\n".join(_text(concept.get("title"), 150) + ": " + _text(concept.get("body"), 700) for concept in concepts)
        else:
            text = _text(challenge.get("summary"), 1200) or "Start by naming the input, required result, and allowed constraints."
        return _response(text + "\n\nWhich part can you explain back in your own words?", "concept", suggestions=["Give me a hint", "Review my code"], next_hint_level=stage)
    if re.search(r"\bquiz\b|\bask me\b|\bpractice\b", lowered):
        cards = _index().search(_context_query(challenge), track, limit=1)
        prompt = cards[0]["question"] if cards else "What are the inputs, the required output, and one edge case for this challenge?"
        return _response(prompt + "\n\nAnswer in a sentence first; then test the idea with one small example.", "question", cards, ["Give me a hint", "Explain the key concept"], stage)
    if re.search(r"\bteach me\b|\bwhere (?:do|should) i start\b|\blearn c\+\+|\bfrom scratch\b", lowered):
        if level == "advanced":
            plan = "Start by stating a correctness invariant, deriving the complexity from the constraints, and checking ownership and boundary behavior. Then inspect compiler output only for a concrete question."
        else:
            plan = "Start with this challenge's input and output. Read one value, compile that small step, then add the calculation and test one example. Build toward conditions, loops, functions, and containers in that order."
        if track == "games":
            plan += " For the game track, connect each new idea to a small state update such as health, position, or inventory."
        return _response(plan + "\n\nWhat information arrives as input, and what single result should this challenge produce?", "question", suggestions=["Explain the input format", "Give me a hint", "Explain the key concept"], next_hint_level=stage)
    cards = _index().search(question, track, limit=1 if level == "beginner" else 2)
    if cards:
        paragraphs = []
        for card in cards:
            paragraph = card["title"] + ": " + card["body"]
            if level == "advanced":
                paragraph += "\n\n" + card["advanced"]
            paragraphs.append(paragraph)
        selected = cards[0]
        socratic = selected.get("game_question", selected["question"]) if track == "games" else selected["question"]
        return _response("\n\n".join(paragraphs) + "\n\n" + socratic, "concept", cards,
                         ["Explain the key concept", "Give me a hint", "Review my code"], stage)
    return _response("I do not have a grounded reference card that answers that specific question. I can help with this challenge, actual Clang diagnostics, test differences, and the C++, LLVM, systems, and game-development topics in the local study cards.\n\nTry naming a concrete concept or paste the relevant diagnostic.",
                     "unknown", suggestions=["Explain the key concept", "Give me a hint", "Review my code"], next_hint_level=stage)
