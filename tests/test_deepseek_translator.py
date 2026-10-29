from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock

from pdf2zh.cache import TranslationCache
from pdf2zh.translator import (
    DeepSeekTranslator,
    FormulaPlaceholderError,
    OutageBackoff,
    RateLimitedError,
    ServiceUnavailableError,
    restore_spaced_acronyms,
)


class FakeClock:
    """Advance instantly through any outage pause the translator asks for."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeCache:
    def __init__(self, entries: dict[str, str] | None = None) -> None:
        self.entries = dict(entries or {})
        self.params: dict[str, object] = {}

    def get(self, text: str) -> str | None:
        return self.entries.get(text)

    def set(self, text: str, translated: str) -> None:
        self.entries[text] = translated

    def update_params(self, params: dict) -> None:
        self.params.update(params)


class FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.payload = payload
        self.status_code = status

    def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self, answer) -> None:
        self.answer = answer
        self.sent: list[dict] = []

    def post(self, _endpoint, json=None, headers=None, timeout=None):
        self.sent.append(json)
        return self.answer(json)


def chat(answer: str, finish: str = "stop") -> dict:
    return {"choices": [{"message": {"content": answer}, "finish_reason": finish}]}


def translations(items: list[dict]) -> dict:
    return chat(json.dumps({"translations": items}))


def echo_answer(payload: dict) -> FakeResponse:
    """Translate a batch by prefixing each segment, keeping its id."""
    segments = json.loads(payload["messages"][1]["content"])["segments"]
    return FakeResponse(
        translations(
            [{"id": segment["id"], "text": f"vi:{segment['text']}"} for segment in segments]
        )
    )


def translator_answering(answer=echo_answer, context: dict | None = None) -> DeepSeekTranslator:
    translator = DeepSeekTranslator(
        "en",
        "vi",
        "deepseek-chat",
        envs={"deepseek_api_key": "test-key", "document_context": context or {}},
    )
    clock = FakeClock()
    translator.session = FakeSession(answer)
    translator.sleep = clock.sleep
    translator.outage = OutageBackoff(clock=clock, sleep=clock.sleep)
    translator.cache = FakeCache()
    return translator


class SpacedAcronymTests(unittest.TestCase):
    def test_an_acronym_the_source_spells_is_glued_back_together(self):
        self.assertEqual(
            restore_spaced_acronyms("RCAS Announcements pdf", "Thông báo R C AS pdf"),
            "Thông báo RCAS pdf",
        )

    def test_letters_mixed_with_the_acronym_are_kept(self):
        self.assertEqual(
            restore_spaced_acronyms("RCAS report", "Báo cáo R C AS"),
            "Báo cáo RCAS",
        )

    def test_letters_not_spelled_in_the_source_are_never_joined(self):
        self.assertEqual(restore_spaced_acronyms("Announcements", "R C AS"), "R C AS")

    def test_the_repair_reaches_the_answer_and_the_cache(self):
        def spaces_the_acronym(payload: dict) -> FakeResponse:
            segments = json.loads(payload["messages"][1]["content"])["segments"]
            items = [
                {"id": segment["id"], "text": f"vi:{segment['text'].replace('RCAS', 'R C AS')}"}
                for segment in segments
            ]
            return FakeResponse(translations(items))

        translator = translator_answering(spaces_the_acronym)
        answers = translator.translate_many(["RCAS Announcements pdf"])
        self.assertEqual(answers, ["vi:RCAS Announcements pdf"])
        self.assertEqual(translator.cache.get("RCAS Announcements pdf"), "vi:RCAS Announcements pdf")


class DeepSeekCliTests(unittest.TestCase):
    def test_the_engine_is_offered_with_its_key_and_model(self):
        from scripts.translate_pdf import DEEPSEEK_MODEL, ENGINES, _parser

        self.assertIn("deepseek", ENGINES)
        args = _parser().parse_args(
            ["book.pdf", "--output-dir", "out", "--engine", "deepseek",
             "--deepseek-api-key", "sk-test"]
        )
        self.assertEqual(args.engine, "deepseek")
        self.assertEqual(args.deepseek_api_key, "sk-test")
        self.assertEqual(args.deepseek_model, DEEPSEEK_MODEL)


class DeepSeekBatchTests(unittest.TestCase):
    def test_a_page_of_segments_is_one_request(self):
        translator = translator_answering()
        segments = [f"Paragraph {number} of the page." for number in range(30)]
        answers = translator.translate_many(segments)
        self.assertEqual(answers, [f"vi:{segment}" for segment in segments])
        self.assertEqual(len(translator.session.sent), 1)

    def test_formula_and_style_tags_travel_inside_their_segment(self):
        translator = translator_answering()
        segments = ["Energy <b0></b0> is conserved.", "<s1>Bold</s1> heading"]
        self.assertEqual(
            translator.translate_many(segments),
            ["vi:Energy <b0></b0> is conserved.", "vi:<s1>Bold</s1> heading"],
        )

    def test_answers_come_back_in_the_requested_order_by_id(self):
        def reversed_answer(payload: dict) -> FakeResponse:
            segments = json.loads(payload["messages"][1]["content"])["segments"]
            items = [
                {"id": segment["id"], "text": f"vi:{segment['text']}"}
                for segment in reversed(segments)
            ]
            return FakeResponse(translations(items))

        translator = translator_answering(reversed_answer)
        segments = ["First.", "Second.", "Third."]
        self.assertEqual(translator.translate_many(segments), [f"vi:{text}" for text in segments])

    def test_a_damaged_formula_marker_keeps_only_its_own_segment(self):
        def drops_a_tag(payload: dict) -> FakeResponse:
            segments = json.loads(payload["messages"][1]["content"])["segments"]
            items = []
            for segment in segments:
                text = segment["text"].replace("<b0></b0>", "")
                items.append({"id": segment["id"], "text": f"vi:{text}"})
            return FakeResponse(translations(items))

        translator = translator_answering(drops_a_tag)
        answers = translator.translate_many(["Energy <b0></b0> is conserved.", "Plain."])
        self.assertIsInstance(answers[0], FormulaPlaceholderError)
        self.assertEqual(answers[1], "vi:Plain.")

    def test_an_unusable_batch_is_split_until_the_bad_segment_stands_alone(self):
        def rejects_the_run(payload: dict) -> FakeResponse:
            segments = json.loads(payload["messages"][1]["content"])["segments"]
            if any(segment["text"] == "bad" for segment in segments):
                return FakeResponse(translations([]))
            return echo_answer(payload)

        translator = translator_answering(rejects_the_run)
        segments = [f"Sentence {number}." for number in range(8)]
        segments[3] = "bad"
        answers = translator.translate_many(segments)
        for answer, segment in zip(answers, segments):
            if segment == "bad":
                self.assertNotIsInstance(answer, str)
            else:
                self.assertEqual(answer, f"vi:{segment}")
        self.assertGreater(len(translator.session.sent), 1)

    def test_a_key_refusal_answers_every_segment_and_stops_asking(self):
        def refuses(_payload: dict) -> FakeResponse:
            return FakeResponse({"error": {"message": "invalid key"}}, status=401)

        translator = translator_answering(refuses)
        answers = translator.translate_many(["One.", "Two."])
        self.assertTrue(all(isinstance(answer, ServiceUnavailableError) for answer in answers))
        self.assertIn("rejected the API key", str(answers[0]))
        requests = len(translator.session.sent)
        translator.translate_many(["Three."])
        self.assertEqual(len(translator.session.sent), requests)

    def test_a_rate_limit_is_not_retried_per_segment(self):
        def limits(_payload: dict) -> FakeResponse:
            return FakeResponse({"error": {"message": "slow down"}}, status=429)

        translator = translator_answering(limits)
        answers = translator.translate_many(["One.", "Two."])
        self.assertTrue(all(isinstance(answer, RateLimitedError) for answer in answers))
        self.assertEqual(len(translator.session.sent), 1)

    def test_repeated_and_cached_segments_are_not_sent(self):
        translator = translator_answering()
        translator.cache = FakeCache({"Header": "Tiêu đề"})
        answers = translator.translate_many(["Header", "Body", "Body", "Header"])
        self.assertEqual(answers, ["Tiêu đề", "vi:Body", "vi:Body", "Tiêu đề"])
        self.assertEqual(len(translator.session.sent), 1)
        self.assertEqual(translator.cache.get("Body"), "vi:Body")


class DeepSeekContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patcher = mock.patch(
            "pdf2zh.translator.DEEPSEEK_GLOSSARY_DIRECTORY", self.temp.name
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def glossary_answer(self, payload: dict) -> FakeResponse:
        system = payload["messages"][0]["content"]
        if "bilingual glossary" in system:
            return FakeResponse(chat(json.dumps({"liquidity risk": "rủi ro thanh khoản"})))
        return echo_answer(payload)

    def test_the_glossary_is_built_once_and_carried_by_every_request(self):
        translator = translator_answering(
            self.glossary_answer,
            context={"title": "Risk report", "sample": "Liquidity risk is the main concern."},
        )
        translator.translate_many(["Liquidity risk rises."])
        translator.translate_many(["Another paragraph."])
        systems = [payload["messages"][0]["content"] for payload in translator.session.sent]
        glossary_requests = [system for system in systems if "bilingual glossary" in system]
        self.assertEqual(len(glossary_requests), 1)
        translation_requests = [system for system in systems if "bilingual glossary" not in system]
        self.assertEqual(len(translation_requests), 2)
        for system in translation_requests:
            self.assertIn("Risk report", system)
        # Only the batch that spells the term carries the entry.
        self.assertIn("rủi ro thanh khoản", translation_requests[0])
        self.assertNotIn("rủi ro thanh khoản", translation_requests[1])

    def test_a_glossary_entry_only_travels_with_a_batch_that_uses_it(self):
        def glossary_then_echo(payload: dict) -> FakeResponse:
            if "bilingual glossary" in payload["messages"][0]["content"]:
                return FakeResponse(chat(json.dumps({"liquidity risk": "rủi ro thanh khoản"})))
            return echo_answer(payload)

        translator = translator_answering(
            glossary_then_echo, context={"sample": "Liquidity risk is the main concern."}
        )
        translator.translate_many(["Liquidity risk rises."])
        translator.translate_many(["Unrelated paragraph."])
        systems = [p["messages"][0]["content"] for p in translator.session.sent]
        translation_requests = [s for s in systems if "bilingual glossary" not in s]
        self.assertIn("rủi ro thanh khoản", translation_requests[0])
        self.assertNotIn("rủi ro thanh khoản", translation_requests[1])

    def test_a_glossary_word_is_not_matched_inside_a_longer_word(self):
        translator = translator_answering()
        translator._glossary = {"risk": "rủi ro"}
        prompt = translator._system_prompt(["Take the brisk walk."])
        self.assertNotIn("rủi ro", prompt)
        prompt = translator._system_prompt(["The risk rises."])
        self.assertIn("rủi ro", prompt)

    def test_a_long_outline_is_capped(self):
        outline = "A" * 400 + "\n" + "B" * 2000
        translator = translator_answering(context={"outline": outline})
        translator.translate_many(["Plain."])
        system = translator.session.sent[-1]["messages"][0]["content"]
        self.assertIn("A" * 400, system)
        # 800 characters survive: 400 A, one newline, 399 B.
        self.assertIn("B" * 399, system)
        self.assertNotIn("B" * 400, system)

    def test_a_second_run_reuses_the_saved_glossary(self):
        context = {"sample": "Liquidity risk is the main concern."}
        first = translator_answering(self.glossary_answer, context=context)
        first.translate_many(["Liquidity risk rises."])
        second = translator_answering(self.glossary_answer, context=context)
        second.translate_many(["Liquidity risk rises."])
        systems = [payload["messages"][0]["content"] for payload in second.session.sent]
        self.assertEqual(sum("bilingual glossary" in system for system in systems), 0)
        self.assertIn("rủi ro thanh khoản", systems[-1])

    def test_a_broken_glossary_step_does_not_stop_the_translation(self):
        def bad_glossary(payload: dict) -> FakeResponse:
            if "bilingual glossary" in payload["messages"][0]["content"]:
                return FakeResponse(chat("not json"))
            return echo_answer(payload)

        translator = translator_answering(bad_glossary, context={"sample": "Some text."})
        self.assertEqual(translator.translate_many(["Plain."]), ["vi:Plain."])

    def test_the_context_digest_is_what_the_cache_is_keyed_on(self):
        translator = translator_answering(context={"title": "Risk report"})
        translator.translate_many(["Plain."])
        self.assertTrue(translator.cache.params.get("context"))

    def test_the_digest_is_stored_through_the_real_cache_api(self):
        """add_params takes a key and a value; a dict raised TypeError only in
        production, because the fake cache accepted either shape."""
        cache = mock.create_autospec(TranslationCache, instance=True)
        cache.get.return_value = None
        translator = translator_answering(context={"title": "Risk report"})
        translator.cache = cache
        self.assertEqual(translator.translate_many(["Plain."]), ["vi:Plain."])
        cache.update_params.assert_called_once()
        cache.add_params.assert_not_called()


if __name__ == "__main__":
    unittest.main()
