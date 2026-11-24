import io
import json
import urllib.error

import pandas as pd
import pytest

from src.evaluation.gemini import GeminiClient, QuotaExhausted, response_text
from src.evaluation.generate_questions import (
    build_candidates,
    parse_questions,
    pick_chunks,
    relevant_index,
    word_overlap,
)

CONFIG = {
    "model": "m",
    "seed": 0,
    "temperature": 0.0,
    "questions_per_fiche": 2,
    "overlap_flag": 0.7,
    "prompt_version": "test",
}


def fake_response(items: list[dict]) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(items)}]}}]}


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(b""))


def make_client(tmp_path, post) -> GeminiClient:
    return GeminiClient("m", "key", tmp_path, 60.0, max_retries=2, post=post, sleep=lambda s: None)


def test_parse_questions_keeps_valid_items_and_counts_the_rest():
    text = json.dumps(
        [
            {"chunk_id": "F1-000", "question": " Q1 ? ", "answer": "a"},
            {"chunk_id": "F9-000", "question": "Q2 ?"},
            {"chunk_id": "F1-001", "question": ""},
        ]
    )
    kept, dropped = parse_questions(text, {"F1-000", "F1-001"})
    assert kept == [{"chunk_id": "F1-000", "question": "Q1 ?", "answer": "a"}]
    assert dropped == {"unknown_chunk_id": 1, "malformed_item": 1}
    assert parse_questions("not json", set())[1] == {"invalid_json": 1}


def test_word_overlap_hand_computed():
    # content words (>= 4 letters): "combien", "coute", "passeport" -> 2 of 3 in the body
    assert word_overlap("Combien coute un passeport ?", "Le passeport coute 86 euros.") == 2 / 3
    assert word_overlap("Et ?", "texte") == 0.0


def test_relevant_index_groups_identical_bodies():
    chunks = pd.DataFrame(
        {"chunk_id": ["A-0", "A-1", "B-0"], "body": ["Même texte", "autre", "même  TEXTE"]}
    )
    index = relevant_index(chunks)
    assert index["A-0"] == index["B-0"] == ["A-0", "B-0"]
    assert index["A-1"] == ["A-1"]


def test_pick_chunks_skips_short_and_duplicate_bodies():
    chunks = pd.DataFrame(
        {"chunk_id": ["c0", "c1", "c2", "c3"], "body": ["x " * 50, "y", "x " * 50, "z " * 50]}
    )
    chunks["n_words"] = chunks["body"].str.split().str.len()
    picked = pick_chunks(chunks, k=5, min_words=40, seed=0)
    assert list(picked["chunk_id"]) == ["c0", "c3"]


def test_client_caches_responses(tmp_path):
    calls = []

    def post(url, body, key):
        calls.append(body)
        return fake_response([])

    client = make_client(tmp_path, post)
    first = client.generate("prompt", {"t": 0})
    second = client.generate("prompt", {"t": 0})
    assert len(calls) == 1
    assert first == second
    assert response_text(first["response"]) == "[]"


def test_client_retries_on_429_then_gives_up(tmp_path):
    outcomes = [http_error(429), fake_response([])]

    def post(url, body, key):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert make_client(tmp_path, post).generate("p", {})["response"] == fake_response([])

    def always_429(url, body, key):
        raise http_error(429)

    with pytest.raises(QuotaExhausted):
        make_client(tmp_path, always_429).generate("other", {})
    assert len(list(tmp_path.glob("*.json"))) == 1  # the failed call is not cached


def test_build_candidates_from_cache_only(tmp_path):
    chunks = pd.DataFrame(
        {
            "chunk_id": ["F1-000", "F2-000"],
            "fiche_id": ["F1", "F2"],
            "body": ["Le passeport coute 86 euros.", "le passeport coute 86 euros."],
            "theme": ["Papiers", "Papiers"],
        }
    )
    items = [{"chunk_id": "F1-000", "question": "Quel est le prix ?", "answer": "86 euros"}]
    client = make_client(tmp_path, lambda u, b, k: fake_response(items))
    plans = [
        {"fiche_id": "F1", "chunk_ids": ["F1-000"], "prompt": "p1"},
        {"fiche_id": "F2", "chunk_ids": ["F2-000"], "prompt": "p2"},
    ]
    gen = {"temperature": 0.0, "seed": 0, "responseMimeType": "application/json"}
    client.generate("p1", gen)
    candidates, stats = build_candidates(plans, client, chunks, CONFIG)
    assert client.calls_made == 1
    assert stats["sheets_without_response"] == 1
    assert candidates[0]["relevant_chunk_ids"] == ["F1-000", "F2-000"]
    assert candidates[0]["source_chunk_id"] == "F1-000"
