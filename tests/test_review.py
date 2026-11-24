import pytest

from src.evaluation.review import (
    ask,
    auto_accept_rest,
    draw_sample,
    frozen_sample,
    latest_decisions,
    make_decision,
    review_stats,
)

CANDIDATE = {
    "id": "F1-q0",
    "question": "Q ?",
    "answer": "A.",
    "relevant_chunk_ids": ["F1-000", "F2-000", "F3-000"],
    "flag_overlap": True,
}


def scripted(*answers: str):
    replies = iter(answers)
    return lambda prompt: next(replies)


def test_ask_accept_edit_narrow_and_reject():
    assert ask(CANDIDATE, scripted("a"))["status"] == "accepted"
    edited = ask(CANDIDATE, scripted("e", "Nouvelle question ?", "a"))
    assert (edited["status"], edited["question"]) == ("edited", "Nouvelle question ?")
    assert edited["answer"] == "A."
    narrowed = ask(CANDIDATE, scripted("k", "0, 2", "a"))
    assert narrowed["relevant_chunk_ids"] == ["F1-000", "F3-000"]
    rejected = ask(CANDIDATE, scripted("x", "r", "c"))
    assert (rejected["status"], rejected["reason"]) == ("rejected", "copied_wording")
    assert ask(CANDIDATE, scripted("s")) is None


def test_ask_edit_answer_accumulates_with_other_edits():
    record = ask(CANDIDATE, scripted("n", "Réponse corrigée.", "e", "Q2 ?", "a"))
    assert record["status"] == "edited"
    assert (record["question"], record["answer"]) == ("Q2 ?", "Réponse corrigée.")
    assert record["edits"] == ["answer", "question"]
    assert record["relevant_chunk_ids"] == CANDIDATE["relevant_chunk_ids"]
    assert ask(CANDIDATE, scripted("n", "discarded", "s")) is None


def test_edited_fields_are_counted():
    decisions = latest_decisions([make_decision(CANDIDATE, "edited", answer="x", edits=["answer"])])
    assert review_stats([CANDIDATE], decisions)["human_review"]["edited_fields"] == {"answer": 1}


def test_latest_decision_wins_and_stats_count_it():
    first = make_decision(CANDIDATE, "accepted")
    second = make_decision(CANDIDATE, "rejected", reason="duplicate")
    other = {"id": "F2-q0", "question": "?", "relevant_chunk_ids": ["F2-001"]}
    decisions = latest_decisions([first, second, make_decision(other, "accepted")])
    stats = review_stats([CANDIDATE, other, {"id": "F3-q0"}], decisions)
    human = stats["human_review"]
    assert stats["undecided"] == 1
    assert human["reviewed"] == 2
    assert human["by_status"] == {"rejected": 1, "accepted": 1}
    assert human["rejection_rate"] == 0.5
    assert human["reject_reasons"] == {"duplicate": 1}
    assert human["rejected_among_overlap_flagged"] == 1


def candidates(n: int) -> list[dict]:
    return [
        {"id": f"F{i}-q0", "question": "?", "relevant_chunk_ids": [f"F{i}-000"]} for i in range(n)
    ]


def test_draw_sample_is_reproducible_and_in_file_order():
    cands = candidates(50)
    ids = draw_sample(cands, 10, seed=42)
    assert len(set(ids)) == 10
    assert ids == draw_sample(cands, 10, seed=42)
    order = [c["id"] for c in cands]
    assert ids == sorted(ids, key=order.index)
    assert ids != draw_sample(cands, 10, seed=1)


def test_frozen_sample_refuses_a_different_sample(tmp_path):
    path = tmp_path / "sample.json"
    cands = candidates(20)
    first = frozen_sample(cands, 5, 42, path)
    assert frozen_sample(cands, 5, 42, path) == first
    with pytest.raises(SystemExit):
        frozen_sample(cands, 6, 42, path)


def test_rejection_rate_uses_the_sample_only_and_auto_accept_fills_the_rest():
    cands = candidates(4)
    sample = {"seed": 42, "ids": ["F0-q0", "F1-q0"]}
    decisions = latest_decisions(
        [
            make_decision(cands[0], "rejected", reason="duplicate"),
            make_decision(cands[2], "rejected", reason="other"),  # outside the sample
        ]
    )
    with pytest.raises(SystemExit):
        auto_accept_rest(cands, decisions, sample["ids"])
    decisions["F1-q0"] = make_decision(cands[1], "accepted")
    auto = auto_accept_rest(cands, decisions, sample["ids"])
    assert [(d["id"], d["status"]) for d in auto] == [("F3-q0", "auto_accepted")]
    decisions.update(latest_decisions(auto))
    stats = review_stats(cands, decisions, sample)
    assert stats["sample"]["complete"] is True
    assert stats["sample"]["rejection_rate"] == 0.5
    assert stats["human_reviewed_outside_sample"] == 1
    assert stats["by_status"] == {"rejected": 2, "accepted": 1, "auto_accepted": 1}
