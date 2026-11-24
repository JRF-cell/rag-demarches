from src.evaluation.review import ask, latest_decisions, make_decision, review_stats

CANDIDATE = {
    "id": "F1-q0",
    "question": "Q ?",
    "relevant_chunk_ids": ["F1-000", "F2-000", "F3-000"],
    "flag_overlap": True,
}


def scripted(*answers: str):
    replies = iter(answers)
    return lambda prompt: next(replies)


def test_ask_accept_edit_narrow_and_reject():
    assert ask(CANDIDATE, scripted("a"))["status"] == "accepted"
    edited = ask(CANDIDATE, scripted("e", "Nouvelle question ?"))
    assert (edited["status"], edited["question"]) == ("edited", "Nouvelle question ?")
    narrowed = ask(CANDIDATE, scripted("k", "0, 2"))
    assert narrowed["relevant_chunk_ids"] == ["F1-000", "F3-000"]
    rejected = ask(CANDIDATE, scripted("x", "r", "c"))
    assert (rejected["status"], rejected["reason"]) == ("rejected", "copied_wording")
    assert ask(CANDIDATE, scripted("s")) is None


def test_latest_decision_wins_and_stats_count_it():
    first = make_decision(CANDIDATE, "accepted")
    second = make_decision(CANDIDATE, "rejected", reason="duplicate")
    other = {"id": "F2-q0", "question": "?", "relevant_chunk_ids": ["F2-001"]}
    decisions = latest_decisions([first, second, make_decision(other, "accepted")])
    stats = review_stats([CANDIDATE, other, {"id": "F3-q0"}], decisions)
    assert stats["reviewed"] == 2
    assert stats["by_status"] == {"rejected": 1, "accepted": 1}
    assert stats["rejection_rate"] == 0.5
    assert stats["reject_reasons"] == {"duplicate": 1}
    assert stats["rejected_among_overlap_flagged"] == 1
