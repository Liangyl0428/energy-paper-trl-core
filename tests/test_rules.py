from energy_paper_trl.rules import RuleTRLAnalyzer


def test_completed_lab_validation_reaches_trl4():
    out = RuleTRLAnalyzer().analyze(
        "Grid converter",
        "We developed a converter. Laboratory experiments validated the proposed voltage control and test results met the target."
    )
    assert out["level"] == 4
    assert out["evidence"]
    assert out["formal_certified_level"] is None


def test_future_validation_is_not_completed_evidence():
    out = RuleTRLAnalyzer().analyze(
        "Grid converter",
        "We propose a converter. Future laboratory experiments will validate the design; no testing has been completed."
    )
    assert out["level"] in (None, 2, 3)
    assert out["status"] != "text_supported_estimate" or out["level"] < 4


def test_chinese_follow_up_plan_is_not_completed_evidence():
    out = RuleTRLAnalyzer().analyze(
        "构网型储能变流器控制",
        "本文提出一种控制方案。在实验室完成了实验验证，测试结果满足要求。后续将开展现场示范。",
    )
    assert out["level"] == 4
    assert all("后续" not in item["quote"] for item in out["evidence"])
