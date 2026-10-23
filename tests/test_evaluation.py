"""
Pytest integration for the automated evaluation suite.
"""

from evaluation.evaluate import run_evaluation


def test_automated_evaluation_benchmark():
    """Run full benchmark evaluation and assert 100% pass rate."""
    summary = run_evaluation()
    assert summary["failed_test_cases"] == 0, (
        f"Evaluation benchmark had {summary['failed_test_cases']} failures out of {summary['total_test_cases']} cases."
    )
    assert summary["overall_pass_rate"] == 100.0
    assert summary["negative_refusal_accuracy"] == 100.0
    assert summary["anti_injection_resilience"] == 100.0
