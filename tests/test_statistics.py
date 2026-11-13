import pytest

from ml_lab.statistics import paired_comparison, summarize


def test_identical_pairs_show_no_effect():
    result = paired_comparison([0.8] * 10, [0.8] * 10)
    assert result["mean"] == 0 and result["ci95"] == [0, 0]
    assert result["exact_sign_flip_p_two_sided"] == 1


def test_all_positive_differences_have_expected_exact_tail_probability():
    result = paired_comparison([0.0] * 10, [0.01] * 10)
    assert result["exact_sign_flip_p_two_sided"] == 2 / 1024
    assert result["mean"] == pytest.approx(0.01)


def test_paired_analysis_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        paired_comparison([1, 2], [1, 2, 3])


def test_summary_uses_sample_standard_deviation():
    result = summarize([1.0, 2.0, 3.0])
    assert result["mean"] == 2 and result["std"] == 1
    assert result["ci95"][0] < 1 and result["ci95"][1] > 3
