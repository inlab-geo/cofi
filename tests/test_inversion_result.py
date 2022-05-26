import pytest
from cofi import SamplingResult


def test_to_arviz():
    res = SamplingResult({"success": True, "sampler": None})
    with pytest.raises(NotImplementedError):
        res.to_arviz()
