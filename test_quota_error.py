from study_lens_utils import format_model_error, is_quota_error


def test_quota_error_detection():
    error = RuntimeError("RESOURCE_EXHAUSTED: 429 quota exceeded")
    assert is_quota_error(error) is True


def test_quota_error_message():
    error = RuntimeError("RESOURCE_EXHAUSTED: 429 quota exceeded")
    message = format_model_error(error)
    assert "quota has been exhausted" in message.lower()
