"""Basic tests to ensure the test suite works."""


def test_basic_math():
    """Test basic math operations."""
    assert 1 + 1 == 2
    assert 2 * 3 == 6


def test_string_operations():
    """Test string operations."""
    assert "hello" + " world" == "hello world"
    assert "test".upper() == "TEST"


def test_list_operations():
    """Test list operations."""
    test_list = [1, 2, 3]
    assert len(test_list) == 3
    assert test_list[0] == 1
