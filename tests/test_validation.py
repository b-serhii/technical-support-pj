def validate_attendance(value):
    return 0 <= value <= 100

def test_validate_attendance():
    assert validate_attendance(85) is True
    assert validate_attendance(0) is True
    assert validate_attendance(100) is True
    assert validate_attendance(-1) is False
    assert validate_attendance(101) is False