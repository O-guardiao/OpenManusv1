from app.config import DaytonaSettings


def test_daytona_key_is_optional_until_daytona_is_used():
    settings = DaytonaSettings()

    assert settings.daytona_api_key == ""
    assert settings.VNC_password is None
