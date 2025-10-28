from pathlib import Path

from app import settings


def test_write_env_preserves_existing(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=foo\nOPENAI_API_KEY=bar\nOTHER_SETTING=value\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "ENV_PATH", env_file)

    settings._write_env(None, None)

    contents = env_file.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=foo" in contents
    assert "OPENAI_API_KEY=bar" in contents
    assert "OTHER_SETTING=value" in contents


def test_write_env_updates_requested_keys(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=foo\nOPENAI_API_KEY=bar\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "ENV_PATH", env_file)

    settings._write_env("new-gem", None)
    settings._write_env(None, "new-openai")

    contents = env_file.read_text(encoding="utf-8").splitlines()
    assert "GEMINI_API_KEY=new-gem" in contents
    assert "OPENAI_API_KEY=new-openai" in contents
