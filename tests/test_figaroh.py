
from typer.testing import CliRunner
from figaroh.cli import app


def test_main():
    runner = CliRunner()
    result = runner.invoke(app, ["Joe", "-a", 10])

    assert "Hello, I'm Joe. I'm 10 years old." in result.output
    assert result.exit_code == 0
