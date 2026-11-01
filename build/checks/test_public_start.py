"""A reused dependency environment cannot conceal an incomplete public lock."""
import pytest

from pathlib import Path

from start import next_steps, verify_installed_inventory


LOCK = "# fictional lock\nfictional-library==1.0 --hash=sha256:" + "a" * 64 + "\n"


def test_exact_library_inventory_permits_only_installer_extra():
    verify_installed_inventory(LOCK, [["fictional_library", "1.0"], ["pip", "26.0"]])


@pytest.mark.parametrize("rows", [[], [["fictional-library", "2.0"]],
    [["fictional-library", "1.0"], ["undeclared-project", "1.0"]],
    [["fictional-library", "1.0"], ["fictional_library", "1.0"]]])
def test_stale_missing_mismatched_and_duplicate_distributions_refused(rows):
    with pytest.raises(ValueError, match="public_install_"):
        verify_installed_inventory(LOCK, rows)


def test_next_steps_keep_fictional_and_live_own_corpus_routes_separate():
    guide = next_steps(Path("C:/public-mosaic"), Path("C:/public-mosaic/.venv-public/Scripts/python.exe"))
    assert guide["fictional_demo"]["command"][-2:] == ["demo", "--serve"]
    own = guide["own_corpus"]
    assert own["scientific_acceptance"] is False
    commands = [item for command in own["before_any_model_request"] for item in command]
    assert "--live" not in commands
    assert "--model-manifest" in commands and "--weights-root" in commands
