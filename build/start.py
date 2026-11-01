"""Install the hash-locked Windows CPU demo, then build and serve it.

This command installs Python libraries only. It never downloads model weights
or calls a model provider. Use --wheelhouse for a fully offline installation.
"""
from __future__ import annotations

import argparse
import configparser
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import venv

from mosaic.workspace import Workspace
from mosaic.public_export import verify_export


def next_steps(root: Path, python: Path) -> dict[str, object]:
    """Return commands for the explicitly separate fictional and own-corpus paths.

    This deliberately prints commands rather than silently advancing from a
    successful fictional fixture into an operator's PDFs or model runtime.
    """
    executable = str(python)
    bootstrap = str(root / "bootstrap.py")
    return {
        "version": "mosaic-public-next-steps/v1",
        "fictional_demo": {
            "command": [executable, "-s", bootstrap, "demo", "--serve"],
            "meaning": "Runs only the bundled fictional fixture; no model weights or provider requests.",
        },
        "own_corpus": {
            "before_any_model_request": [
                [executable, "-s", bootstrap, "plan", "--config", "config/my-corpus.toml"],
                [executable, "-s", bootstrap, "preflight", "--config", "config/my-corpus.toml"],
                [executable, "-s", bootstrap, "corpus", "--config", "config/my-corpus.toml",
                 "--model-manifest", "config/my-model-manifest.json", "--weights-root", "models/my-model",
                 "--driver-profile", "config/my-driver.json"],
            ],
            "requirements": [
                "Copy and edit the public corpus, model declaration, and native driver examples.",
                "List only PDFs that the operator is permitted to process.",
                "Stage a complete local open-weights model and verify its manifest before adding --live.",
                "Review CORPUS_WORKFLOW.md before using --live; no Linux/GPU model route is established by this launcher.",
            ],
            "scientific_acceptance": False,
        },
    }


def verify_installed_inventory(lock_text: str, installed: list[list[str]]) -> None:
    """Reject missing, different or undeclared libraries in a reused environment."""
    normalize = lambda name: re.sub(r"[-_.]+", "-", name).casefold()
    expected = {}
    for line in lock_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+) --hash=sha256:[a-f0-9]{64}", line)
        if match is None or normalize(match[1]) in expected:
            raise ValueError("public_install_lock_invalid")
        expected[normalize(match[1])] = match[2]
    if not expected:
        raise ValueError("public_install_lock_invalid")
    observed = {}
    for row in installed:
        if (not isinstance(row, list) or len(row) != 2 or not all(isinstance(item, str) for item in row)
                or normalize(row[0]) in observed):
            raise ValueError("public_install_inventory_invalid")
        observed[normalize(row[0])] = row[1]
    # pip is the interpreter's installer, not an application dependency. All
    # other distributions must have their exact version in the reviewed lock.
    observed.pop("pip", None)
    if observed != expected:
        raise ValueError("public_install_environment_not_exact")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--no-serve", action="store_true")
    parser.add_argument("--print-next-steps", action="store_true",
                        help="Install and verify the public environment, then print explicit fictional and own-corpus commands without running either route.")
    parser.add_argument("--literature-demo", action="store_true",
                        help="Run the separate two-paper fictional literature-review fixture after installation.")
    parser.add_argument("--port", type=int, default=8798)
    args = parser.parse_args(argv)
    if sys.version_info[:2] != (3, 12) or platform.system() != "Windows" or platform.machine().lower() not in ("amd64", "x86_64"):
        parser.error("This dependency lock supports Windows x86_64 CPython 3.12 only; Linux GPU runtime is a separate unvalidated profile.")
    root = Path(__file__).resolve().parent
    # Installation is supported from the explicit export, not a private
    # checkout that happens to contain this launcher.
    verify_export(root, reject_unlisted=True)
    workspace = Workspace(repo_root=root)
    raw_environment = root / ".venv-public"
    environment = workspace.check(raw_environment)
    if raw_environment != environment or raw_environment.is_symlink() or getattr(raw_environment, "is_junction", lambda: False)():
        parser.error("linked environment directory refused")
    if not (environment / "Scripts/python.exe").is_file():
        venv.EnvBuilder(with_pip=True, clear=False, system_site_packages=False).create(environment)
    environment_config = configparser.ConfigParser()
    environment_config.read_string("[venv]\n" + (environment / "pyvenv.cfg").read_text(encoding="utf-8"))
    if environment_config.getboolean("venv", "include-system-site-packages", fallback=True):
        parser.error("the public environment must exclude system site packages")
    python = environment / "Scripts/python.exe"
    if workspace.check(python) != python.absolute() or python.is_symlink():
        parser.error("linked environment interpreter refused")
    clean_env = {k: v for k, v in os.environ.items()
                 if k not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} and not k.startswith("PIP_")}
    clean_env.update({"PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    install = [str(python), "-I", "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
               "--no-cache-dir", "--require-hashes", "-r", str(root / "requirements-public-windows.lock")]
    if args.wheelhouse:
        wheel_root = args.wheelhouse.absolute()
        if str(wheel_root).startswith(("\\\\", "//")) or wheel_root != wheel_root.resolve() or not wheel_root.is_dir():
            parser.error("wheelhouse must be an existing local directory without symlink/junction redirection")
        install += ["--no-index", "--find-links", str(args.wheelhouse.resolve())]
    else:
        install += ["--index-url", "https://pypi.org/simple"]
    print("Installing the pinned CPU demonstration dependencies. No model weights or scientific provider requests.", flush=True)
    subprocess.run(install, cwd=root, env=clean_env, check=True)
    inventory = subprocess.run(
        [str(python), "-I", "-c", "import json; from importlib.metadata import distributions; print(json.dumps([[d.metadata['Name'], d.version] for d in distributions()]))"],
        cwd=root, env=clean_env, capture_output=True, text=True, check=True)
    verify_installed_inventory((root / "requirements-public-windows.lock").read_text(encoding="utf-8"), json.loads(inventory.stdout))
    subprocess.run([str(python), "-I", "-m", "pip", "check"], cwd=root, env=clean_env, check=True)
    if args.print_next_steps and args.literature_demo:
        parser.error("--print-next-steps and --literature-demo cannot be combined")
    if args.print_next_steps:
        print(json.dumps(next_steps(root, python), sort_keys=True), flush=True)
        return 0
    command = [str(python), "-s", str(root / "bootstrap.py"),
               "literature-demo" if args.literature_demo else "demo", "--port", str(args.port)]
    if not args.no_serve:
        command += ["--serve"]
    return subprocess.call(command, cwd=root, env=clean_env)


if __name__ == "__main__":
    raise SystemExit(main())
