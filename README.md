# OpenManusv1

Command-line agent fork of [OpenManus](https://github.com/FoundationAgents/OpenManus).
Maintained at [O-guardiao/OpenManusv1](https://github.com/O-guardiao/OpenManusv1).
The upstream MIT license and attribution are preserved in LICENSE.

## Setup

Requires Python 3.12 or newer. Install the dependencies into a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item config/config.example.toml config/config.toml
```

Configure your provider, model and credentials locally. Do not commit that file.
OPENMANUS_CONFIG can select an existing configuration file explicitly.

## Run from the terminal

```powershell
.\.venv\Scripts\python.exe main.py --help
.\.venv\Scripts\python.exe main.py --prompt "Describe your task" --session new --stream
.\.venv\Scripts\python.exe session_main.py --help
```

Tools require explicit per-run grants such as --allow-tool str_replace_editor.
Resuming a conversation does not restore previous grants. --session persists
conversation content in the selected local SQLite database; protect that file.
Unresolved effects prevent automatic resumption. Pausing and cancellation are
cooperative and do not prove that an external effect did not occur.

## Validate and build

```powershell
.\.venv\Scripts\python.exe scripts/verify_harness.py --build
```

The local suite includes deterministic HTTP/SSE fixtures, real CLI subprocesses,
session recovery and file-tool checks. Docker integration is excluded from this
command. The wheel is exercised outside the source tree using installed
dependencies; this is not a clean-install or real-model performance benchmark.
Optional native/WebAssembly verifiers use the Go module under cockpit/.

Source, tests, required runtime resources and sanitized configuration examples
are published. Internal architecture, strategy, transcripts, reports, credentials,
IDE settings and generated artifacts stay outside the public tree. Review the
staged files before every publication; .gitignore cannot remove past commits.
