"""Local regression and documentation gate; no remote model or package install.

Run with the project's prepared Python environment. Docker integration is an
explicit separate commissioning step, not a passing claim of this command.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def documentation_errors() -> list[str]:
    # The public distribution intentionally excludes internal documentation.
    return [] if (ROOT / 'README.md').is_file() else ['Missing public README.md']


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', help='Also build and smoke the wheel')
    args = parser.parse_args()
    parent = ROOT / 'workspace' / 'harness-checks'
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run-', dir=parent))
    started = time.perf_counter()
    report = {'started_at': datetime.now(timezone.utc).isoformat(),
              'assurance': 'local_regression', 'docker_integration': 'not_run',
              'real_model_evaluation': 'not_run', 'steps': [], 'output': str(output)}
    failure = 0
    errors = documentation_errors()
    sources = [*ROOT.joinpath('app').rglob('*.py'), ROOT / 'main.py', ROOT / 'session_main.py']
    for source in sources:
        try:
            ast.parse(source.read_text(encoding='utf-8-sig'), filename=str(source))
        except (SyntaxError, UnicodeError) as error:
            errors.append(f'{source.relative_to(ROOT)}: {error}')
    report['source_and_docs_errors'] = errors
    if errors:
        failure = 1
    commands = [('pytest', [sys.executable, '-m', 'pytest', 'tests',
                 '--ignore=tests/sandbox', '-q', '-p', 'no:cacheprovider',
                 '--basetemp', str(output / 'tmp'), '--junitxml', str(output / 'tests.xml')])]
    if args.build:
        commands.append(('build', [sys.executable, 'scripts/build_harness.py',
                                  '--output', str(output / 'package')]))
    for name, command in commands:
        print(f'{name}: running; log {output / (name + ".log")}', flush=True)
        with (output / (name + '.log')).open('w', encoding='utf-8') as log:
            try:
                result = subprocess.run(command, cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=600)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        report['steps'].append({'name': name, 'argv': command, 'exit_code': code})
        if code:
            failure = 1
            break
    xml = output / 'tests.xml'
    if xml.is_file():
        report['tests'] = [suite.attrib for suite in ET.parse(xml).getroot().iter('testsuite')]
    report.update(exit_code=failure, duration_seconds=round(time.perf_counter() - started, 3))
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return failure


if __name__ == '__main__':
    raise SystemExit(main())
