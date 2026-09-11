from pathlib import Path
import re

from setuptools import find_namespace_packages, setup


ROOT = Path(__file__).resolve().parent

with (ROOT / "README.md").open("r", encoding="utf-8") as fh:
    long_description = fh.read()

# Keep the checked-in environment as the dependency authority. Build and test
# tooling is not a runtime dependency; no dependencies are downloaded by setup.
runtime_requirements = []
for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
    requirement = line.strip()
    if not requirement or requirement.startswith("#"):
        continue
    name = re.split(r"[<>=!~\[;\s]", requirement, maxsplit=1)[0].lower().replace("_", "-")
    if name not in {"pytest", "pytest-asyncio", "setuptools"}:
        runtime_requirements.append(requirement)

setup(
    name="openmanus",
    version="0.1.0",
    author="mannaandpoem and OpenManus Team",
    author_email="mannaandpoem@gmail.com",
    description="A versatile agent that can solve various tasks using multiple tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FoundationAgents/OpenManus",
    packages=find_namespace_packages(include=["app", "app.*"]),
    py_modules=["main", "session_main"],
    package_data={
        "app.oak": [f"bundles/openmanus-v1/{name}.json"
                    for name in ("schema", "functions", "graph", "kernel.lock")],
        "app.cockpit": ["static/cockpit.css", "static/verifier.js"],
    },
    install_requires=runtime_requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    entry_points={
        "console_scripts": [
            "openmanus=main:cli",
            "openmanus-session=session_main:cli",
        ],
    },
)
