from setuptools import setup, find_packages


def get_version():
    """Read version from version.py file"""
    version_file = "agentx/version.py"
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VERSION"):
                return line.split("=")[1].strip().strip('"').strip("'")
    raise RuntimeError("Unable to find version string.")


def get_long_description():
    """Read README.md file"""
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()


setup(
    name="agentx-python",
    version=get_version(),
    packages=find_packages(),
    # PEP 561: ships type hints for consumers' type checkers (mypy/pyright). Without this marker,
    # every symbol imported from `agentx` is treated as untyped `Any`, silently disabling type
    # checking for anything that touches the SDK.
    package_data={"agentx": ["py.typed"]},
    zip_safe=False,
    entry_points={
        # Thin launcher for the self-hostable governance engine (AgentX-ai/AgentX-trace-eval),
        # not this SDK itself, see agentx/cli.py's module docstring for why it lazily downloads
        # rather than bundling that engine in this package.
        "console_scripts": [
            "agentx-trace-eval=agentx.cli:main",
            # Moveworks Data API -> AgentX traces sync (see agentx/integrations/moveworks.py) -
            # a pull importer, since Moveworks agents run in their cloud with no in-process hook.
            "agentx-moveworks=agentx.integrations.moveworks:cli_main",
        ],
    },
    install_requires=[
        "urllib3>=1.26.11",
        "certifi",
        "requests",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "langchain": ["langchain-core>=0.1.0"],
        "crewai": ["crewai>=0.80.0"],
        "openai-agents": ["openai-agents>=0.0.3"],
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.25.0"],
        "google-adk": ["google-adk>=1.0.0"],
        "google-genai": ["google-genai>=1.0.0"],
        "litellm": ["litellm>=1.0.0"],
        "llamaindex": ["llama-index-core>=0.10.0"],
        "autogen": ["autogen-agentchat>=0.4.0", "autogen-core>=0.4.0"],
        "all": [
            "langchain-core>=0.1.0",
            "crewai>=0.80.0",
            "openai-agents>=0.0.3",
            "openai>=1.0.0",
            "anthropic>=0.25.0",
            "google-adk>=1.0.0",
            "google-genai>=1.0.0",
            "litellm>=1.0.0",
            "llama-index-core>=0.10.0",
            "autogen-agentchat>=0.4.0",
            "autogen-core>=0.4.0",
        ],
    },
    author="Robin Wang and AgentX Team",
    author_email="contact@agentx.so",
    description="Official Python SDK for AgentX (https://www.agentx.so/)",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/AgentX-ai/AgentX-python",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
