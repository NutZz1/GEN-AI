"""Agent configuration: every knob lives here, read once from the environment (optionally via a
.env file - see .env.example). Nothing here is a secret; API keys are read by agent/llm.py directly
from the environment and never stored in this module or logged anywhere.
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass  # python-dotenv is optional; env vars set another way still work

PROVIDER = os.environ.get("AVATARBLEND_LLM_PROVIDER", "anthropic")  # "anthropic" | "ollama" | "none"

ANTHROPIC_MODEL = os.environ.get("AVATARBLEND_AGENT_MODEL", "claude-sonnet-5")
OLLAMA_MODEL = os.environ.get("AVATARBLEND_OLLAMA_MODEL", "llama3.1")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

MAX_TOKENS = int(os.environ.get("AVATARBLEND_AGENT_MAX_TOKENS", "1024"))
MAX_RETRIES = int(os.environ.get("AVATARBLEND_AGENT_MAX_RETRIES", "3"))
REQUEST_TIMEOUT_S = float(os.environ.get("AVATARBLEND_AGENT_TIMEOUT", "30"))

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "agent_runs.jsonl"
