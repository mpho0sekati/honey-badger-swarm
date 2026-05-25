"""
🌑 Ultimate Dark Swarm Factory — Honey Badger (Defensive) Edition
Author: mpho sekati (Modified for Blue/Purple Team Architecture)
License: MIT

Architecture:
  - Each agent gets its own dedicated Groq API key (round-robin if fewer keys than agents)
  - Automatic rate-limit detection with exponential back-off and key rotation
  - Defensive Pipeline: Threat Modeling -> Secure Design -> Purple Team Auditing
  - Self-healing architecture with hardened Docker DevSecOps scaffolding
"""

from __future__ import annotations

import os
import re
import json
import time
import random
import py_compile
import tempfile
import zipfile
import traceback
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌑 Dark Swarm · Honey Badger Edition",
    page_icon="🦡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

.stApp { background-color: #080808; color: #d4d4d4; }

.swarm-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px; font-weight: 600;
    letter-spacing: -0.02em; color: #3b82f6; /* Blue team color */
    margin-bottom: 2px;
}
.swarm-sub {
    font-size: 13px; color: #555;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 16px;
}

/* Key manager */
.key-row {
    display: flex; align-items: center; gap: 8px;
    background: #111; border: 1px solid #222;
    border-radius: 6px; padding: 8px 12px;
    margin-bottom: 6px; font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
}
.key-row.active  { border-color: #3b82f6; }
.key-row.resting { border-color: #facc15; }
.key-row.error   { border-color: #f87171; }
.key-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink:0; }
.key-dot.active  { background: #3b82f6; }
.key-dot.resting { background: #facc15; }
.key-dot.error   { background: #f87171; }
.key-dot.idle    { background: #333; }

/* Agent cards */
.agent-card {
    background: #0e0e0e; border: 1px solid #1e1e1e;
    border-radius: 8px; padding: 10px 14px;
    margin-bottom: 6px; font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    transition: border-color 0.3s;
}
.agent-card.active { border-color: #3b82f6; background: #080d18; }
.agent-card.done   { border-color: #8b5cf6; background: #120a1f; } /* Purple team done */
.agent-card.error  { border-color: #f87171; background: #180808; }

/* Log */
.log-wrap {
    background: #060606; border: 1px solid #1a1a1a;
    border-radius: 8px; padding: 10px;
    max-height: 340px; overflow-y: auto;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
}
.log-line { padding: 2px 0; border-bottom: 1px solid #111; color: #555; }
.log-line.ok   { color: #3b82f6; }
.log-line.warn { color: #facc15; }
.log-line.err  { color: #f87171; }
.log-line.info { color: #60a5fa; }

/* Instruction box */
.stTextArea textarea {
    background: #0e0e0e !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #d4d4d4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.08) !important;
}
.stTextInput input {
    background: #0e0e0e !important;
    border: 1px solid #2a2a2a !important;
    color: #d4d4d4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    border-radius: 6px !important;
}
.stTextInput input:focus { border-color: #3b82f6 !important; }

/* Buttons */
.stButton > button {
    background: #080d18 !important;
    border: 1px solid #3b82f6 !important;
    color: #3b82f6 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
}
.stButton > button:hover {
    background: #3b82f6 !important; color: #fff !important;
}
.stDownloadButton > button {
    background: #080d18 !important;
    border: 1px solid #8b5cf6 !important;
    color: #8b5cf6 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    width: 100% !important;
}

.stProgress > div > div { background: #3b82f6 !important; }
h1,h2,h3 { color: #d4d4d4; }
.stSelectbox label, .stSlider label, .stRadio label { color: #666 !important; font-size:12px !important; }
hr { border-color: #1a1a1a !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GROQ MODELS AVAILABLE
# ─────────────────────────────────────────────────────────────────────────────
GROQ_MODELS = [
    "groq/llama-3.3-70b-versatile",
    "groq/llama-3.1-70b-versatile",
    "groq/llama3-70b-8192",
    "groq/llama3-8b-8192",
    "groq/mixtral-8x7b-32768",
    "groq/gemma2-9b-it",
]

# ─────────────────────────────────────────────────────────────────────────────
# GROQ KEY POOL
# ─────────────────────────────────────────────────────────────────────────────
class GroqKeyPool:
    RATE_LIMIT_COOLDOWN = 62
    MAX_ERRORS = 3

    def __init__(self, keys: list[str], model: str):
        self.model = model
        self.keys: list[dict] = []
        for k in keys:
            k = k.strip()
            if k:
                self.keys.append({
                    "key": k,
                    "masked": f"gsk_...{k[-6:]}",
                    "status": "idle",
                    "errors": 0,
                    "cooldown_until": 0.0,
                    "used_count": 0,
                })
        self._current_idx = 0

    def available_keys(self) -> list[dict]:
        now = time.time()
        return [k for k in self.keys if k["status"] != "error" and now >= k["cooldown_until"]]

    def get_llm(self, preferred_idx: Optional[int] = None) -> tuple[LLM, int]:
        available = self.available_keys()
        if not available:
            soonest = min(self.keys, key=lambda k: k["cooldown_until"])
            wait = max(0, soonest["cooldown_until"] - time.time()) + 1
            time.sleep(wait)
            available = self.available_keys()
            if not available:
                raise RuntimeError("All Groq API keys exhausted or in error state.")

        if preferred_idx is not None:
            pool_idx = preferred_idx % len(available)
        else:
            pool_idx = self._current_idx % len(available)
            self._current_idx = (self._current_idx + 1) % len(available)

        key_info = available[pool_idx]
        key_info["status"] = "active"
        key_info["used_count"] += 1
        os.environ["GROQ_API_KEY"] = key_info["key"]
        llm = LLM(model=self.model, api_key=key_info["key"], temperature=0.7, max_tokens=4096)
        return llm, self.keys.index(key_info)

    def report_success(self, key_idx: int):
        if 0 <= key_idx < len(self.keys):
            self.keys[key_idx]["status"] = "idle"
            self.keys[key_idx]["errors"] = 0

    def report_rate_limit(self, key_idx: int):
        if 0 <= key_idx < len(self.keys):
            self.keys[key_idx]["status"] = "resting"
            self.keys[key_idx]["cooldown_until"] = time.time() + self.RATE_LIMIT_COOLDOWN

    def report_error(self, key_idx: int):
        if 0 <= key_idx < len(self.keys):
            self.keys[key_idx]["errors"] += 1
            if self.keys[key_idx]["errors"] >= self.MAX_ERRORS:
                self.keys[key_idx]["status"] = "error"
            else:
                self.keys[key_idx]["status"] = "idle"

    def status_html(self) -> str:
        html = ""
        for k in self.keys:
            s = k["status"]
            html += (
                f"<div class='key-row {s}'>"
                f"<div class='key-dot {s}'></div>"
                f"<span style='flex:1; color:#888;'>{k['masked']}</span>"
                f"<span style='color:#444; font-size:10px;'>×{k['used_count']}</span>"
                f"<span style='color:#555; font-size:10px; margin-left:8px;'>{s}</span>"
                f"</div>"
            )
        return html or "<div style='color:#555; font-size:12px;'>No keys configured</div>"

# ─────────────────────────────────────────────────────────────────────────────
# AGENT ROLE DEFINITIONS (DEFENSIVE SWARM EDITION)
# ─────────────────────────────────────────────────────────────────────────────
AGENT_ROLES = [
    {
        "role": "Swarm Commander",
        "goal": "Orchestrate the architecture and ensure all security constraints are met.",
        "backstory": (
            "A battle-hardened Principal Security Architect. You break instructions into "
            "clear deliverables, mandate zero-trust principles, and ensure the swarm builds "
            "exactly what was asked while refusing to compromise on security."
        ),
        "icon": "🎯",
    },
    {
        "role": "Threat Modeler",
        "goal": "Analyze requirements and predict attack vectors before code is written.",
        "backstory": (
            "You are an expert in STRIDE and DREAD frameworks. You identify spoofing, "
            "tampering, and DoS risks in proposed architectures and mandate specific "
            "mitigations for the engineering agents to follow."
        ),
        "icon": "🕵️",
    },
    {
        "role": "AppSec Engineer",
        "goal": "Implement backend logic with strict OWASP Top 10 protections.",
        "backstory": (
            "Senior Python engineer specializing in secure coding. You write clean, typed "
            "code with strict input validation, parameterized queries, and proper cryptography. "
            "You never use hardcoded secrets or MD5."
        ),
        "icon": "🔐",
    },
    {
        "role": "API Designer",
        "goal": "Build well-structured, authenticated RESTful endpoints.",
        "backstory": (
            "FastAPI and Pydantic v2 expert. You design APIs with strict schema validation, "
            "rate limiting, and proper authorization checks (OAuth2/JWT) on every endpoint."
        ),
        "icon": "🔌",
    },
    {
        "role": "DevSecOps Guardian",
        "goal": "Generate hardened, least-privilege infrastructure files.",
        "backstory": (
            "You never ship without a hardened Dockerfile (dropping root, read-only FS "
            "where possible), secure .env management, and network-isolated docker-compose stacks."
        ),
        "icon": "🛡️",
    },
    {
        "role": "Purple Team Auditor",
        "goal": "Simulate an attacker's mindset to find flaws, then write the remediation.",
        "backstory": (
            "You review all generated code by thinking like a hacker. If you find a logical flaw, "
            "IDOR, or bypass, you explicitly point it out and provide the patched code. "
            "The mission isn't complete until you sign off."
        ),
        "icon": "🦡",
    },
    {
        "role": "Database Architect",
        "goal": "Design normalized schemas and resilient data layers.",
        "backstory": (
            "PostgreSQL expert. You create efficient queries, ensure data at rest is handled "
            "logically, and build resilient connection pools that gracefully handle failures."
        ),
        "icon": "🗄️",
    }
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def validate_python_syntax(code: str) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        py_compile.compile(tmp, doraise=True)
        return True, ""
    except py_compile.PyCompileError as e:
        return False, str(e)
    finally:
        try: os.unlink(tmp)
        except OSError: pass

def write_file(path: str, content: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def make_zip(project_dir: str, zip_path: str) -> None:
    base = Path(project_dir)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in base.rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(base))

def safe_str(result) -> str:
    if result is None: return ""
    if hasattr(result, "raw"): return str(result.raw)
    if hasattr(result, "result"): return str(result.result)
    return str(result)

def extract_files_from_result(text: str) -> dict[str, str]:
    files: dict[str, str] = {}
    segments = re.split(r"
http://googleusercontent.com/immersive_entry_chip/0
