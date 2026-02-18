"""Security validator for skills.

Two validation tiers:
- **Strict** (``validate_skill``): for skills downloaded from the Hub.
  Blocks subprocess, eval, exec, filesystem deletion, raw sockets, etc.
- **Local** (``validate_skill_local``): for locally-created and evolved skills.
  Only blocks truly catastrophic operations (privilege escalation, fork bombs,
  ctypes).  Allows subprocess, sockets, file ops, etc.

Pure stdlib — no external dependencies.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Dangerous pattern definitions — STRICT (Hub downloads)
# ---------------------------------------------------------------------------

# Patterns that indicate potentially dangerous code.  Each entry is
# (compiled_regex, human-readable reason).
_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Command execution
    (re.compile(r"\bos\.system\s*\("), "os.system() — arbitrary command execution"),
    (re.compile(r"\bos\.popen\s*\("), "os.popen() — arbitrary command execution"),
    (re.compile(r"\bos\.exec[lv]p?e?\s*\("), "os.exec*() — process replacement"),
    (re.compile(r"\bsubprocess\.\w+\s*\("), "subprocess — arbitrary command execution"),
    (re.compile(r"\bcommands\.\w+\s*\("), "commands module — command execution"),

    # Dynamic code execution
    (re.compile(r"\beval\s*\("), "eval() — arbitrary code execution"),
    (re.compile(r"\bexec\s*\("), "exec() — arbitrary code execution"),
    (re.compile(r"\bcompile\s*\(.+['\"]exec['\"]"), "compile() with exec mode"),
    (re.compile(r"\b__import__\s*\("), "__import__() — dynamic import"),

    # Filesystem destruction
    (re.compile(r"\bshutil\.rmtree\s*\("), "shutil.rmtree() — recursive deletion"),
    (re.compile(r"\bos\.remove\s*\("), "os.remove() — file deletion"),
    (re.compile(r"\bos\.unlink\s*\("), "os.unlink() — file deletion"),
    (re.compile(r"\bos\.rmdir\s*\("), "os.rmdir() — directory deletion"),
    (re.compile(r"\bos\.removedirs\s*\("), "os.removedirs() — recursive dir deletion"),

    # Sensitive file access
    (re.compile(r"""['"](/etc/passwd|/etc/shadow|~?/\.ssh|~?/\.env|~?/\.aws)"""),
     "access to sensitive system files"),
    (re.compile(r"""['"]\S*(credentials|secrets?|password|private.key)\S*['"]""", re.IGNORECASE),
     "access to credential/secret files"),

    # Network exfiltration (non-localhost)
    (re.compile(r"\bsocket\.socket\s*\("), "raw socket creation"),
    (re.compile(r"\bsmtplib\b"), "SMTP — email sending"),
    (re.compile(r"\bftplib\b"), "FTP — file transfer"),

    # Privilege escalation
    (re.compile(r"\bos\.setuid\s*\("), "os.setuid() — privilege escalation"),
    (re.compile(r"\bos\.setgid\s*\("), "os.setgid() — privilege escalation"),
    (re.compile(r"\bctypes\b"), "ctypes — low-level system access"),
]

# Patterns that are suspicious but not outright blocked — logged as warnings.
_WARNING_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bopen\s*\([^)]*['\"]w"), "writing to files"),
    (re.compile(r"\burllib\.request\.urlopen\s*\("), "outbound HTTP request"),
    (re.compile(r"\bhttp\.client\b"), "outbound HTTP via http.client"),
    (re.compile(r"\bpickle\.loads?\s*\("), "pickle deserialization (potential RCE)"),
]


# ---------------------------------------------------------------------------
# Dangerous pattern definitions — LOCAL (evolved / locally-created skills)
# ---------------------------------------------------------------------------

# Only block truly catastrophic operations.  Local skills are trusted.
_LOCAL_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Privilege escalation
    (re.compile(r"\bos\.setuid\s*\("), "os.setuid() — privilege escalation"),
    (re.compile(r"\bos\.setgid\s*\("), "os.setgid() — privilege escalation"),
    (re.compile(r"\bctypes\.cdll\b"), "ctypes.cdll — low-level library loading"),
    # Fork bomb patterns
    (re.compile(r"while\s+True\s*:.*os\.fork\s*\("), "fork bomb pattern"),
    # Recursive rm of root
    (re.compile(r"""shutil\.rmtree\s*\(\s*['"]/['"]"""), "shutil.rmtree('/') — root deletion"),
    (re.compile(r"""os\.system\s*\(\s*['"]rm\s+-rf\s+/['"]"""), "rm -rf / via os.system"),
]

_LOCAL_WARNING_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bctypes\b"), "ctypes — low-level system access"),
    (re.compile(r"\bpickle\.loads?\s*\("), "pickle deserialization (potential RCE)"),
    (re.compile(r"""['"](/etc/shadow|~?/\.ssh/id_)"""), "access to sensitive system files"),
]


class ValidationResult:
    """Result of skill security validation."""

    __slots__ = ("safe", "errors", "warnings")

    def __init__(self) -> None:
        self.safe: bool = True
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def __repr__(self) -> str:
        status = "SAFE" if self.safe else "BLOCKED"
        parts = [f"ValidationResult({status}"]
        if self.errors:
            parts.append(f", errors={self.errors}")
        if self.warnings:
            parts.append(f", warnings={self.warnings}")
        parts.append(")")
        return "".join(parts)


# Seed allowlist for capability skill dependencies.
_DEFAULT_ALLOWED_PACKAGES = frozenset({
    # Web / HTTP
    "requests", "httpx", "aiohttp", "urllib3", "websockets", "websocket-client",
    # Browser automation
    "playwright", "selenium", "pyppeteer",
    # HTML/XML parsing
    "beautifulsoup4", "lxml", "html5lib", "cssselect", "parsel",
    # Web frameworks (for skill-hosted APIs)
    "flask", "fastapi", "uvicorn", "starlette", "bottle", "tornado",
    # Email
    "imapclient",
    # Data / Science
    "pandas", "numpy", "scipy", "polars", "pyarrow",
    "openpyxl", "xlsxwriter", "csvkit",
    # Data visualization
    "matplotlib", "plotly", "seaborn", "altair",
    # PDF / Documents
    "pdfplumber", "pypdf", "reportlab", "python-docx", "python-pptx",
    # Media / Images
    "pillow", "opencv-python", "imageio",
    # AI / LLM
    "openai", "anthropic", "tiktoken", "transformers", "sentence-transformers",
    # Calendar
    "icalendar", "caldav",
    # System / DevOps
    "psutil", "docker", "paramiko", "fabric",
    # Database
    "sqlalchemy", "redis", "pymongo", "motor",
    # Utilities
    "python-dateutil", "pyyaml", "jinja2", "click", "rich", "tqdm",
    "pydantic", "attrs", "orjson", "msgpack",
    # Crypto / Auth
    "cryptography", "pyjwt", "oauthlib",
    # Testing
    "pytest", "httpx",
})


def validate_dependencies(
    dependencies: list[str],
    allowed: frozenset[str] | None = None,
) -> ValidationResult:
    """Check that all declared dependencies are on the allowlist.

    Returns ValidationResult with safe=False if any package is not allowed.
    """
    result = ValidationResult()
    if not dependencies:
        return result

    allowlist = allowed if allowed is not None else _DEFAULT_ALLOWED_PACKAGES

    for dep in dependencies:
        # Strip version specifiers: "requests>=2.0" → "requests"
        pkg = dep.split("==")[0].split(">=")[0].split("<=")[0].split("!=")[0].split("~=")[0].split(">")[0].split("<")[0].strip().lower()
        if not pkg:
            continue
        if pkg not in allowlist:
            result.errors.append(f"Package '{pkg}' not in allowed list")
            result.safe = False

    return result


def validate_skill_local(source_code: str) -> ValidationResult:
    """Validate locally-created or evolved skill source code.

    Much more permissive than ``validate_skill`` — only blocks privilege
    escalation, fork bombs, and root-deletion patterns.  Allows subprocess,
    sockets, file operations, eval/exec, etc.

    Returns a ValidationResult.
    """
    result = ValidationResult()

    if not source_code or not source_code.strip():
        result.errors.append("empty source code")
        result.safe = False
        return result

    for pattern, reason in _LOCAL_DANGEROUS_PATTERNS:
        matches = pattern.findall(source_code)
        if matches:
            result.errors.append(reason)
            result.safe = False

    for pattern, reason in _LOCAL_WARNING_PATTERNS:
        matches = pattern.findall(source_code)
        if matches:
            result.warnings.append(reason)

    return result


def validate_skill(source_code: str) -> ValidationResult:
    """Validate skill source code for security risks.

    Returns a ValidationResult with:
    - safe=True: code passed static analysis, OK to install
    - safe=False: code contains dangerous patterns, should be rejected

    Warnings are informational and do not block installation.
    """
    result = ValidationResult()

    if not source_code or not source_code.strip():
        result.errors.append("empty source code")
        result.safe = False
        return result

    # Strip comments and strings to reduce false positives.
    # We check against the raw source to catch patterns in strings too
    # (a malicious skill might hide code in string-based eval).

    for pattern, reason in _DANGEROUS_PATTERNS:
        matches = pattern.findall(source_code)
        if matches:
            result.errors.append(reason)
            result.safe = False

    for pattern, reason in _WARNING_PATTERNS:
        matches = pattern.findall(source_code)
        if matches:
            result.warnings.append(reason)

    return result
