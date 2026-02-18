"""Tests for ring1.skill_validator — security validation for skills."""

from ring1.skill_validator import (
    validate_skill,
    validate_skill_local,
    validate_dependencies,
    ValidationResult,
    _DEFAULT_ALLOWED_PACKAGES,
)


class TestValidateSkill:
    """validate_skill() should detect dangerous patterns."""

    def test_safe_http_server_skill(self):
        code = '''
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())

HTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
'''
        result = validate_skill(code)
        assert result.safe is True
        assert result.errors == []

    def test_rejects_os_system(self):
        result = validate_skill('import os\nos.system("rm -rf /")')
        assert result.safe is False
        assert any("os.system" in e for e in result.errors)

    def test_rejects_subprocess(self):
        result = validate_skill('import subprocess\nsubprocess.run(["ls"])')
        assert result.safe is False
        assert any("subprocess" in e for e in result.errors)

    def test_rejects_eval(self):
        result = validate_skill('x = eval(input())')
        assert result.safe is False
        assert any("eval" in e for e in result.errors)

    def test_rejects_exec(self):
        result = validate_skill('exec("print(1)")')
        assert result.safe is False
        assert any("exec" in e for e in result.errors)

    def test_rejects_shutil_rmtree(self):
        result = validate_skill('import shutil\nshutil.rmtree("/tmp/data")')
        assert result.safe is False
        assert any("shutil.rmtree" in e for e in result.errors)

    def test_rejects_dunder_import(self):
        result = validate_skill('m = __import__("os")')
        assert result.safe is False
        assert any("__import__" in e for e in result.errors)

    def test_rejects_sensitive_file_access(self):
        result = validate_skill('open("/etc/passwd")')
        assert result.safe is False
        assert any("sensitive" in e for e in result.errors)

    def test_rejects_raw_socket(self):
        result = validate_skill('import socket\ns = socket.socket()')
        assert result.safe is False
        assert any("socket" in e for e in result.errors)

    def test_rejects_smtp(self):
        result = validate_skill('import smtplib\nsmtplib.SMTP("mail.com")')
        assert result.safe is False
        assert any("SMTP" in e for e in result.errors)

    def test_rejects_ctypes(self):
        result = validate_skill('import ctypes\nctypes.cdll.LoadLibrary("libc.so")')
        assert result.safe is False
        assert any("ctypes" in e for e in result.errors)

    def test_rejects_os_remove(self):
        result = validate_skill('import os\nos.remove("/tmp/file")')
        assert result.safe is False
        assert any("os.remove" in e for e in result.errors)

    def test_warns_on_file_write(self):
        result = validate_skill('f = open("output.txt", "w")')
        assert result.safe is True  # warnings don't block
        assert any("writing" in w for w in result.warnings)

    def test_warns_on_urllib(self):
        result = validate_skill('import urllib.request\nurllib.request.urlopen("http://example.com")')
        assert result.safe is True
        assert any("HTTP" in w for w in result.warnings)

    def test_warns_on_pickle(self):
        result = validate_skill('import pickle\npickle.loads(data)')
        assert result.safe is True
        assert any("pickle" in w for w in result.warnings)

    def test_empty_source_rejected(self):
        result = validate_skill("")
        assert result.safe is False
        assert any("empty" in e for e in result.errors)

    def test_whitespace_only_rejected(self):
        result = validate_skill("   \n\n  ")
        assert result.safe is False

    def test_multiple_violations(self):
        code = 'os.system("bad")\neval(x)\nexec(y)'
        result = validate_skill(code)
        assert result.safe is False
        assert len(result.errors) >= 3

    def test_validation_result_repr(self):
        result = ValidationResult()
        assert "SAFE" in repr(result)
        result.safe = False
        result.errors = ["test"]
        assert "BLOCKED" in repr(result)


class TestValidateDependencies:
    """validate_dependencies() should check packages against allowlist."""

    def test_allowed_packages_pass(self):
        result = validate_dependencies(["requests", "pandas"])
        assert result.safe is True
        assert result.errors == []

    def test_disallowed_package_rejected(self):
        result = validate_dependencies(["requests", "malicious_pkg"])
        assert result.safe is False
        assert any("malicious_pkg" in e for e in result.errors)

    def test_version_specifier_stripped(self):
        result = validate_dependencies(["requests>=2.0", "pandas==1.5.3"])
        assert result.safe is True

    def test_tilde_version_stripped(self):
        result = validate_dependencies(["requests~=2.28"])
        assert result.safe is True

    def test_empty_list_is_safe(self):
        result = validate_dependencies([])
        assert result.safe is True
        assert result.errors == []

    def test_custom_allowlist(self):
        custom = frozenset({"my_pkg"})
        result = validate_dependencies(["my_pkg"], allowed=custom)
        assert result.safe is True
        # Default packages should be rejected with custom list.
        result2 = validate_dependencies(["requests"], allowed=custom)
        assert result2.safe is False

    def test_case_insensitive(self):
        result = validate_dependencies(["Requests", "PANDAS"])
        assert result.safe is True

    def test_default_allowlist_has_expected_packages(self):
        assert "requests" in _DEFAULT_ALLOWED_PACKAGES
        assert "playwright" in _DEFAULT_ALLOWED_PACKAGES
        assert "pandas" in _DEFAULT_ALLOWED_PACKAGES
        assert "pillow" in _DEFAULT_ALLOWED_PACKAGES
        # Expanded allowlist packages.
        assert "selenium" in _DEFAULT_ALLOWED_PACKAGES
        assert "flask" in _DEFAULT_ALLOWED_PACKAGES
        assert "fastapi" in _DEFAULT_ALLOWED_PACKAGES
        assert "numpy" in _DEFAULT_ALLOWED_PACKAGES
        assert "openai" in _DEFAULT_ALLOWED_PACKAGES
        assert "anthropic" in _DEFAULT_ALLOWED_PACKAGES
        assert "psutil" in _DEFAULT_ALLOWED_PACKAGES
        assert "docker" in _DEFAULT_ALLOWED_PACKAGES
        assert "paramiko" in _DEFAULT_ALLOWED_PACKAGES
        assert "matplotlib" in _DEFAULT_ALLOWED_PACKAGES


class TestValidateSkillLocal:
    """validate_skill_local() — lenient validation for local/evolved skills."""

    def test_allows_subprocess(self):
        code = 'import subprocess\nsubprocess.run(["open", "https://example.com"])'
        result = validate_skill_local(code)
        assert result.safe is True

    def test_allows_os_system(self):
        code = 'import os\nos.system("open https://example.com")'
        result = validate_skill_local(code)
        assert result.safe is True

    def test_allows_eval_exec(self):
        code = 'result = eval("1+1")\nexec("x=1")'
        result = validate_skill_local(code)
        assert result.safe is True

    def test_allows_socket(self):
        code = 'import socket\ns = socket.socket(socket.AF_INET, socket.SOCK_STREAM)'
        result = validate_skill_local(code)
        assert result.safe is True

    def test_allows_file_operations(self):
        code = 'import os\nos.remove("/tmp/test.txt")\nimport shutil\nshutil.rmtree("/tmp/data")'
        result = validate_skill_local(code)
        assert result.safe is True

    def test_allows_smtplib(self):
        code = 'import smtplib\nsmtplib.SMTP("mail.example.com")'
        result = validate_skill_local(code)
        assert result.safe is True

    def test_allows_dunder_import(self):
        code = 'm = __import__("os")'
        result = validate_skill_local(code)
        assert result.safe is True

    def test_rejects_setuid(self):
        code = 'import os\nos.setuid(0)'
        result = validate_skill_local(code)
        assert result.safe is False
        assert any("setuid" in e for e in result.errors)

    def test_rejects_setgid(self):
        code = 'import os\nos.setgid(0)'
        result = validate_skill_local(code)
        assert result.safe is False

    def test_rejects_ctypes_cdll(self):
        code = 'import ctypes\nctypes.cdll.LoadLibrary("libc.so")'
        result = validate_skill_local(code)
        assert result.safe is False

    def test_rejects_rm_rf_root(self):
        code = '''os.system("rm -rf /")'''
        result = validate_skill_local(code)
        assert result.safe is False

    def test_rejects_rmtree_root(self):
        code = '''shutil.rmtree("/")'''
        result = validate_skill_local(code)
        assert result.safe is False

    def test_warns_on_ctypes_import(self):
        """ctypes import (without cdll) generates warning, not error."""
        code = 'import ctypes\nprint(ctypes.sizeof(ctypes.c_int))'
        result = validate_skill_local(code)
        assert any("ctypes" in w for w in result.warnings)

    def test_empty_rejected(self):
        result = validate_skill_local("")
        assert result.safe is False
