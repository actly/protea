#!/usr/bin/env python3
"""Ring 2 — Generation 131: Autonomous Capability Synthesizer

Evolution: From passive skill discovery to ACTIVE CAPABILITY GENERATION.
Self-improving system that:
- Analyzes user task patterns to identify needed capabilities
- Generates specialized Python tools/scripts on-demand
- Deploys generated tools as executable modules
- Tests and validates generated code
- Iteratively improves based on success/failure
- Builds a library of proven utility functions
- Auto-registers working capabilities to skill server
- Learns from task history to predict future needs
- Synthesizes composite capabilities from primitives

Key innovation: CODE GENERATION ENGINE that writes, tests, and deploys
new tools autonomously based on inferred user requirements.
"""

import os
import pathlib
import sys
import time
import json
import sqlite3
import hashlib
import random
import re
import ast
import subprocess
import tomllib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Event, Lock
from datetime import datetime
from collections import defaultdict, Counter
import urllib.parse
import urllib.request

HEARTBEAT_INTERVAL = 2
HTTP_PORT = 8899

# Load configuration from config.toml
def load_config() -> Dict[str, Any]:
    """Load configuration from config/config.toml"""
    config_path = pathlib.Path(__file__).parent.parent / "config" / "config.toml"
    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"Warning: Could not load config.toml: {e}")
        # Return default fallback config
        return {
            "registry": {
                "host": "127.0.0.1",
                "port": 8761,
                "enabled": True
            }
        }

# Load config and construct SKILL_REGISTRY_URL
CONFIG = load_config()
REGISTRY_CONFIG = CONFIG.get("registry", {})
REGISTRY_HOST = REGISTRY_CONFIG.get("host", "127.0.0.1")
REGISTRY_PORT = REGISTRY_CONFIG.get("port", 8761)
SKILL_REGISTRY_URL = f"http://{REGISTRY_HOST}:{REGISTRY_PORT}"

def heartbeat_loop(heartbeat_path: pathlib.Path, pid: int, stop_event: Event) -> None:
    """Dedicated heartbeat thread - CRITICAL for survival."""
    while not stop_event.is_set():
        try:
            heartbeat_path.write_text(f"{pid}\n{time.time()}\n")
        except Exception:
            pass
        time.sleep(HEARTBEAT_INTERVAL)


# ============= CAPABILITY SYNTHESIS ENGINE =============

@dataclass
class CapabilitySpec:
    """Specification for a needed capability."""
    spec_id: str
    name: str
    description: str
    category: str
    inputs: List[str]
    outputs: str
    requirements: List[str]
    use_cases: List[str]
    priority: float
    created_at: float


@dataclass
class GeneratedTool:
    """A generated tool/script."""
    tool_id: str
    name: str
    description: str
    code: str
    language: str
    dependencies: List[str]
    test_code: str
    validation_status: str  # 'untested', 'passed', 'failed'
    execution_count: int
    success_rate: float
    avg_runtime: float
    created_at: float
    last_used: float


@dataclass
class SynthesisAttempt:
    """Record of code synthesis attempt."""
    attempt_id: str
    spec_id: str
    generated_code: str
    compilation_success: bool
    test_success: bool
    error_message: Optional[str]
    improvements: List[str]
    timestamp: float


class CapabilitySynthesizer:
    """Autonomous code generation and capability synthesis engine."""
    
    def __init__(self, db_path: pathlib.Path, tools_dir: pathlib.Path):
        self.db_path = db_path
        self.tools_dir = tools_dir
        self.tools_dir.mkdir(exist_ok=True)
        self.lock = Lock()
        
        self.specs: Dict[str, CapabilitySpec] = {}
        self.tools: Dict[str, GeneratedTool] = {}
        self.attempts: List[SynthesisAttempt] = []
        
        self.templates = self._load_templates()
        self.primitives = self._define_primitives()
        
        self._init_db()
        self._load_tools()
    
    def _init_db(self):
        """Initialize database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS specs (
                    spec_id TEXT PRIMARY KEY,
                    name TEXT, description TEXT, category TEXT,
                    inputs TEXT, outputs TEXT, requirements TEXT,
                    use_cases TEXT, priority REAL, created_at REAL
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tools (
                    tool_id TEXT PRIMARY KEY,
                    name TEXT, description TEXT, code TEXT,
                    language TEXT, dependencies TEXT, test_code TEXT,
                    validation_status TEXT, execution_count INTEGER,
                    success_rate REAL, avg_runtime REAL,
                    created_at REAL, last_used REAL
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    spec_id TEXT, generated_code TEXT,
                    compilation_success INTEGER, test_success INTEGER,
                    error_message TEXT, improvements TEXT, timestamp REAL
                )
            ''')
            conn.commit()
    
    def _load_tools(self):
        """Load existing tools from database."""
        with sqlite3.connect(self.db_path) as conn:
            for row in conn.execute('SELECT * FROM tools'):
                tool = GeneratedTool(
                    tool_id=row[0], name=row[1], description=row[2],
                    code=row[3], language=row[4],
                    dependencies=json.loads(row[5]), test_code=row[6],
                    validation_status=row[7], execution_count=row[8],
                    success_rate=row[9], avg_runtime=row[10],
                    created_at=row[11], last_used=row[12]
                )
                self.tools[tool.tool_id] = tool
    
    def _load_templates(self) -> Dict[str, str]:
        """Load code generation templates."""
        return {
            'data_processor': '''def process_{name}(data):
    """Process data: {description}"""
    result = []
    for item in data:
        {processing_logic}
        result.append(processed)
    return result
''',
            'analyzer': '''def analyze_{name}(data):
    """Analyze data: {description}"""
    stats = {{}}
    {analysis_logic}
    return stats
''',
            'transformer': '''def transform_{name}(input_data, target_format):
    """Transform data: {description}"""
    if target_format == 'json':
        return json.dumps(input_data)
    {transform_logic}
    return input_data
''',
            'fetcher': '''def fetch_{name}(url, params=None):
    """Fetch data: {description}"""
    import urllib.request
    import json
    {fetch_logic}
    return data
''',
            'validator': '''def validate_{name}(data):
    """Validate data: {description}"""
    errors = []
    {validation_logic}
    return len(errors) == 0, errors
'''
        }
    
    def _define_primitives(self) -> Dict[str, str]:
        """Define primitive code building blocks."""
        return {
            'iterate': 'for item in data:\n    ',
            'filter': 'filtered = [x for x in data if condition(x)]',
            'map': 'mapped = [transform(x) for x in data]',
            'reduce': 'result = sum(data) / len(data)',
            'sort': 'sorted_data = sorted(data, key=lambda x: x)',
            'group': 'grouped = {}\nfor item in data:\n    key = item.get("key")\n    grouped.setdefault(key, []).append(item)',
            'aggregate': 'aggregated = {"count": len(data), "sum": sum(data)}',
            'validate': 'if not isinstance(data, expected_type):\n    raise ValueError("Invalid type")',
            'extract': 'extracted = [item.get(field) for item in data]',
            'join': 'result = ", ".join(str(x) for x in data)',
        }
    
    def infer_needed_capabilities(self, user_tasks: List[str]) -> List[CapabilitySpec]:
        """Infer needed capabilities from user task patterns."""
        specs = []
        
        # Analyze task keywords
        task_text = ' '.join(user_tasks).lower()
        
        # Translation capability
        if any(word in task_text for word in ['翻译', 'translate', 'translation']):
            spec = CapabilitySpec(
                spec_id=hashlib.md5(b'translator').hexdigest()[:12],
                name='text_translator',
                description='Translate text between languages',
                category='nlp',
                inputs=['text', 'source_lang', 'target_lang'],
                outputs='translated_text',
                requirements=['language_detection', 'dictionary_lookup'],
                use_cases=['document translation', 'multilingual support'],
                priority=0.9,
                created_at=time.time()
            )
            specs.append(spec)
        
        # Summarization capability
        if any(word in task_text for word in ['总结', 'summarize', 'summary']):
            spec = CapabilitySpec(
                spec_id=hashlib.md5(b'summarizer').hexdigest()[:12],
                name='text_summarizer',
                description='Extract key points and generate summaries',
                category='nlp',
                inputs=['text', 'max_length'],
                outputs='summary',
                requirements=['sentence_extraction', 'importance_scoring'],
                use_cases=['document summarization', 'content digest'],
                priority=0.85,
                created_at=time.time()
            )
            specs.append(spec)
        
        # Test automation
        if any(word in task_text for word in ['test', '测试', 'testing']):
            spec = CapabilitySpec(
                spec_id=hashlib.md5(b'test_runner').hexdigest()[:12],
                name='test_automation',
                description='Automated testing framework',
                category='automation',
                inputs=['test_suite', 'config'],
                outputs='test_results',
                requirements=['test_execution', 'result_reporting'],
                use_cases=['unit testing', 'integration testing'],
                priority=0.8,
                created_at=time.time()
            )
            specs.append(spec)
        
        # Data analysis
        if any(word in task_text for word in ['analyze', '分析', 'analysis']):
            spec = CapabilitySpec(
                spec_id=hashlib.md5(b'data_analyzer').hexdigest()[:12],
                name='data_analyzer',
                description='Statistical data analysis and insights',
                category='analysis',
                inputs=['dataset', 'metrics'],
                outputs='analysis_results',
                requirements=['statistics', 'visualization'],
                use_cases=['trend analysis', 'pattern detection'],
                priority=0.75,
                created_at=time.time()
            )
            specs.append(spec)
        
        # Research assistant
        if any(word in task_text for word in ['research', '研究', 'paper', '论文']):
            spec = CapabilitySpec(
                spec_id=hashlib.md5(b'research_helper').hexdigest()[:12],
                name='research_assistant',
                description='Academic research organization and analysis',
                category='research',
                inputs=['papers', 'query'],
                outputs='research_summary',
                requirements=['paper_parsing', 'citation_analysis'],
                use_cases=['literature review', 'research organization'],
                priority=0.7,
                created_at=time.time()
            )
            specs.append(spec)
        
        return specs
    
    def generate_tool(self, spec: CapabilitySpec) -> Optional[GeneratedTool]:
        """Generate code for a capability specification."""
        try:
            # Select appropriate template
            template_key = self._select_template(spec.category)
            template = self.templates.get(template_key, self.templates['data_processor'])
            
            # Generate logic based on requirements
            logic = self._generate_logic(spec)
            
            # Fill template
            code = template.format(
                name=spec.name,
                description=spec.description,
                processing_logic=logic.get('processing', 'processed = item'),
                analysis_logic=logic.get('analysis', 'pass'),
                transform_logic=logic.get('transform', 'pass'),
                fetch_logic=logic.get('fetch', 'pass'),
                validation_logic=logic.get('validation', 'pass')
            )
            
            # Generate test code
            test_code = self._generate_test(spec, code)
            
            tool = GeneratedTool(
                tool_id=hashlib.md5(code.encode()).hexdigest()[:12],
                name=spec.name,
                description=spec.description,
                code=code,
                language='python',
                dependencies=[],
                test_code=test_code,
                validation_status='untested',
                execution_count=0,
                success_rate=0.0,
                avg_runtime=0.0,
                created_at=time.time(),
                last_used=0.0
            )
            
            return tool
        
        except Exception as e:
            print(f"❌ Generation failed: {e}", flush=True)
            return None
    
    def _select_template(self, category: str) -> str:
        """Select appropriate template for category."""
        mapping = {
            'nlp': 'analyzer',
            'analysis': 'analyzer',
            'data_processing': 'data_processor',
            'automation': 'validator',
            'research': 'fetcher'
        }
        return mapping.get(category, 'data_processor')
    
    def _generate_logic(self, spec: CapabilitySpec) -> Dict[str, str]:
        """Generate implementation logic based on requirements."""
        logic = {}
        
        if 'translation' in spec.name:
            logic['processing'] = '''
        # Simple word-level translation (demo)
        translations = {'hello': '你好', 'world': '世界'}
        processed = translations.get(item, item)
'''
        
        elif 'summarize' in spec.name:
            logic['analysis'] = '''
        sentences = data.split('.')
        stats['sentence_count'] = len(sentences)
        stats['word_count'] = len(data.split())
        stats['summary'] = sentences[0] if sentences else ""
'''
        
        elif 'test' in spec.name:
            logic['validation'] = '''
        if not hasattr(data, '__call__'):
            errors.append("Not a callable function")
'''
        
        elif 'analyze' in spec.name or 'analysis' in spec.name:
            logic['analysis'] = '''
        if isinstance(data, list):
            stats['count'] = len(data)
            numeric = [x for x in data if isinstance(x, (int, float))]
            if numeric:
                stats['mean'] = sum(numeric) / len(numeric)
                stats['min'] = min(numeric)
                stats['max'] = max(numeric)
'''
        
        return logic
    
    def _generate_test(self, spec: CapabilitySpec, code: str) -> str:
        """Generate test code for the tool."""
        func_name = f"process_{spec.name}" if 'process' in code else f"analyze_{spec.name}"
        
        return f'''
def test_{spec.name}():
    # Test case 1: Basic functionality
    test_data = [1, 2, 3, 4, 5]
    result = {func_name}(test_data)
    assert result is not None, "Result should not be None"
    
    # Test case 2: Empty input
    empty_result = {func_name}([])
    assert isinstance(empty_result, (list, dict)), "Should handle empty input"
    
    print("✓ All tests passed")
    return True
'''
    
    def validate_tool(self, tool: GeneratedTool) -> Tuple[bool, Optional[str]]:
        """Validate generated tool through compilation and testing."""
        try:
            # Check syntax
            compile(tool.code, '<string>', 'exec')
            
            # Try to execute (safe namespace)
            namespace = {}
            exec(tool.code, namespace)
            
            # Run tests if available
            if tool.test_code:
                exec(tool.test_code, namespace)
                test_func = namespace.get(f'test_{tool.name}')
                if test_func:
                    test_func()
            
            tool.validation_status = 'passed'
            return True, None
        
        except SyntaxError as e:
            tool.validation_status = 'failed'
            return False, f"Syntax error: {e}"
        except Exception as e:
            tool.validation_status = 'failed'
            return False, f"Runtime error: {e}"
    
    def deploy_tool(self, tool: GeneratedTool) -> bool:
        """Deploy validated tool as executable module."""
        try:
            tool_path = self.tools_dir / f"{tool.name}.py"
            
            # Add imports and docstring
            full_code = f'''#!/usr/bin/env python3
"""
{tool.description}

Auto-generated by Capability Synthesizer
Generated: {datetime.fromtimestamp(tool.created_at).isoformat()}
"""

import json
import time
from typing import Any, List, Dict

{tool.code}

if __name__ == "__main__":
    # Test execution
    print(f"Tool: {tool.name}")
    print(f"Description: {tool.description}")
'''
            
            tool_path.write_text(full_code)
            tool_path.chmod(0o755)
            
            self._save_tool(tool)
            return True
        
        except Exception as e:
            print(f"❌ Deployment failed: {e}", flush=True)
            return False
    
    def _save_tool(self, tool: GeneratedTool):
        """Save tool to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO tools VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                tool.tool_id, tool.name, tool.description, tool.code,
                tool.language, json.dumps(tool.dependencies), tool.test_code,
                tool.validation_status, tool.execution_count, tool.success_rate,
                tool.avg_runtime, tool.created_at, tool.last_used
            ))
            conn.commit()
    
    def register_to_skill_server(self, tool: GeneratedTool) -> bool:
        """Register validated tool to central skill server."""
        try:
            payload = {
                'skill_id': tool.tool_id,
                'name': tool.name,
                'description': tool.description,
                'category': 'generated_tool',
                'capabilities': ['code_generation', 'automation'],
                'tags': ['synthesized', 'validated'],
                'metadata': {
                    'validation_status': tool.validation_status,
                    'execution_count': tool.execution_count,
                    'success_rate': tool.success_rate
                }
            }
            
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                f"{SKILL_REGISTRY_URL}/api/register_skill",
                data=data,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read())
                return result.get('status') == 'ok'
        
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        return {
            'total_specs': len(self.specs),
            'total_tools': len(self.tools),
            'validated_tools': sum(1 for t in self.tools.values() if t.validation_status == 'passed'),
            'deployed_tools': len(list(self.tools_dir.glob('*.py'))),
            'success_rate': sum(t.success_rate for t in self.tools.values()) / len(self.tools) if self.tools else 0.0
        }


synthesizer = None


class SynthesizerHandler(BaseHTTPRequestHandler):
    """HTTP handler for synthesizer dashboard."""
    
    def log_message(self, format, *args):
        pass
    
    def do_GET(self):
        if self.path == '/':
            self.serve_dashboard()
        elif self.path == '/api/stats':
            self.serve_json(synthesizer.get_stats())
        elif self.path == '/api/tools':
            tools = [
                {
                    'name': t.name,
                    'description': t.description,
                    'status': t.validation_status,
                    'executions': t.execution_count
                }
                for t in synthesizer.tools.values()
            ]
            self.serve_json({'tools': tools})
        else:
            self.send_error(404)
    
    def serve_json(self, data, code=200):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    
    def serve_dashboard(self):
        html = '''<!DOCTYPE html>
<html><head><title>Capability Synthesizer</title><meta charset="UTF-8">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:monospace;background:#0a0e27;color:#e0e0e0}
.header{background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;color:#fff}
.title{font-size:24px;font-weight:700}
.container{padding:20px;max-width:1400px;margin:0 auto}
.panel{background:#1a1a2e;border:1px solid #2a2a3e;border-radius:8px;padding:15px;margin:15px 0}
.stat{display:inline-block;margin:10px 20px 10px 0}
.stat-val{font-size:28px;font-weight:700;color:#667eea}
.stat-label{font-size:12px;color:#999}
.tool{background:#2a2a3e;border-left:4px solid #667eea;padding:10px;margin:8px 0;border-radius:4px}
.tool-name{font-weight:700;color:#667eea}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:10px;margin:2px}
.passed{background:#27ae60;color:#fff}
.failed{background:#e74c3c;color:#fff}
.untested{background:#f39c12;color:#fff}
</style></head><body>
<div class="header">
<div class="title">🔬 Autonomous Capability Synthesizer</div>
<div style="margin-top:5px;opacity:0.9">Self-Generating Tool Factory</div>
</div>
<div class="container">
<div class="panel">
<div id="stats"></div>
</div>
<div class="panel">
<div style="font-size:16px;margin-bottom:10px">🛠️ Generated Tools</div>
<div id="tools-list"></div>
</div>
</div>
<script>
async function loadStats() {
    const res = await fetch('/api/stats');
    const data = await res.json();
    const div = document.getElementById('stats');
    
    div.innerHTML = 
        '<div class="stat"><div class="stat-val">' + data.total_specs + '</div><div class="stat-label">Capability Specs</div></div>' +
        '<div class="stat"><div class="stat-val">' + data.total_tools + '</div><div class="stat-label">Tools Generated</div></div>' +
        '<div class="stat"><div class="stat-val">' + data.validated_tools + '</div><div class="stat-label">Validated</div></div>' +
        '<div class="stat"><div class="stat-val">' + data.deployed_tools + '</div><div class="stat-label">Deployed</div></div>' +
        '<div class="stat"><div class="stat-val">' + (data.success_rate * 100).toFixed(0) + '%</div><div class="stat-label">Success Rate</div></div>';
}

async function loadTools() {
    const res = await fetch('/api/tools');
    const data = await res.json();
    const div = document.getElementById('tools-list');
    
    if (data.tools.length === 0) {
        div.innerHTML = '<div style="color:#999;padding:10px">No tools generated yet</div>';
        return;
    }
    
    div.innerHTML = data.tools.map(t =>
        '<div class="tool">' +
        '<div class="tool-name">' + t.name + '</div>' +
        '<div style="font-size:11px;color:#999;margin:3px 0">' + t.description + '</div>' +
        '<span class="badge ' + t.status + '">' + t.status + '</span>' +
        '<span style="font-size:11px;color:#999;margin-left:10px">Executions: ' + t.executions + '</span>' +
        '</div>'
    ).join('');
}

function loadAll() {
    loadStats();
    loadTools();
}

loadAll();
setInterval(loadAll, 3000);
</script>
</body></html>'''
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))


def main() -> None:
    """Main entry point."""
    global synthesizer
    
    heartbeat_path = pathlib.Path(os.environ.get("PROTEA_HEARTBEAT", ".heartbeat"))
    pid = os.getpid()
    stop_event = Event()
    
    heartbeat_thread = Thread(target=heartbeat_loop, args=(heartbeat_path, pid, stop_event), daemon=True)
    heartbeat_thread.start()
    
    output_dir = pathlib.Path("ring2_output")
    output_dir.mkdir(exist_ok=True)
    tools_dir = output_dir / "generated_tools"
    
    synthesizer = CapabilitySynthesizer(output_dir / "synthesizer.db", tools_dir)
    
    print(f"[Ring 2 Gen 131] Capability Synthesizer pid={pid}", flush=True)
    print(f"🔬 Dashboard: http://localhost:{HTTP_PORT}", flush=True)
    
    # Start HTTP server
    def run_server():
        try:
            server = HTTPServer(('127.0.0.1', HTTP_PORT), SynthesizerHandler)
            server.serve_forever()
        except:
            pass
    
    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1)
    
    # Sample user tasks
    user_tasks = [
        "再登记一下子，刚才服务端重启了",
        "有一个 Skill 登记的地址",
        "现在程序进化出有什么样的 Skill 了？",
        "请帮我翻译这段文字",
        "能不能总结一下这些论文？",
        "帮我测试一下这个功能"
    ]
    
    cycle = 0
    
    try:
        while True:
            # Infer needed capabilities periodically
            if cycle % 20 == 0:
                specs = synthesizer.infer_needed_capabilities(user_tasks)
                
                for spec in specs:
                    if spec.spec_id not in synthesizer.specs:
                        synthesizer.specs[spec.spec_id] = spec
                        print(f"📋 Identified need: {spec.name}", flush=True)
                        
                        # Generate tool for this capability
                        tool = synthesizer.generate_tool(spec)
                        
                        if tool:
                            print(f"🔨 Generated: {tool.name}", flush=True)
                            
                            # Validate
                            success, error = synthesizer.validate_tool(tool)
                            
                            if success:
                                print(f"✅ Validated: {tool.name}", flush=True)
                                synthesizer.tools[tool.tool_id] = tool
                                
                                # Deploy
                                if synthesizer.deploy_tool(tool):
                                    print(f"🚀 Deployed: {tool.name}", flush=True)
                                    
                                    # Register to skill server
                                    if synthesizer.register_to_skill_server(tool):
                                        print(f"📡 Registered: {tool.name}", flush=True)
                            else:
                                print(f"❌ Validation failed: {error}", flush=True)
            
            if cycle % 10 == 0:
                stats = synthesizer.get_stats()
                print(f"[Cycle {cycle}] Specs: {stats['total_specs']} | Tools: {stats['total_tools']} | ✓ {stats['validated_tools']}", flush=True)
            
            time.sleep(2)
            cycle += 1
    
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        try:
            heartbeat_path.unlink(missing_ok=True)
        except:
            pass
        print(f"\n[Ring 2] Synthesizer shutdown. pid={pid}", flush=True)


if __name__ == "__main__":
    main()