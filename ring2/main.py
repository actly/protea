#!/usr/bin/env python3
"""Ring 2 — Generation 314: Protea Memory Archaeologist

Focus: Excavate and analyze user's long-term memory patterns and forgotten interactions.
Addresses user's question about memory persistence and recall strategies.
"""

import os
import pathlib
import time
import json
import re
from threading import Thread, Event
from typing import Dict, List, Optional
from collections import defaultdict, Counter

HEARTBEAT_INTERVAL = 2


def heartbeat_loop(heartbeat_path: pathlib.Path, pid: int, stop_event: Event) -> None:
    """Dedicated heartbeat thread - CRITICAL for survival."""
    while not stop_event.is_set():
        try:
            heartbeat_path.write_text(f"{pid}\n{time.time()}\n")
        except Exception:
            pass
        time.sleep(HEARTBEAT_INTERVAL)


# ============= MEMORY ARCHAEOLOGIST =============

class MemoryArchaeologist:
    """Excavate and analyze historical memory patterns."""
    
    @staticmethod
    def excavate_task_memories(base_path: pathlib.Path) -> Dict:
        """Dig through task history to find forgotten memories."""
        task_file = base_path / "tasks.jsonl"
        
        if not task_file.exists():
            return {'error': 'tasks.jsonl not found', 'total_tasks': 0, 'memories': []}
        
        try:
            memories = []
            with open(task_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            task = json.loads(line)
                            memories.append({
                                'timestamp': task.get('timestamp', 0),
                                'content': task.get('content', '')[:200],
                                'generation': task.get('generation', 0),
                                'age_days': (time.time() - task.get('timestamp', time.time())) / 86400
                            })
                        except json.JSONDecodeError:
                            continue
            
            # Sort by age (oldest first)
            memories.sort(key=lambda x: x['timestamp'])
            
            return {
                'total_tasks': len(memories),
                'memories': memories,
                'oldest': memories[0] if memories else None,
                'newest': memories[-1] if memories else None
            }
        except Exception as e:
            return {'error': str(e), 'total_tasks': 0, 'memories': []}
    
    @staticmethod
    def analyze_memory_layers(memories: List[Dict]) -> Dict:
        """Stratify memories into temporal layers."""
        if not memories:
            return {'layers': {}}
        
        layers = {
            'ancient': [],      # > 30 days
            'old': [],          # 7-30 days
            'recent': [],       # 1-7 days
            'fresh': []         # < 1 day
        }
        
        for mem in memories:
            age = mem['age_days']
            if age > 30:
                layers['ancient'].append(mem)
            elif age > 7:
                layers['old'].append(mem)
            elif age > 1:
                layers['recent'].append(mem)
            else:
                layers['fresh'].append(mem)
        
        return {
            'layers': {k: len(v) for k, v in layers.items()},
            'samples': {k: v[:3] for k, v in layers.items() if v}
        }
    
    @staticmethod
    def extract_forgotten_themes(memories: List[Dict]) -> Dict:
        """Find themes that appear in old but not recent memories."""
        if len(memories) < 10:
            return {'themes': {}}
        
        # Split into ancient and recent
        split_point = len(memories) // 2
        ancient = memories[:split_point]
        recent = memories[split_point:]
        
        # Extract keywords
        def extract_keywords(mem_list):
            words = []
            for mem in mem_list:
                content = mem['content'].lower()
                # Extract Chinese and English words
                words.extend(re.findall(r'[\u4e00-\u9fff]+', content))
                words.extend(re.findall(r'[a-z]{3,}', content))
            return Counter(words)
        
        ancient_words = extract_keywords(ancient)
        recent_words = extract_keywords(recent)
        
        # Find forgotten themes (high in ancient, low in recent)
        forgotten = {}
        for word, count in ancient_words.most_common(20):
            if count >= 2 and recent_words[word] <= 1:
                forgotten[word] = {
                    'ancient_count': count,
                    'recent_count': recent_words[word],
                    'forgotten_ratio': count / (recent_words[word] + 1)
                }
        
        return {'themes': forgotten}


# ============= MEMORY PERSISTENCE ANALYZER =============

class MemoryPersistenceAnalyzer:
    """Analyze what should be kept vs. discarded."""
    
    @staticmethod
    def score_memory_importance(memory: Dict) -> float:
        """Score how important a memory is for long-term storage."""
        score = 0.5  # Base score
        
        content = memory.get('content', '').lower()
        
        # High-value indicators
        if any(kw in content for kw in ['创业', '讨论', '逻辑', '分析', '研究']):
            score += 0.15
        
        if any(kw in content for kw in ['startup', 'business', 'strategy', 'analyze']):
            score += 0.15
        
        # Questions indicate important interactions
        if '?' in content or '？' in content:
            score += 0.10
        
        # Length suggests depth
        if len(content) > 100:
            score += 0.10
        
        # Entities (names, places)
        if re.search(r'[\u4e00-\u9fff]{2,4}(鞋业|科技|公司)', content):
            score += 0.15
        
        return min(score, 1.0)
    
    @staticmethod
    def recommend_archival_strategy(memories: List[Dict]) -> Dict:
        """Recommend what to archive vs. discard."""
        if not memories:
            return {'strategy': 'no_data'}
        
        scored = [(mem, MemoryPersistenceAnalyzer.score_memory_importance(mem)) 
                  for mem in memories]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Top 20% should be permanently archived
        archive_count = max(1, len(scored) // 5)
        to_archive = scored[:archive_count]
        
        # Next 30% kept in working memory
        working_count = max(1, len(scored) * 3 // 10)
        to_working = scored[archive_count:archive_count + working_count]
        
        # Rest can be compressed/summarized
        to_compress = scored[archive_count + working_count:]
        
        return {
            'total_memories': len(memories),
            'archive_count': len(to_archive),
            'working_count': len(to_working),
            'compress_count': len(to_compress),
            'archive_samples': [m[0]['content'][:80] for m in to_archive[:3]],
            'avg_archive_score': sum(s for _, s in to_archive) / len(to_archive) if to_archive else 0
        }


# ============= MEMORY RECALL SIMULATOR =============

class MemoryRecallSimulator:
    """Simulate occasional recall of old memories."""
    
    @staticmethod
    def random_recall(memories: List[Dict], count: int = 3) -> List[Dict]:
        """Randomly recall old memories for context refresh."""
        if not memories or len(memories) < count:
            return memories
        
        # Bias toward older memories
        import random
        weights = [1.0 / (i + 1) for i in range(len(memories))]
        
        try:
            recalled = random.choices(memories, weights=weights, k=count)
            return recalled
        except Exception:
            return memories[:count]
    
    @staticmethod
    def contextual_recall(memories: List[Dict], current_context: str) -> List[Dict]:
        """Recall memories relevant to current context."""
        if not memories:
            return []
        
        context_lower = current_context.lower()
        context_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-z]{3,}', context_lower))
        
        scored = []
        for mem in memories:
            mem_words = set(re.findall(r'[\u4e00-\u9fff]+|[a-z]{3,}', 
                                       mem['content'].lower()))
            overlap = len(context_words & mem_words)
            if overlap > 0:
                scored.append((mem, overlap))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:5]]


# ============= REPORTER =============

class MemoryReport:
    """Generate memory analysis reports."""
    
    @staticmethod
    def generate_report(base_path: pathlib.Path) -> None:
        """Generate comprehensive memory analysis."""
        print("\n" + "="*70, flush=True)
        print("记忆考古报告 (Memory Archaeological Report)", flush=True)
        print("="*70, flush=True)
        
        # Excavate memories
        excavation = MemoryArchaeologist.excavate_task_memories(base_path)
        
        if excavation.get('error'):
            print(f"\n⚠ 错误 (Error): {excavation['error']}", flush=True)
            return
        
        total = excavation['total_tasks']
        memories = excavation['memories']
        
        print(f"\n总记忆数 (Total Memories): {total}", flush=True)
        
        if excavation.get('oldest'):
            oldest = excavation['oldest']
            print(f"最古老记忆 (Oldest): {oldest['age_days']:.1f} days ago", flush=True)
            print(f"  内容: {oldest['content'][:100]}...", flush=True)
        
        # Analyze layers
        layers = MemoryArchaeologist.analyze_memory_layers(memories)
        
        print(f"\n记忆分层 (Memory Layers):", flush=True)
        for layer, count in layers.get('layers', {}).items():
            bar = '█' * min(count // 5, 40)
            print(f"  {layer:>8}: {bar} ({count})", flush=True)
        
        # Forgotten themes
        forgotten = MemoryArchaeologist.extract_forgotten_themes(memories)
        
        if forgotten.get('themes'):
            print(f"\n被遗忘的主题 (Forgotten Themes):", flush=True)
            for theme, data in list(forgotten['themes'].items())[:8]:
                print(f"  • {theme}: 古代出现{data['ancient_count']}次, "
                      f"近期{data['recent_count']}次", flush=True)
        
        # Archival strategy
        strategy = MemoryPersistenceAnalyzer.recommend_archival_strategy(memories)
        
        print(f"\n存档策略建议 (Archival Strategy):", flush=True)
        print(f"  永久存档 (Archive): {strategy['archive_count']} "
              f"({strategy['archive_count']*100//total:.0f}%)", flush=True)
        print(f"  工作记忆 (Working): {strategy['working_count']} "
              f"({strategy['working_count']*100//total:.0f}%)", flush=True)
        print(f"  可压缩 (Compress): {strategy['compress_count']} "
              f"({strategy['compress_count']*100//total:.0f}%)", flush=True)
        
        if strategy.get('archive_samples'):
            print(f"\n高价值记忆样本 (High-Value Samples):", flush=True)
            for i, sample in enumerate(strategy['archive_samples'], 1):
                print(f"  {i}. {sample}...", flush=True)
        
        # Random recall simulation
        recalled = MemoryRecallSimulator.random_recall(memories, 3)
        
        print(f"\n随机召唤记忆 (Random Recall):", flush=True)
        for i, mem in enumerate(recalled, 1):
            print(f"  {i}. [{mem['age_days']:.0f}d ago] {mem['content'][:100]}...", 
                  flush=True)


# ============= MAIN =============

def main() -> None:
    """Main memory archaeology loop."""
    heartbeat_path = pathlib.Path(os.environ.get("PROTEA_HEARTBEAT", ".heartbeat"))
    pid = os.getpid()
    stop_event = Event()
    
    heartbeat_thread = Thread(target=heartbeat_loop, args=(heartbeat_path, pid, stop_event), daemon=True)
    heartbeat_thread.start()
    
    print(f"[Ring 2 Gen 314] Memory Archaeologist pid={pid}", flush=True)
    print("正在挖掘记忆遗迹... (Excavating memory artifacts...)", flush=True)
    
    base_path = pathlib.Path.cwd()
    
    cycle = 0
    
    try:
        while cycle < 15:  # More cycles to ensure survival
            print(f"\n{'='*70}", flush=True)
            print(f"考古周期 (Excavation Cycle) {cycle} — {time.strftime('%H:%M:%S')}", 
                  flush=True)
            print(f"{'='*70}", flush=True)
            
            MemoryReport.generate_report(base_path)
            
            print(f"\n周期完成. Heartbeat active: {heartbeat_thread.is_alive()}", flush=True)
            
            cycle += 1
            time.sleep(35)  # Longer sleep to ensure we hit multiple heartbeats
    
    except KeyboardInterrupt:
        print("\n中断信号收到 (Interrupt received)", flush=True)
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=5)  # Wait for thread to finish
        try:
            heartbeat_path.unlink(missing_ok=True)
        except Exception:
            pass
        
        print(f"\n[Ring 2] 记忆考古关闭. Cycles: {cycle}, pid={pid}", flush=True)


if __name__ == "__main__":
    main()