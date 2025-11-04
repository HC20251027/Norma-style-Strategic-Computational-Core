#!/usr/bin/env python3
"""
记忆管理器
负责管理对话历史、长期记忆和知识库

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from ..utils.logger import NormaLogger

@dataclass
class MemoryEntry:
    """记忆条目"""
    id: str
    session_id: str
    content: str
    memory_type: str  # short_term, long_term, knowledge, episodic
    importance: float  # 重要性评分 (0-1)
    timestamp: datetime
    metadata: Dict[str, Any]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }

class ConversationMemory:
    """对话记忆类"""
    
    def __init__(self, session_id: str, max_short_term: int = 50):
        self.session_id = session_id
        self.max_short_term = max_short_term
        
        # 短期记忆（最近对话）
        self.short_term_memory: deque = deque(maxlen=max_short_term)
        
        # 工作记忆（当前对话焦点）
        self.working_memory: Dict[str, Any] = {}
        
        # 记忆索引
        self.memory_index: Dict[str, List[str]] = defaultdict(list)  # key -> memory_ids
        
        # 重要性阈值
        self.importance_threshold = 0.6
        
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
    
    def add_memory(
        self,
        content: str,
        memory_type: str = "short_term",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """添加记忆"""
        
        memory_id = str(uuid.uuid4())
        
        memory_entry = MemoryEntry(
            id=memory_id,
            session_id=self.session_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # 根据记忆类型存储
        if memory_type == "short_term":
            self.short_term_memory.append(memory_entry)
        elif memory_type == "working":
            self.working_memory[memory_id] = memory_entry
        
        # 更新索引
        self._update_index(memory_entry)
        
        self.last_accessed = datetime.now()
        return memory_id
    
    def _update_index(self, memory_entry: MemoryEntry) -> None:
        """更新记忆索引"""
        
        # 基于内容关键词建立索引
        keywords = self._extract_keywords(memory_entry.content)
        for keyword in keywords:
            self.memory_index[keyword].append(memory_entry.id)
    
    def _extract_keywords(self, content: str) -> List[str]:
        """提取关键词"""
        
        # 简单的关键词提取（实际应用中可以使用更复杂的NLP技术）
        import re
        
        # 提取中文词汇和英文单词
        chinese_words = re.findall(r'[\u4e00-\u9fa5]+', content)
        english_words = re.findall(r'\b[a-zA-Z]+\b', content)
        
        # 过滤停用词
        stop_words = {"的", "了", "在", "是", "我", "你", "他", "她", "它", "这", "那", "有", "没", "不", "要", "会", "能", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        keywords = []
        for word in chinese_words + english_words:
            if len(word) > 1 and word not in stop_words:
                keywords.append(word.lower())
        
        return list(set(keywords))  # 去重
    
    def get_recent_memories(self, count: int = 10) -> List[MemoryEntry]:
        """获取最近的记忆"""
        return list(self.short_term_memory)[-count:]
    
    def get_working_memory(self) -> Dict[str, MemoryEntry]:
        """获取工作记忆"""
        self.last_accessed = datetime.now()
        return self.working_memory.copy()
    
    def search_memories(self, query: str, limit: int = 10) -> List[Tuple[MemoryEntry, float]]:
        """搜索记忆"""
        
        query_keywords = self._extract_keywords(query)
        scored_memories = []
        
        # 搜索短期记忆
        for memory in self.short_term_memory:
            score = self._calculate_relevance_score(memory, query_keywords)
            if score > 0.1:  # 最低相关性阈值
                scored_memories.append((memory, score))
        
        # 搜索工作记忆
        for memory in self.working_memory.values():
            score = self._calculate_relevance_score(memory, query_keywords)
            if score > 0.1:
                scored_memories.append((memory, score))
        
        # 按相关性排序
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return scored_memories[:limit]
    
    def _calculate_relevance_score(self, memory: MemoryEntry, query_keywords: List[str]) -> float:
        """计算相关性评分"""
        
        memory_keywords = self._extract_keywords(memory.content)
        
        # 计算关键词重叠度
        overlap = len(set(query_keywords) & set(memory_keywords))
        if not query_keywords:
            return 0.0
        
        keyword_score = overlap / len(query_keywords)
        
        # 考虑重要性权重
        importance_score = memory.importance
        
        # 考虑时间衰减（越新的记忆权重越高）
        time_diff = (datetime.now() - memory.timestamp).total_seconds()
        time_decay = max(0.1, 1.0 / (1.0 + time_diff / 86400))  # 24小时衰减
        
        # 综合评分
        final_score = (keyword_score * 0.4 + importance_score * 0.4 + time_decay * 0.2)
        
        return final_score
    
    def promote_to_long_term(self, memory_id: str) -> bool:
        """将记忆提升为长期记忆"""
        
        # 在短期记忆中查找
        for i, memory in enumerate(self.short_term_memory):
            if memory.id == memory_id:
                memory.memory_type = "long_term"
                memory.importance = min(1.0, memory.importance + 0.2)  # 提升重要性
                return True
        
        return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        
        short_term_count = len(self.short_term_memory)
        working_memory_count = len(self.working_memory)
        
        # 计算平均重要性
        all_memories = list(self.short_term_memory) + list(self.working_memory.values())
        avg_importance = sum(m.importance for m in all_memories) / len(all_memories) if all_memories else 0
        
        # 计算总访问次数
        total_accesses = sum(m.access_count for m in all_memories)
        
        return {
            "session_id": self.session_id,
            "short_term_count": short_term_count,
            "working_memory_count": working_memory_count,
            "total_memories": short_term_count + working_memory_count,
            "average_importance": avg_importance,
            "total_accesses": total_accesses,
            "index_size": len(self.memory_index),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat()
        }

class MemoryManager:
    """记忆管理器"""
    
    def __init__(self, conversation_limit: int = 100):
        """初始化记忆管理器
        
        Args:
            conversation_limit: 对话历史限制
        """
        self.conversation_limit = conversation_limit
        self.logger = NormaLogger("memory_manager")
        
        # 内存中的会话记忆
        self.session_memories: Dict[str, ConversationMemory] = {}
        
        # 数据库路径
        self.db_path = Path("/workspace/norma_agent/data/memory.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 记忆统计
        self.memory_stats = {
            "total_memories": 0,
            "total_sessions": 0,
            "average_importance": 0.0
        }
        
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """初始化记忆管理器"""
        try:
            self.logger.info("初始化记忆管理器...")
            
            # 初始化数据库
            await self._init_database()
            
            # 加载已存在的会话记忆
            await self._load_existing_memories()
            
            self.is_initialized = True
            self.logger.info("记忆管理器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"记忆管理器初始化失败: {e}")
            return False
    
    async def _init_database(self) -> None:
        """初始化数据库"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建记忆表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                importance REAL NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # 创建会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                memory_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)')
        
        conn.commit()
        conn.close()
        
        self.logger.info("记忆数据库初始化完成")
    
    async def _load_existing_memories(self) -> None:
        """加载已存在的记忆"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取所有活跃会话
            cursor.execute('''
                SELECT session_id, created_at, last_accessed, message_count, memory_count, status
                FROM sessions 
                WHERE status = 'active'
                ORDER BY last_accessed DESC
            ''')
            
            sessions = cursor.fetchall()
            
            for session_data in sessions:
                session_id = session_data[0]
                
                # 创建会话记忆对象
                conversation_memory = ConversationMemory(session_id)
                self.session_memories[session_id] = conversation_memory
                
                # 从数据库加载记忆
                await self._load_session_memories_from_db(session_id)
            
            conn.close()
            
            self.logger.info(f"已加载 {len(sessions)} 个活跃会话的记忆")
            
        except Exception as e:
            self.logger.error(f"加载已存在记忆失败: {e}")
    
    async def _load_session_memories_from_db(self, session_id: str) -> None:
        """从数据库加载会话记忆"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取会话的记忆
        cursor.execute('''
            SELECT id, content, memory_type, importance, timestamp, metadata, access_count, last_accessed
            FROM memories 
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT 50
        ''', (session_id,))
        
        memories = cursor.fetchall()
        
        for memory_data in memories:
            memory_entry = MemoryEntry(
                id=memory_data[0],
                session_id=session_id,
                content=memory_data[1],
                memory_type=memory_data[2],
                importance=memory_data[3],
                timestamp=datetime.fromisoformat(memory_data[4]),
                metadata=json.loads(memory_data[5]) if memory_data[5] else {},
                access_count=memory_data[6],
                last_accessed=datetime.fromisoformat(memory_data[7]) if memory_data[7] else None
            )
            
            # 添加到短期记忆
            if memory_entry.memory_type in ["short_term", "working"]:
                self.session_memories[session_id].short_term_memory.append(memory_entry)
            
            # 更新索引
            keywords = self.session_memories[session_id]._extract_keywords(memory_entry.content)
            for keyword in keywords:
                self.session_memories[session_id].memory_index[keyword].append(memory_entry.id)
        
        conn.close()
    
    async def create_session(self, session_id: str) -> bool:
        """创建新会话"""
        
        try:
            # 在数据库中创建会话记录
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO sessions 
                (session_id, created_at, last_accessed, message_count, memory_count, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                0,
                0,
                'active'
            ))
            
            conn.commit()
            conn.close()
            
            # 创建内存中的会话记忆
            self.session_memories[session_id] = ConversationMemory(session_id)
            
            self.logger.info(f"创建会话记忆: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"创建会话记忆失败 {session_id}: {e}")
            return False
    
    async def add_conversation_memory(
        self,
        session_id: str,
        content: str,
        memory_type: str = "short_term",
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """添加对话记忆"""
        
        try:
            # 确保会话存在
            if session_id not in self.session_memories:
                await self.create_session(session_id)
            
            # 添加到内存
            memory_id = self.session_memories[session_id].add_memory(
                content, memory_type, importance, metadata
            )
            
            # 保存到数据库
            await self._save_memory_to_db(session_id, memory_id)
            
            # 更新会话统计
            await self._update_session_stats(session_id)
            
            self.logger.debug(f"添加对话记忆 {session_id}: {memory_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"添加对话记忆失败 {session_id}: {e}")
            return None
    
    async def _save_memory_to_db(self, session_id: str, memory_id: str) -> None:
        """保存记忆到数据库"""
        
        memory = None
        for m in self.session_memories[session_id].short_term_memory:
            if m.id == memory_id:
                memory = m
                break
        
        if not memory:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO memories 
            (id, session_id, content, memory_type, importance, timestamp, metadata, access_count, last_accessed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            memory.id,
            memory.session_id,
            memory.content,
            memory.memory_type,
            memory.importance,
            memory.timestamp.isoformat(),
            json.dumps(memory.metadata, ensure_ascii=False),
            memory.access_count,
            memory.last_accessed.isoformat() if memory.last_accessed else None,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    async def _update_session_stats(self, session_id: str) -> None:
        """更新会话统计"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        memory_count = len(self.session_memories[session_id].short_term_memory)
        
        cursor.execute('''
            UPDATE sessions 
            SET last_accessed = ?, memory_count = ?
            WHERE session_id = ?
        ''', (
            datetime.now().isoformat(),
            memory_count,
            session_id
        ))
        
        conn.commit()
        conn.close()
    
    async def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取对话历史"""
        
        try:
            if session_id not in self.session_memories:
                return []
            
            memories = self.session_memories[session_id].get_recent_memories(limit)
            return [memory.to_dict() for memory in memories]
            
        except Exception as e:
            self.logger.error(f"获取对话历史失败 {session_id}: {e}")
            return []
    
    async def search_memories(
        self,
        session_id: str,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """搜索记忆"""
        
        try:
            if session_id not in self.session_memories:
                return []
            
            results = self.session_memories[session_id].search_memories(query, limit)
            return [
                {
                    "memory": memory.to_dict(),
                    "relevance_score": score
                }
                for memory, score in results
            ]
            
        except Exception as e:
            self.logger.error(f"搜索记忆失败 {session_id}: {e}")
            return []
    
    async def get_session_memory_stats(self, session_id: str) -> Dict[str, Any]:
        """获取会话记忆统计"""
        
        try:
            if session_id not in self.session_memories:
                return {"session_id": session_id, "status": "not_found"}
            
            return self.session_memories[session_id].get_memory_stats()
            
        except Exception as e:
            self.logger.error(f"获取会话记忆统计失败 {session_id}: {e}")
            return {"session_id": session_id, "status": "error", "error": str(e)}
    
    async def archive_session(self, session_id: str) -> bool:
        """归档会话"""
        
        try:
            # 更新数据库中的会话状态
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE sessions 
                SET status = 'archived', last_accessed = ?
                WHERE session_id = ?
            ''', (datetime.now().isoformat(), session_id))
            
            conn.commit()
            conn.close()
            
            # 从内存中移除
            if session_id in self.session_memories:
                del self.session_memories[session_id]
            
            self.logger.info(f"归档会话: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"归档会话失败 {session_id}: {e}")
            return False
    
    async def cleanup_old_memories(self, days: int = 30) -> int:
        """清理旧记忆"""
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 删除旧的低重要性记忆
            cursor.execute('''
                DELETE FROM memories 
                WHERE timestamp < ? AND importance < 0.3
            ''', (cutoff_date.isoformat(),))
            
            deleted_count = cursor.rowcount
            
            # 清理无会话引用的记忆
            cursor.execute('''
                DELETE FROM memories 
                WHERE session_id NOT IN (SELECT session_id FROM sessions WHERE status = 'active')
            ''')
            
            orphaned_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            total_cleaned = deleted_count + orphaned_count
            
            self.logger.info(f"清理旧记忆: {total_cleaned} 条")
            return total_cleaned
            
        except Exception as e:
            self.logger.error(f"清理旧记忆失败: {e}")
            return 0
    
    async def get_global_memory_stats(self) -> Dict[str, Any]:
        """获取全局记忆统计"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 获取数据库统计
            cursor.execute('SELECT COUNT(*) FROM memories')
            total_memories = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM sessions WHERE status = "active"')
            active_sessions = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(importance) FROM memories')
            avg_importance = cursor.fetchone()[0] or 0.0
            
            cursor.execute('SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type')
            memory_type_dist = dict(cursor.fetchall())
            
            conn.close()
            
            # 获取内存统计
            memory_stats = {
                "total_memories": total_memories,
                "active_sessions": active_sessions,
                "memory_sessions": len(self.session_memories),
                "average_importance": avg_importance,
                "memory_type_distribution": memory_type_dist,
                "timestamp": datetime.now().isoformat()
            }
            
            return memory_stats
            
        except Exception as e:
            self.logger.error(f"获取全局记忆统计失败: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        
        try:
            return {
                "component": "memory_manager",
                "status": "healthy" if self.is_initialized else "stopped",
                "active_sessions": len(self.session_memories),
                "database_path": str(self.db_path),
                "conversation_limit": self.conversation_limit,
                "initialized": self.is_initialized
            }
            
        except Exception as e:
            return {
                "component": "memory_manager",
                "status": "error",
                "error": str(e)
            }