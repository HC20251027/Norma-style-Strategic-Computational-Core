"""
核心记忆管理系统
支持短期记忆、长期记忆、情节记忆和语义记忆的分层存储
"""

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import pickle


class MemoryType(Enum):
    """记忆类型枚举"""
    SHORT_TERM = "short_term"      # 短期记忆（几分钟到几小时）
    LONG_TERM = "long_term"        # 长期记忆（几天到几年）
    EPISODIC = "episodic"          # 情节记忆（具体事件）
    SEMANTIC = "semantic"          # 语义记忆（概念知识）
    PROCEDURAL = "procedural"      # 程序记忆（技能和习惯）


class MemoryImportance(Enum):
    """记忆重要性级别"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MemoryItem:
    """记忆项数据结构"""
    id: str
    content: str
    memory_type: MemoryType
    importance: MemoryImportance
    created_at: datetime
    last_accessed: datetime
    access_count: int
    tags: List[str]
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    associations: List[str] = None  # 关联的记忆ID
    
    def __post_init__(self):
        if self.associations is None:
            self.associations = []
        if not isinstance(self.created_at, datetime):
            self.created_at = datetime.fromisoformat(self.created_at)
        if not isinstance(self.last_accessed, datetime):
            self.last_accessed = datetime.fromisoformat(self.last_accessed)


class MemoryManager:
    """记忆管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get('db_path', 'memory.db')
        self.max_short_term_memories = config.get('max_short_term_memories', 100)
        self.long_term_threshold = config.get('long_term_threshold', 24)  # 小时
        self.importance_threshold = config.get('importance_threshold', MemoryImportance.MEDIUM)
        
        # 记忆缓存
        self.short_term_cache: Dict[str, MemoryItem] = {}
        self.recency_weights = {}
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化SQLite数据库"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # 创建表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                importance INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                tags TEXT,
                metadata TEXT,
                embedding BLOB,
                associations TEXT
            )
        ''')
        
        # 创建索引
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_last_accessed ON memories(last_accessed)')
        
        self.conn.commit()
    
    def _generate_memory_id(self, content: str) -> str:
        """生成记忆ID"""
        return hashlib.md5(f"{content}{time.time()}".encode()).hexdigest()
    
    async def initialize(self):
        """初始化记忆管理器"""
        # 从数据库加载长期记忆
        await self._load_long_term_memories()
        print("记忆管理器初始化完成")
    
    async def _load_long_term_memories(self):
        """从数据库加载长期记忆"""
        cursor = self.conn.execute('''
            SELECT * FROM memories 
            WHERE memory_type IN ('long_term', 'episodic', 'semantic', 'procedural')
            ORDER BY last_accessed DESC
            LIMIT 1000
        ''')
        
        for row in cursor.fetchall():
            memory = self._row_to_memory_item(row)
            # 预加载到缓存
            self.recency_weights[memory.id] = 1.0
    
    def _row_to_memory_item(self, row) -> MemoryItem:
        """将数据库行转换为MemoryItem"""
        return MemoryItem(
            id=row['id'],
            content=row['content'],
            memory_type=MemoryType(row['memory_type']),
            importance=MemoryImportance(row['importance']),
            created_at=datetime.fromisoformat(row['created_at']),
            last_accessed=datetime.fromisoformat(row['last_accessed']),
            access_count=row['access_count'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            embedding=pickle.loads(row['embedding']) if row['embedding'] else None,
            associations=json.loads(row['associations']) if row['associations'] else []
        )
    
    def _memory_item_to_row(self, memory: MemoryItem) -> Dict:
        """将MemoryItem转换为数据库行"""
        return {
            'id': memory.id,
            'content': memory.content,
            'memory_type': memory.memory_type.value,
            'importance': memory.importance.value,
            'created_at': memory.created_at.isoformat(),
            'last_accessed': memory.last_accessed.isoformat(),
            'access_count': memory.access_count,
            'tags': json.dumps(memory.tags),
            'metadata': json.dumps(memory.metadata),
            'embedding': pickle.dumps(memory.embedding) if memory.embedding else None,
            'associations': json.dumps(memory.associations)
        }
    
    async def store_short_term_memory(
        self, 
        content: str, 
        context: Optional[Dict[str, Any]] = None,
        importance: Optional[MemoryImportance] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """存储短期记忆"""
        
        memory_id = self._generate_memory_id(content)
        
        memory = MemoryItem(
            id=memory_id,
            content=content,
            memory_type=MemoryType.SHORT_TERM,
            importance=importance or MemoryImportance.MEDIUM,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            tags=tags or [],
            metadata=context or {},
            associations=[]
        )
        
        # 存储到缓存
        self.short_term_cache[memory_id] = memory
        self.recency_weights[memory_id] = 1.0
        
        # 检查是否需要转换为长期记忆
        await self._check_memory_consolidation(memory)
        
        return memory_id
    
    async def store_long_term_memory(
        self,
        content: str,
        memory_type: MemoryType,
        importance: MemoryImportance,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> str:
        """存储长期记忆"""
        
        memory_id = self._generate_memory_id(content)
        
        memory = MemoryItem(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            tags=tags or [],
            metadata=metadata or {},
            embedding=embedding,
            associations=[]
        )
        
        # 存储到数据库
        row_data = self._memory_item_to_row(memory)
        self.conn.execute('''
            INSERT INTO memories (
                id, content, memory_type, importance, created_at, 
                last_accessed, access_count, tags, metadata, embedding, associations
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row_data['id'], row_data['content'], row_data['memory_type'],
            row_data['importance'], row_data['created_at'], row_data['last_accessed'],
            row_data['access_count'], row_data['tags'], row_data['metadata'],
            row_data['embedding'], row_data['associations']
        ))
        self.conn.commit()
        
        # 更新缓存权重
        self.recency_weights[memory_id] = 1.0
        
        return memory_id
    
    async def _check_memory_consolidation(self, memory: MemoryItem):
        """检查记忆是否需要巩固到长期记忆"""
        
        # 计算记忆年龄（小时）
        age_hours = (datetime.now() - memory.created_at).total_seconds() / 3600
        
        # 检查是否满足长期记忆条件
        should_consolidate = (
            age_hours >= self.long_term_threshold or
            memory.importance.value >= self.importance_threshold.value or
            memory.access_count >= 5
        )
        
        if should_consolidate:
            # 转换为长期记忆
            memory.memory_type = MemoryType.LONG_TERM
            
            # 存储到数据库
            row_data = self._memory_item_to_row(memory)
            self.conn.execute('''
                INSERT OR REPLACE INTO memories (
                    id, content, memory_type, importance, created_at, 
                    last_accessed, access_count, tags, metadata, embedding, associations
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row_data['id'], row_data['content'], row_data['memory_type'],
                row_data['importance'], row_data['created_at'], row_data['last_accessed'],
                row_data['access_count'], row_data['tags'], row_data['metadata'],
                row_data['embedding'], row_data['associations']
            ))
            self.conn.commit()
            
            # 从短期缓存移除
            if memory.id in self.short_term_cache:
                del self.short_term_cache[memory.id]
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """检索记忆"""
        
        # 首先检查短期缓存
        if memory_id in self.short_term_cache:
            memory = self.short_term_cache[memory_id]
            memory.last_accessed = datetime.now()
            memory.access_count += 1
            self.recency_weights[memory_id] = 1.0
            return memory
        
        # 从数据库检索
        cursor = self.conn.execute('SELECT * FROM memories WHERE id = ?', (memory_id,))
        row = cursor.fetchone()
        
        if row:
            memory = self._row_to_memory_item(row)
            memory.last_accessed = datetime.now()
            memory.access_count += 1
            
            # 更新数据库
            self.conn.execute('''
                UPDATE memories 
                SET last_accessed = ?, access_count = ?
                WHERE id = ?
            ''', (memory.last_accessed.isoformat(), memory.access_count, memory_id))
            self.conn.commit()
            
            # 更新缓存权重
            self.recency_weights[memory_id] = 1.0
            
            return memory
        
        return None
    
    async def search_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: Optional[MemoryImportance] = None
    ) -> List[MemoryItem]:
        """搜索记忆"""
        
        # 构建SQL查询
        sql = "SELECT * FROM memories WHERE 1=1"
        params = []
        
        if memory_types:
            placeholders = ','.join(['?' for _ in memory_types])
            sql += f" AND memory_type IN ({placeholders})"
            params.extend([mt.value for mt in memory_types])
        
        if tags:
            for tag in tags:
                sql += " AND tags LIKE ?"
                params.append(f'%"{tag}"%')
        
        if min_importance:
            sql += " AND importance >= ?"
            params.append(min_importance.value)
        
        sql += " ORDER BY last_accessed DESC, importance DESC LIMIT ?"
        params.append(limit)
        
        cursor = self.conn.execute(sql, params)
        memories = []
        
        for row in cursor.fetchall():
            memory = self._row_to_memory_item(row)
            memories.append(memory)
        
        return memories
    
    async def get_associated_memories(self, memory_id: str) -> List[MemoryItem]:
        """获取关联记忆"""
        memory = await self.retrieve_memory(memory_id)
        if not memory:
            return []
        
        associated_memories = []
        for assoc_id in memory.associations:
            assoc_memory = await self.retrieve_memory(assoc_id)
            if assoc_memory:
                associated_memories.append(assoc_memory)
        
        return associated_memories
    
    async def associate_memories(self, memory_id1: str, memory_id2: str):
        """建立记忆关联"""
        memory1 = await self.retrieve_memory(memory_id1)
        memory2 = await self.retrieve_memory(memory_id2)
        
        if memory1 and memory2:
            if memory_id2 not in memory1.associations:
                memory1.associations.append(memory_id2)
            if memory_id1 not in memory2.associations:
                memory2.associations.append(memory_id1)
            
            # 更新数据库
            for memory in [memory1, memory2]:
                row_data = self._memory_item_to_row(memory)
                self.conn.execute('''
                    UPDATE memories 
                    SET associations = ?
                    WHERE id = ?
                ''', (row_data['associations'], memory.id))
            self.conn.commit()
    
    async def forget_memory(self, memory_id: str):
        """忘记记忆"""
        # 从短期缓存移除
        if memory_id in self.short_term_cache:
            del self.short_term_cache[memory_id]
        
        # 从数据库删除
        self.conn.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
        self.conn.commit()
        
        # 移除权重
        if memory_id in self.recency_weights:
            del self.recency_weights[memory_id]
    
    async def cleanup_old_memories(self, days: int = 30):
        """清理旧记忆"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # 删除旧的低重要性短期记忆
        self.conn.execute('''
            DELETE FROM memories 
            WHERE memory_type = 'short_term' 
            AND created_at < ? 
            AND importance < ?
        ''', (cutoff_date.isoformat(), MemoryImportance.MEDIUM.value))
        
        # 删除旧的长期记忆（如果重要性很低）
        self.conn.execute('''
            DELETE FROM memories 
            WHERE memory_type IN ('long_term', 'episodic', 'semantic', 'procedural')
            AND created_at < ? 
            AND importance = ?
            AND access_count = 1
        ''', (cutoff_date.isoformat(), MemoryImportance.LOW.value))
        
        self.conn.commit()
        
        # 清理缓存
        self.short_term_cache = {
            k: v for k, v in self.short_term_cache.items()
            if v.created_at > cutoff_date
        }
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        cursor = self.conn.execute('''
            SELECT memory_type, COUNT(*) as count,
                   AVG(access_count) as avg_access_count
            FROM memories 
            GROUP BY memory_type
        ''')
        
        stats = {}
        for row in cursor.fetchall():
            stats[row['memory_type']] = {
                'count': row['count'],
                'avg_access_count': row['avg_access_count']
            }
        
        stats['short_term_cache_size'] = len(self.short_term_cache)
        stats['total_memories'] = sum(s['count'] for s in stats.values())
        
        return stats
    
    async def close(self):
        """关闭记忆管理器"""
        self.conn.close()
        print("记忆管理器已关闭")