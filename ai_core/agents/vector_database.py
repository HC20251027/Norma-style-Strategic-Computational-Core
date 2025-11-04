"""
向量数据库系统
支持高维向量的存储、索引和相似性搜索
"""

import asyncio
import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import pickle
import faiss
from sklearn.metrics.pairwise import cosine_similarity


class VectorIndexType(Enum):
    """向量索引类型"""
    FLAT = "flat"              # 暴力搜索
    IVF = "ivf"                # 倒排文件
    HNSW = "hnsw"              # 分层导航
    PQ = "pq"                  # 产品量化


@dataclass
class VectorItem:
    """向量项"""
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    collection: str
    created_at: datetime
    updated_at: datetime
    tags: List[str]
    
    def __post_init__(self):
        if not isinstance(self.created_at, datetime):
            self.created_at = datetime.fromisoformat(self.created_at)
        if not isinstance(self.updated_at, datetime):
            self.updated_at = datetime.fromisoformat(self.updated_at)
        if isinstance(self.vector, np.ndarray):
            self.vector = self.vector.tolist()


class VectorDatabase:
    """向量数据库"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get('db_path', 'vector_db.db')
        self.dimension = config.get('dimension', 384)  # 默认向量维度
        self.index_type = VectorIndexType(config.get('index_type', 'flat'))
        self.max_vectors_per_collection = config.get('max_vectors_per_collection', 100000)
        
        # 数据库连接
        self.conn = None
        
        # FAISS索引
        self.faiss_indexes: Dict[str, faiss.Index] = {}
        self.collection_stats: Dict[str, Dict[str, Any]] = {}
        
        # 向量缓存
        self.vector_cache: Dict[str, VectorItem] = {}
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化SQLite数据库"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # 创建向量表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                collection TEXT NOT NULL,
                vector BLOB NOT NULL,
                metadata TEXT,
                tags TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(id, collection)
            )
        ''')
        
        # 创建集合表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS collections (
                name TEXT PRIMARY KEY,
                dimension INTEGER NOT NULL,
                index_type TEXT NOT NULL,
                vector_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # 创建索引
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_collection ON vectors(collection)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON vectors(created_at)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_tags ON vectors(tags)')
        
        self.conn.commit()
    
    def _generate_id(self, content: str, collection: str) -> str:
        """生成向量ID"""
        return hashlib.md5(f"{collection}:{content}:{datetime.now().isoformat()}".encode()).hexdigest()
    
    async def initialize(self):
        """初始化向量数据库"""
        await self._load_collections()
        print("向量数据库初始化完成")
    
    async def _load_collections(self):
        """加载所有集合"""
        cursor = self.conn.execute('SELECT * FROM collections')
        
        for row in cursor.fetchall():
            collection_name = row['name']
            dimension = row['dimension']
            index_type = VectorIndexType(row['index_type'])
            
            # 创建FAISS索引
            await self._create_faiss_index(collection_name, dimension, index_type)
            
            # 加载向量数据
            await self._load_vectors_for_collection(collection_name)
            
            # 更新统计信息
            self.collection_stats[collection_name] = {
                'dimension': dimension,
                'index_type': index_type,
                'vector_count': row['vector_count']
            }
    
    async def _create_faiss_index(self, collection_name: str, dimension: int, index_type: VectorIndexType):
        """创建FAISS索引"""
        if index_type == VectorIndexType.FLAT:
            # 暴力搜索索引
            index = faiss.IndexFlatIP(dimension)  # 内积相似度
        elif index_type == VectorIndexType.IVF:
            # 倒排文件索引
            nlist = min(100, max(1, self.collection_stats.get(collection_name, {}).get('vector_count', 0) // 39))
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif index_type == VectorIndexType.HNSW:
            # HNSW索引
            index = faiss.IndexHNSWFlat(dimension, 32)
        elif index_type == VectorIndexType.PQ:
            # 产品量化索引
            nbits = 8
            m = min(64, dimension // 2)
            index = faiss.IndexPQ(dimension, m, nbits)
        else:
            index = faiss.IndexFlatIP(dimension)
        
        self.faiss_indexes[collection_name] = index
    
    async def _load_vectors_for_collection(self, collection_name: str):
        """为集合加载向量数据"""
        cursor = self.conn.execute('''
            SELECT * FROM vectors WHERE collection = ?
        ''', (collection_name,))
        
        vectors = []
        vector_ids = []
        
        for row in cursor.fetchall():
            vector = pickle.loads(row['vector'])
            vectors.append(vector)
            vector_ids.append(row['id'])
            
            # 缓存向量项
            vector_item = VectorItem(
                id=row['id'],
                vector=vector,
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                collection=row['collection'],
                created_at=datetime.fromisoformat(row['created_at']),
                updated_at=datetime.fromisoformat(row['updated_at']),
                tags=json.loads(row['tags']) if row['tags'] else []
            )
            self.vector_cache[row['id']] = vector_item
        
        if vectors:
            # 添加到FAISS索引
            vectors_array = np.array(vectors, dtype=np.float32)
            self.faiss_indexes[collection_name].add(vectors_array)
    
    async def create_collection(
        self,
        collection_name: str,
        dimension: int,
        index_type: VectorIndexType = VectorIndexType.FLAT
    ):
        """创建新集合"""
        
        # 检查集合是否已存在
        cursor = self.conn.execute('SELECT name FROM collections WHERE name = ?', (collection_name,))
        if cursor.fetchone():
            return  # 集合已存在
        
        # 创建集合记录
        now = datetime.now().isoformat()
        self.conn.execute('''
            INSERT INTO collections (name, dimension, index_type, vector_count, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?)
        ''', (collection_name, dimension, index_type.value, now, now))
        self.conn.commit()
        
        # 创建FAISS索引
        await self._create_faiss_index(collection_name, dimension, index_type)
        
        # 更新统计信息
        self.collection_stats[collection_name] = {
            'dimension': dimension,
            'index_type': index_type,
            'vector_count': 0
        }
    
    async def insert_vector(
        self,
        vector: Union[List[float], np.ndarray],
        collection: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        vector_id: Optional[str] = None
    ) -> str:
        """插入向量"""
        
        # 确保向量是numpy数组
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)
        elif not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        
        # 生成ID
        if not vector_id:
            content = str(vector.tolist()) + str(metadata or {})
            vector_id = self._generate_id(content, collection)
        
        # 创建向量项
        vector_item = VectorItem(
            id=vector_id,
            vector=vector.tolist(),
            metadata=metadata or {},
            collection=collection,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=tags or []
        )
        
        # 检查集合是否存在
        if collection not in self.collection_stats:
            await self.create_collection(collection, len(vector))
        
        # 添加到FAISS索引
        if collection in self.faiss_indexes:
            self.faiss_indexes[collection].add(vector.reshape(1, -1))
        
        # 存储到数据库
        self.conn.execute('''
            INSERT OR REPLACE INTO vectors (id, collection, vector, metadata, tags, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            vector_id,
            collection,
            pickle.dumps(vector),
            json.dumps(vector_item.metadata),
            json.dumps(vector_item.tags),
            vector_item.created_at.isoformat(),
            vector_item.updated_at.isoformat()
        ))
        
        # 更新集合统计
        self.conn.execute('''
            UPDATE collections 
            SET vector_count = vector_count + 1, updated_at = ?
            WHERE name = ?
        ''', (datetime.now().isoformat(), collection))
        self.conn.commit()
        
        # 更新缓存和统计
        self.vector_cache[vector_id] = vector_item
        if collection in self.collection_stats:
            self.collection_stats[collection]['vector_count'] += 1
        
        return vector_id
    
    async def search_similar(
        self,
        query_vector: Union[List[float], np.ndarray],
        collection: str,
        top_k: int = 10,
        threshold: float = 0.7,
        filter_tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        
        if collection not in self.faiss_indexes:
            return []
        
        # 确保查询向量是numpy数组
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector, dtype=np.float32)
        elif not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        # 搜索
        index = self.faiss_indexes[collection]
        if index.ntotal == 0:
            return []
        
        # FAISS搜索
        distances, indices = index.search(query_vector.reshape(1, -1), top_k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # 无效索引
                continue
            
            # 计算相似度分数
            similarity_score = float(distance)
            
            # 过滤阈值
            if similarity_score < threshold:
                continue
            
            # 获取向量ID（需要从数据库查询）
            cursor = self.conn.execute('''
                SELECT id FROM vectors WHERE rowid = ? AND collection = ?
            ''', (idx + 1, collection))
            row = cursor.fetchone()
            
            if row:
                vector_id = row['id']
                vector_item = self.vector_cache.get(vector_id)
                
                if vector_item:
                    # 标签过滤
                    if filter_tags and not any(tag in vector_item.tags for tag in filter_tags):
                        continue
                    
                    results.append({
                        'id': vector_id,
                        'similarity_score': similarity_score,
                        'distance': float(distance),
                        'metadata': vector_item.metadata,
                        'tags': vector_item.tags,
                        'rank': i + 1
                    })
        
        return results
    
    async def get_vector(self, vector_id: str) -> Optional[VectorItem]:
        """获取向量"""
        return self.vector_cache.get(vector_id)
    
    async def update_vector(
        self,
        vector_id: str,
        vector: Optional[Union[List[float], np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """更新向量"""
        vector_item = self.vector_cache.get(vector_id)
        if not vector_item:
            return
        
        # 更新向量
        if vector is not None:
            if isinstance(vector, list):
                vector = np.array(vector, dtype=np.float32)
            elif not isinstance(vector, np.ndarray):
                vector = np.array(vector, dtype=np.float32)
            vector_item.vector = vector.tolist()
        
        # 更新元数据
        if metadata:
            vector_item.metadata.update(metadata)
        
        # 更新标签
        if tags:
            vector_item.tags = tags
        
        vector_item.updated_at = datetime.now()
        
        # 更新数据库
        self.conn.execute('''
            UPDATE vectors 
            SET vector = ?, metadata = ?, tags = ?, updated_at = ?
            WHERE id = ?
        ''', (
            pickle.dumps(np.array(vector_item.vector, dtype=np.float32)),
            json.dumps(vector_item.metadata),
            json.dumps(vector_item.tags),
            vector_item.updated_at.isoformat(),
            vector_id
        ))
        self.conn.commit()
        
        # 更新缓存
        self.vector_cache[vector_id] = vector_item
    
    async def delete_vector(self, vector_id: str):
        """删除向量"""
        vector_item = self.vector_cache.get(vector_id)
        if not vector_item:
            return
        
        collection = vector_item.collection
        
        # 从数据库删除
        self.conn.execute('DELETE FROM vectors WHERE id = ?', (vector_id,))
        
        # 更新集合统计
        self.conn.execute('''
            UPDATE collections 
            SET vector_count = vector_count - 1, updated_at = ?
            WHERE name = ?
        ''', (datetime.now().isoformat(), collection))
        self.conn.commit()
        
        # 从缓存删除
        del self.vector_cache[vector_id]
        
        # 更新统计信息
        if collection in self.collection_stats:
            self.collection_stats[collection]['vector_count'] -= 1
    
    async def batch_insert(
        self,
        vectors: List[Union[List[float], np.ndarray]],
        collection: str,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        tags_list: Optional[List[List[str]]] = None
    ) -> List[str]:
        """批量插入向量"""
        
        vector_ids = []
        
        for i, vector in enumerate(vectors):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else None
            tags = tags_list[i] if tags_list and i < len(tags_list) else None
            
            vector_id = await self.insert_vector(
                vector=vector,
                collection=collection,
                metadata=metadata,
                tags=tags
            )
            vector_ids.append(vector_id)
        
        return vector_ids
    
    async def get_collection_stats(self, collection: str) -> Optional[Dict[str, Any]]:
        """获取集合统计信息"""
        return self.collection_stats.get(collection)
    
    async def list_collections(self) -> List[str]:
        """列出所有集合"""
        return list(self.collection_stats.keys())
    
    async def delete_collection(self, collection: str):
        """删除集合"""
        # 删除所有向量
        self.conn.execute('DELETE FROM vectors WHERE collection = ?', (collection,))
        
        # 删除集合记录
        self.conn.execute('DELETE FROM collections WHERE name = ?', (collection,))
        self.conn.commit()
        
        # 清理索引和缓存
        if collection in self.faiss_indexes:
            del self.faiss_indexes[collection]
        if collection in self.collection_stats:
            del self.collection_stats[collection]
        
        # 清理向量缓存
        vector_ids_to_remove = [
            vid for vid, item in self.vector_cache.items() 
            if item.collection == collection
        ]
        for vid in vector_ids_to_remove:
            del self.vector_cache[vid]
    
    async def compute_vector_similarity(
        self,
        vector1: Union[List[float], np.ndarray],
        vector2: Union[List[float], np.ndarray],
        metric: str = 'cosine'
    ) -> float:
        """计算两个向量的相似度"""
        
        # 确保向量是numpy数组
        if isinstance(vector1, list):
            vector1 = np.array(vector1, dtype=np.float32)
        elif not isinstance(vector1, np.ndarray):
            vector1 = np.array(vector1, dtype=np.float32)
        
        if isinstance(vector2, list):
            vector2 = np.array(vector2, dtype=np.float32)
        elif not isinstance(vector2, np.ndarray):
            vector2 = np.array(vector2, dtype=np.float32)
        
        if metric == 'cosine':
            # 余弦相似度
            similarity = cosine_similarity([vector1], [vector2])[0][0]
            return float(similarity)
        elif metric == 'euclidean':
            # 欧几里得距离
            distance = np.linalg.norm(vector1 - vector2)
            return float(1 / (1 + distance))  # 转换为相似度
        else:
            raise ValueError(f"不支持的相似度度量: {metric}")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        stats = {
            'total_vectors': len(self.vector_cache),
            'total_collections': len(self.collection_stats),
            'collections': {}
        }
        
        # 集合统计
        for collection, collection_stats in self.collection_stats.items():
            stats['collections'][collection] = collection_stats
        
        return stats
    
    async def close(self):
        """关闭向量数据库"""
        self.conn.close()
        print("向量数据库已关闭")