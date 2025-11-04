"""
知识库管理器
协调记忆管理、知识图谱、向量数据库等组件的统一管理
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib


class KnowledgeType(Enum):
    """知识类型"""
    FACTUAL = "factual"              # 事实性知识
    PROCEDURAL = "procedural"        # 程序性知识
    CONCEPTUAL = "conceptual"        # 概念性知识
    CONTEXTUAL = "contextual"        # 上下文知识
    TEMPORAL = "temporal"            # 时间性知识
    CAUSAL = "causal"                # 因果性知识


class UpdateStrategy(Enum):
    """更新策略"""
    APPEND = "append"                # 追加
    MERGE = "merge"                  # 合并
    REPLACE = "replace"              # 替换
    INCREMENTAL = "incremental"      # 增量更新


@dataclass
class KnowledgeItem:
    """知识项"""
    id: str
    content: str
    knowledge_type: KnowledgeType
    confidence: float
    source: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    tags: List[str]
    vector_embedding: Optional[List[float]] = None
    entity_ids: List[str] = None     # 关联的实体ID
    relation_ids: List[str] = None   # 关联的关系ID
    
    def __post_init__(self):
        if not isinstance(self.created_at, datetime):
            self.created_at = datetime.fromisoformat(self.created_at)
        if not isinstance(self.updated_at, datetime):
            self.updated_at = datetime.fromisoformat(self.updated_at)
        if self.entity_ids is None:
            self.entity_ids = []
        if self.relation_ids is None:
            self.relation_ids = []


@dataclass
class UpdateResult:
    """更新结果"""
    success: bool
    items_updated: List[str]
    items_created: List[str]
    entities_created: List[str]
    relations_created: List[str]
    vectors_created: List[str]
    errors: List[str]
    processing_time: float


class KnowledgeBaseManager:
    """知识库管理器"""
    
    def __init__(
        self,
        memory_manager,
        knowledge_graph,
        vector_database,
        config: Dict[str, Any]
    ):
        self.memory_manager = memory_manager
        self.knowledge_graph = knowledge_graph
        self.vector_db = vector_database
        
        self.config = config
        
        # 知识库配置
        self.knowledge_collections = {
            'facts': 'factual_knowledge',
            'procedures': 'procedural_knowledge',
            'concepts': 'conceptual_knowledge',
            'contexts': 'contextual_knowledge',
            'temporal': 'temporal_knowledge',
            'causal': 'causal_knowledge'
        }
        
        # 知识缓存
        self.knowledge_cache: Dict[str, KnowledgeItem] = {}
        self.update_history: List[Dict[str, Any]] = []
        
        # 质量控制
        self.confidence_threshold = config.get('confidence_threshold', 0.6)
        self.max_knowledge_items = config.get('max_knowledge_items', 100000)
        
        # 统计信息
        self.knowledge_stats = {
            'total_items': 0,
            'items_by_type': {},
            'last_update': None,
            'update_frequency': 0.0
        }
    
    async def initialize(self):
        """初始化知识库管理器"""
        # 创建必要的集合
        for collection_name in self.knowledge_collections.values():
            if collection_name not in await self.vector_db.list_collections():
                await self.vector_db.create_collection(collection_name, 384)
        
        # 加载现有知识
        await self._load_existing_knowledge()
        
        print("知识库管理器初始化完成")
    
    async def _load_existing_knowledge(self):
        """加载现有知识"""
        # 从记忆系统加载知识相关记忆
        knowledge_memories = await self.memory_manager.search_memories(
            query="",
            memory_types=['episodic', 'semantic'],
            limit=1000
        )
        
        for memory in knowledge_memories:
            if 'knowledge' in memory.metadata:
                knowledge_item = KnowledgeItem(
                    id=memory.id,
                    content=memory.content,
                    knowledge_type=KnowledgeType(memory.metadata.get('knowledge_type', 'factual')),
                    confidence=memory.importance.value / 4.0,
                    source='memory_import',
                    created_at=memory.created_at,
                    updated_at=memory.last_accessed,
                    metadata=memory.metadata,
                    tags=memory.tags
                )
                self.knowledge_cache[knowledge_item.id] = knowledge_item
        
        self.knowledge_stats['total_items'] = len(self.knowledge_cache)
    
    async def add_knowledge(
        self,
        content: str,
        knowledge_type: KnowledgeType,
        confidence: float,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        vector_embedding: Optional[List[float]] = None,
        update_strategy: UpdateStrategy = UpdateStrategy.APPEND
    ) -> str:
        """添加知识"""
        
        knowledge_id = self._generate_knowledge_id(content)
        
        knowledge_item = KnowledgeItem(
            id=knowledge_id,
            content=content,
            knowledge_type=knowledge_type,
            confidence=confidence,
            source=source,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata=metadata or {},
            tags=tags or [],
            vector_embedding=vector_embedding
        )
        
        # 根据更新策略处理
        if update_strategy == UpdateStrategy.APPEND:
            await self._append_knowledge(knowledge_item)
        elif update_strategy == UpdateStrategy.MERGE:
            await self._merge_knowledge(knowledge_item)
        elif update_strategy == UpdateStrategy.REPLACE:
            await self._replace_knowledge(knowledge_item)
        elif update_strategy == UpdateStrategy.INCREMENTAL:
            await self._incremental_update_knowledge(knowledge_item)
        
        return knowledge_id
    
    async def _append_knowledge(self, knowledge_item: KnowledgeItem):
        """追加知识"""
        # 添加到缓存
        self.knowledge_cache[knowledge_item.id] = knowledge_item
        
        # 创建向量表示
        if not knowledge_item.vector_embedding:
            knowledge_item.vector_embedding = await self._generate_embedding(knowledge_item.content)
        
        # 存储到向量数据库
        collection_name = self.knowledge_collections[knowledge_item.knowledge_type.value]
        vector_id = await self.vector_db.insert_vector(
            vector=knowledge_item.vector_embedding,
            collection=collection_name,
            metadata={
                'knowledge_id': knowledge_item.id,
                'content': knowledge_item.content,
                'knowledge_type': knowledge_item.knowledge_type.value,
                'confidence': knowledge_item.confidence,
                'source': knowledge_item.source
            },
            tags=knowledge_item.tags
        )
        
        # 提取实体和关系
        await self._extract_entities_and_relations(knowledge_item)
        
        # 更新统计
        self._update_statistics()
    
    async def _merge_knowledge(self, new_item: KnowledgeItem):
        """合并知识"""
        # 查找相似知识
        similar_items = await self._find_similar_knowledge(new_item)
        
        if similar_items:
            # 合并到最相似的项
            best_match = similar_items[0]
            existing_item = self.knowledge_cache[best_match['id']]
            
            # 合并内容
            if new_item.content not in existing_item.content:
                existing_item.content += f"\n\n{new_item.content}"
            
            # 更新置信度（取较高值）
            existing_item.confidence = max(existing_item.confidence, new_item.confidence)
            
            # 合并标签
            for tag in new_item.tags:
                if tag not in existing_item.tags:
                    existing_item.tags.append(tag)
            
            # 合并元数据
            existing_item.metadata.update(new_item.metadata)
            existing_item.updated_at = datetime.now()
            
            # 更新向量
            await self._update_knowledge_vector(existing_item)
            
        else:
            # 没有相似项，直接追加
            await self._append_knowledge(new_item)
    
    async def _replace_knowledge(self, new_item: KnowledgeItem):
        """替换知识"""
        # 查找现有知识
        existing_item = self.knowledge_cache.get(new_item.id)
        
        if existing_item:
            # 删除旧的向量
            await self._delete_knowledge_vector(existing_item)
            
            # 替换内容
            existing_item.content = new_item.content
            existing_item.confidence = new_item.confidence
            existing_item.metadata = new_item.metadata
            existing_item.tags = new_item.tags
            existing_item.updated_at = datetime.now()
            
            # 重新生成向量
            await self._update_knowledge_vector(existing_item)
        else:
            # 不存在，直接追加
            await self._append_knowledge(new_item)
    
    async def _incremental_update_knowledge(self, new_item: KnowledgeItem):
        """增量更新知识"""
        # 查找相关知识
        related_items = await self._find_related_knowledge(new_item)
        
        # 更新相关项
        for related_item in related_items:
            # 检查是否需要更新
            similarity = await self._calculate_knowledge_similarity(new_item, related_item)
            
            if similarity > 0.5:  # 相似度阈值
                # 增量更新
                await self._incrementally_update_item(new_item, related_item)
        
        # 如果没有相关项，追加新项
        if not related_items:
            await self._append_knowledge(new_item)
    
    async def _find_similar_knowledge(self, knowledge_item: KnowledgeItem) -> List[Dict[str, Any]]:
        """查找相似知识"""
        
        collection_name = self.knowledge_collections[knowledge_item.knowledge_type.value]
        
        if not knowledge_item.vector_embedding:
            knowledge_item.vector_embedding = await self._generate_embedding(knowledge_item.content)
        
        similar_results = await self.vector_db.search_similar(
            query_vector=knowledge_item.vector_embedding,
            collection=collection_name,
            top_k=5,
            threshold=0.7
        )
        
        return similar_results
    
    async def _find_related_knowledge(self, knowledge_item: KnowledgeItem) -> List[KnowledgeItem]:
        """查找相关知识"""
        
        related_items = []
        
        # 基于标签查找
        for tag in knowledge_item.tags:
            for item in self.knowledge_cache.values():
                if tag in item.tags and item.id != knowledge_item.id:
                    related_items.append(item)
        
        # 基于内容相似性查找
        for item in list(self.knowledge_cache.values())[:10]:  # 限制搜索范围
            if item.id != knowledge_item.id:
                similarity = await self._calculate_knowledge_similarity(knowledge_item, item)
                if similarity > 0.3:
                    related_items.append(item)
        
        return related_items
    
    async def _calculate_knowledge_similarity(self, item1: KnowledgeItem, item2: KnowledgeItem) -> float:
        """计算知识项相似度"""
        
        # 基于向量相似度
        if item1.vector_embedding and item2.vector_embedding:
            vector_sim = await self.vector_db.compute_vector_similarity(
                item1.vector_embedding,
                item2.vector_embedding,
                metric='cosine'
            )
        else:
            vector_sim = 0.0
        
        # 基于标签相似度
        common_tags = set(item1.tags).intersection(set(item2.tags))
        tag_sim = len(common_tags) / max(len(item1.tags), len(item2.tags), 1)
        
        # 基于类型相似度
        type_sim = 1.0 if item1.knowledge_type == item2.knowledge_type else 0.5
        
        # 综合相似度
        similarity = (vector_sim * 0.5) + (tag_sim * 0.3) + (type_sim * 0.2)
        
        return similarity
    
    async def _extract_entities_and_relations(self, knowledge_item: KnowledgeItem):
        """从知识项中提取实体和关系"""
        
        # 简单的实体提取（实际应用中可以使用NLP工具）
        words = knowledge_item.content.split()
        
        # 提取可能的实体
        for word in words:
            if len(word) > 2 and word[0].isupper():
                # 假设首字母大写的词是实体
                try:
                    entity_id = await self.knowledge_graph.add_entity(
                        name=word,
                        entity_type='concept',
                        properties={'source_knowledge': knowledge_item.id}
                    )
                    knowledge_item.entity_ids.append(entity_id)
                except:
                    pass  # 实体可能已存在
        
        # 建立实体间关系
        for i, entity_id1 in enumerate(knowledge_item.entity_ids):
            for entity_id2 in knowledge_item.entity_ids[i+1:]:
                try:
                    relation_id = await self.knowledge_graph.add_relation(
                        source_entity_id=entity_id1,
                        target_entity_id=entity_id2,
                        relation_type='related_to',
                        properties={'source_knowledge': knowledge_item.id}
                    )
                    knowledge_item.relation_ids.append(relation_id)
                except:
                    pass  # 关系可能已存在
    
    async def _generate_embedding(self, content: str) -> List[float]:
        """生成内容嵌入向量"""
        # 简化实现：基于词汇的简单向量表示
        words = content.lower().split()
        vector = np.zeros(384)
        
        for word in words:
            hash_value = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vector[hash_value % 384] += 1.0
        
        # 归一化
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
        
        return vector.tolist()
    
    async def _update_knowledge_vector(self, knowledge_item: KnowledgeItem):
        """更新知识向量"""
        
        # 重新生成向量
        knowledge_item.vector_embedding = await self._generate_embedding(knowledge_item.content)
        
        # 更新向量数据库
        collection_name = self.knowledge_collections[knowledge_item.knowledge_type.value]
        
        # 这里需要实现向量更新逻辑
        # 由于VectorDatabase类没有直接更新方法，这里简化处理
        pass
    
    async def _delete_knowledge_vector(self, knowledge_item: KnowledgeItem):
        """删除知识向量"""
        # 简化实现
        pass
    
    async def _incrementally_update_item(self, new_item: KnowledgeItem, existing_item: KnowledgeItem):
        """增量更新知识项"""
        
        # 添加新的信息片段
        if new_item.content not in existing_item.content:
            existing_item.content += f"\n\n{new_item.content}"
        
        # 更新置信度
        existing_item.confidence = max(existing_item.confidence, new_item.confidence)
        
        # 添加新标签
        for tag in new_item.tags:
            if tag not in existing_item.tags:
                existing_item.tags.append(tag)
        
        # 更新元数据
        existing_item.metadata.update(new_item.metadata)
        existing_item.updated_at = datetime.now()
    
    async def update_from_interaction(
        self,
        user_input: str,
        memory_id: str,
        reasoning_result: Optional[Dict[str, Any]] = None
    ) -> UpdateResult:
        """从交互中更新知识"""
        
        start_time = datetime.now()
        items_created = []
        items_updated = []
        entities_created = []
        relations_created = []
        vectors_created = []
        errors = []
        
        try:
            # 从推理结果中提取新知识
            if reasoning_result and reasoning_result.get('conclusions'):
                for conclusion in reasoning_result['conclusions']:
                    knowledge_content = conclusion.get('conclusion', '')
                    if knowledge_content and len(knowledge_content) > 10:
                        knowledge_id = await self.add_knowledge(
                            content=knowledge_content,
                            knowledge_type=KnowledgeType.CONCEPTUAL,
                            confidence=conclusion.get('confidence', 0.5),
                            source='reasoning',
                            metadata={
                                'reasoning_id': reasoning_result.get('reasoning_id'),
                                'conclusion_type': conclusion.get('type'),
                                'source_memory': memory_id
                            },
                            tags=['reasoning', 'inferred']
                        )
                        items_created.append(knowledge_id)
            
            # 从用户输入中提取知识
            if user_input and len(user_input) > 20:
                knowledge_id = await self.add_knowledge(
                    content=user_input,
                    knowledge_type=KnowledgeType.CONTEXTUAL,
                    confidence=0.7,
                    source='user_interaction',
                    metadata={
                        'memory_id': memory_id,
                        'interaction_timestamp': datetime.now().isoformat()
                    },
                    tags=['user_input', 'contextual']
                )
                items_created.append(knowledge_id)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 更新历史记录
            self.update_history.append({
                'timestamp': datetime.now(),
                'source': 'interaction',
                'items_created': len(items_created),
                'items_updated': len(items_updated),
                'processing_time': processing_time
            })
            
            return UpdateResult(
                success=True,
                items_updated=items_updated,
                items_created=items_created,
                entities_created=entities_created,
                relations_created=relations_created,
                vectors_created=vectors_created,
                errors=errors,
                processing_time=processing_time
            )
            
        except Exception as e:
            errors.append(str(e))
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return UpdateResult(
                success=False,
                items_updated=items_updated,
                items_created=items_created,
                entities_created=entities_created,
                relations_created=relations_created,
                vectors_created=vectors_created,
                errors=errors,
                processing_time=processing_time
            )
    
    async def search_knowledge(
        self,
        query: str,
        knowledge_types: Optional[List[KnowledgeType]] = None,
        min_confidence: float = 0.0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """搜索知识"""
        
        results = []
        
        # 确定搜索集合
        collections_to_search = []
        if knowledge_types:
            for kt in knowledge_types:
                if kt.value in self.knowledge_collections:
                    collections_to_search.append(self.knowledge_collections[kt.value])
        else:
            collections_to_search = list(self.knowledge_collections.values())
        
        # 在每个集合中搜索
        for collection in collections_to_search:
            collection_results = await self._search_in_collection(
                query, collection, min_confidence, limit // len(collections_to_search)
            )
            results.extend(collection_results)
        
        # 按置信度排序
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results[:limit]
    
    async def _search_in_collection(
        self,
        query: str,
        collection: str,
        min_confidence: float,
        limit: int
    ) -> List[Dict[str, Any]]:
        """在特定集合中搜索"""
        
        # 生成查询向量
        query_embedding = await self._generate_embedding(query)
        
        # 向量搜索
        vector_results = await self.vector_db.search_similar(
            query_vector=query_embedding,
            collection=collection,
            top_k=limit,
            threshold=min_confidence
        )
        
        # 构建结果
        search_results = []
        for result in vector_results:
            knowledge_id = result['metadata'].get('knowledge_id')
            if knowledge_id and knowledge_id in self.knowledge_cache:
                knowledge_item = self.knowledge_cache[knowledge_id]
                
                search_results.append({
                    'id': knowledge_id,
                    'content': knowledge_item.content,
                    'knowledge_type': knowledge_item.knowledge_type.value,
                    'confidence': knowledge_item.confidence,
                    'source': knowledge_item.source,
                    'tags': knowledge_item.tags,
                    'metadata': knowledge_item.metadata,
                    'similarity_score': result['similarity_score'],
                    'created_at': knowledge_item.created_at.isoformat(),
                    'updated_at': knowledge_item.updated_at.isoformat()
                })
        
        return search_results
    
    async def get_knowledge_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        
        # 统计各类知识数量
        type_counts = {}
        for item in self.knowledge_cache.values():
            kt = item.knowledge_type.value
            type_counts[kt] = type_counts.get(kt, 0) + 1
        
        # 计算平均置信度
        confidences = [item.confidence for item in self.knowledge_cache.values()]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # 计算更新频率
        if self.update_history:
            recent_updates = [
                update for update in self.update_history
                if update['timestamp'] > datetime.now() - timedelta(days=7)
            ]
            update_frequency = len(recent_updates) / 7.0  # 每天平均更新次数
        else:
            update_frequency = 0.0
        
        # 知识来源分布
        source_counts = {}
        for item in self.knowledge_cache.values():
            source = item.source
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            'total_items': len(self.knowledge_cache),
            'items_by_type': type_counts,
            'average_confidence': avg_confidence,
            'update_frequency_per_day': update_frequency,
            'source_distribution': source_counts,
            'last_update': self.update_history[-1]['timestamp'] if self.update_history else None,
            'total_updates': len(self.update_history),
            'cache_size': len(self.knowledge_cache)
        }
    
    async def export_knowledge(self, format: str = 'json') -> str:
        """导出知识库"""
        
        knowledge_data = []
        
        for item in self.knowledge_cache.values():
            item_data = asdict(item)
            # 转换datetime为字符串
            item_data['created_at'] = item.created_at.isoformat()
            item_data['updated_at'] = item.updated_at.isoformat()
            knowledge_data.append(item_data)
        
        if format == 'json':
            return json.dumps(knowledge_data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    async def import_knowledge(self, knowledge_data: str, format: str = 'json') -> UpdateResult:
        """导入知识库"""
        
        start_time = datetime.now()
        items_created = []
        errors = []
        
        try:
            if format == 'json':
                data = json.loads(knowledge_data)
                
                for item_data in data:
                    # 重建KnowledgeItem
                    item_data['created_at'] = datetime.fromisoformat(item_data['created_at'])
                    item_data['updated_at'] = datetime.fromisoformat(item_data['updated_at'])
                    
                    knowledge_item = KnowledgeItem(**item_data)
                    
                    # 添加到知识库
                    await self._append_knowledge(knowledge_item)
                    items_created.append(knowledge_item.id)
                
            else:
                raise ValueError(f"不支持的导入格式: {format}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return UpdateResult(
                success=True,
                items_updated=[],
                items_created=items_created,
                entities_created=[],
                relations_created=[],
                vectors_created=[],
                errors=errors,
                processing_time=processing_time
            )
            
        except Exception as e:
            errors.append(str(e))
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return UpdateResult(
                success=False,
                items_updated=[],
                items_created=[],
                entities_created=[],
                relations_created=[],
                vectors_created=[],
                errors=errors,
                processing_time=processing_time
            )
    
    def _generate_knowledge_id(self, content: str) -> str:
        """生成知识ID"""
        return hashlib.md5(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()
    
    def _update_statistics(self):
        """更新统计信息"""
        self.knowledge_stats['total_items'] = len(self.knowledge_cache)
        self.knowledge_stats['last_update'] = datetime.now()
        
        # 按类型统计
        type_counts = {}
        for item in self.knowledge_cache.values():
            kt = item.knowledge_type.value
            type_counts[kt] = type_counts.get(kt, 0) + 1
        self.knowledge_stats['items_by_type'] = type_counts
        
        # 计算更新频率
        if len(self.update_history) > 1:
            time_span = (self.update_history[-1]['timestamp'] - self.update_history[0]['timestamp']).total_seconds()
            if time_span > 0:
                self.knowledge_stats['update_frequency'] = len(self.update_history) / (time_span / 86400)  # 每天
    
    async def cleanup_knowledge(self, older_than_days: int = 30, min_confidence: float = 0.3):
        """清理知识"""
        
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        items_to_remove = []
        
        for item_id, item in self.knowledge_cache.items():
            if (item.updated_at < cutoff_date and 
                item.confidence < min_confidence and
                item.source != 'user_interaction'):  # 保留用户交互内容
                items_to_remove.append(item_id)
        
        # 移除低质量知识
        for item_id in items_to_remove:
            del self.knowledge_cache[item_id]
        
        print(f"已清理 {len(items_to_remove)} 个低质量知识项")
        
        return len(items_to_remove)