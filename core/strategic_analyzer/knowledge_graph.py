"""
知识图谱系统
支持实体、关系和属性的图结构存储与推理
"""

import asyncio
import json
import sqlite3
import networkx as nx
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import pickle


class EntityType(Enum):
    """实体类型"""
    PERSON = "person"
    CONCEPT = "concept"
    OBJECT = "object"
    EVENT = "event"
    LOCATION = "location"
    TIME = "time"
    ORGANIZATION = "organization"
    CUSTOM = "custom"


class RelationType(Enum):
    """关系类型"""
    IS_A = "is_a"           # 分类关系
    PART_OF = "part_of"     # 部分关系
    RELATED_TO = "related_to"  # 相关关系
    CAUSES = "causes"       # 因果关系
    SIMILAR_TO = "similar_to"  # 相似关系
    OPPOSITE_OF = "opposite_of"  # 相反关系
    LOCATED_IN = "located_in"  # 位置关系
    OCCURS_AT = "occurs_at"  # 时间关系
    INTERACTS_WITH = "interacts_with"  # 交互关系
    CUSTOM = "custom"       # 自定义关系


@dataclass
class Entity:
    """实体类"""
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any]
    aliases: List[str]
    confidence: float  # 置信度 0-1
    created_at: datetime
    updated_at: datetime
    
    def __post_init__(self):
        if not isinstance(self.created_at, datetime):
            self.created_at = datetime.fromisoformat(self.created_at)
        if not isinstance(self.updated_at, datetime):
            self.updated_at = datetime.fromisoformat(self.updated_at)


@dataclass
class Relation:
    """关系类"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    properties: Dict[str, Any]
    confidence: float
    created_at: datetime
    weight: float = 1.0  # 关系权重
    
    def __post_init__(self):
        if not isinstance(self.created_at, datetime):
            self.created_at = datetime.fromisoformat(self.created_at)


class KnowledgeGraph:
    """知识图谱类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get('db_path', 'knowledge_graph.db')
        self.max_entities = config.get('max_entities', 10000)
        self.max_relations = config.get('max_relations', 50000)
        
        # NetworkX图对象
        self.graph = nx.MultiDiGraph()
        
        # 实体和关系缓存
        self.entity_cache: Dict[str, Entity] = {}
        self.relation_cache: Dict[str, Relation] = {}
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """初始化SQLite数据库"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # 创建实体表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                properties TEXT,
                aliases TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # 创建关系表
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT,
                confidence REAL DEFAULT 1.0,
                weight REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_entity_id) REFERENCES entities(id),
                FOREIGN KEY (target_entity_id) REFERENCES entities(id)
            )
        ''')
        
        # 创建索引
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_relation_type ON relations(relation_type)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_source_entity ON relations(source_entity_id)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_target_entity ON relations(target_entity_id)')
        
        self.conn.commit()
    
    def _generate_id(self, prefix: str, content: str) -> str:
        """生成唯一ID"""
        return hashlib.md5(f"{prefix}:{content}:{datetime.now().isoformat()}".encode()).hexdigest()
    
    async def initialize(self):
        """初始化知识图谱"""
        await self._load_from_database()
        print("知识图谱初始化完成")
    
    async def _load_from_database(self):
        """从数据库加载实体和关系"""
        # 加载实体
        cursor = self.conn.execute('SELECT * FROM entities')
        for row in cursor.fetchall():
            entity = self._row_to_entity(row)
            self.entity_cache[entity.id] = entity
            self.graph.add_node(entity.id, **asdict(entity))
        
        # 加载关系
        cursor = self.conn.execute('SELECT * FROM relations')
        for row in cursor.fetchall():
            relation = self._row_to_relation(row)
            self.relation_cache[relation.id] = relation
            self.graph.add_edge(
                relation.source_entity_id,
                relation.target_entity_id,
                key=relation.id,
                **asdict(relation)
            )
    
    def _row_to_entity(self, row) -> Entity:
        """将数据库行转换为Entity"""
        return Entity(
            id=row['id'],
            name=row['name'],
            entity_type=EntityType(row['entity_type']),
            properties=json.loads(row['properties']) if row['properties'] else {},
            aliases=json.loads(row['aliases']) if row['aliases'] else [],
            confidence=row['confidence'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at'])
        )
    
    def _row_to_relation(self, row) -> Relation:
        """将数据库行转换为Relation"""
        return Relation(
            id=row['id'],
            source_entity_id=row['source_entity_id'],
            target_entity_id=row['target_entity_id'],
            relation_type=RelationType(row['relation_type']),
            properties=json.loads(row['properties']) if row['properties'] else {},
            confidence=row['confidence'],
            created_at=datetime.fromisoformat(row['created_at']),
            weight=row['weight']
        )
    
    def _entity_to_row(self, entity: Entity) -> Dict:
        """将Entity转换为数据库行"""
        return {
            'id': entity.id,
            'name': entity.name,
            'entity_type': entity.entity_type.value,
            'properties': json.dumps(entity.properties),
            'aliases': json.dumps(entity.aliases),
            'confidence': entity.confidence,
            'created_at': entity.created_at.isoformat(),
            'updated_at': entity.updated_at.isoformat()
        }
    
    def _relation_to_row(self, relation: Relation) -> Dict:
        """将Relation转换为数据库行"""
        return {
            'id': relation.id,
            'source_entity_id': relation.source_entity_id,
            'target_entity_id': relation.target_entity_id,
            'relation_type': relation.relation_type.value,
            'properties': json.dumps(relation.properties),
            'confidence': relation.confidence,
            'weight': relation.weight,
            'created_at': relation.created_at.isoformat()
        }
    
    async def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        properties: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
        confidence: float = 1.0
    ) -> str:
        """添加实体"""
        
        entity_id = self._generate_id("entity", name)
        
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            properties=properties or {},
            aliases=aliases or [],
            confidence=confidence,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # 存储到缓存
        self.entity_cache[entity_id] = entity
        
        # 添加到图
        self.graph.add_node(entity_id, **asdict(entity))
        
        # 存储到数据库
        row_data = self._entity_to_row(entity)
        self.conn.execute('''
            INSERT INTO entities (
                id, name, entity_type, properties, aliases, 
                confidence, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row_data['id'], row_data['name'], row_data['entity_type'],
            row_data['properties'], row_data['aliases'], row_data['confidence'],
            row_data['created_at'], row_data['updated_at']
        ))
        self.conn.commit()
        
        return entity_id
    
    async def add_relation(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: RelationType,
        properties: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        weight: float = 1.0
    ) -> str:
        """添加关系"""
        
        relation_id = self._generate_id("relation", f"{source_entity_id}_{target_entity_id}_{relation_type.value}")
        
        relation = Relation(
            id=relation_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relation_type=relation_type,
            properties=properties or {},
            confidence=confidence,
            created_at=datetime.now(),
            weight=weight
        )
        
        # 存储到缓存
        self.relation_cache[relation_id] = relation
        
        # 添加到图
        self.graph.add_edge(
            source_entity_id,
            target_entity_id,
            key=relation_id,
            **asdict(relation)
        )
        
        # 存储到数据库
        row_data = self._relation_to_row(relation)
        self.conn.execute('''
            INSERT INTO relations (
                id, source_entity_id, target_entity_id, relation_type,
                properties, confidence, weight, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row_data['id'], row_data['source_entity_id'], row_data['target_entity_id'],
            row_data['relation_type'], row_data['properties'], row_data['confidence'],
            row_data['weight'], row_data['created_at']
        ))
        self.conn.commit()
        
        return relation_id
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        return self.entity_cache.get(entity_id)
    
    async def find_entities_by_name(self, name: str) -> List[Entity]:
        """根据名称查找实体"""
        matching_entities = []
        for entity in self.entity_cache.values():
            if name.lower() in entity.name.lower() or any(name.lower() in alias.lower() for alias in entity.aliases):
                matching_entities.append(entity)
        return matching_entities
    
    async def find_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """根据类型查找实体"""
        return [entity for entity in self.entity_cache.values() if entity.entity_type == entity_type]
    
    async def get_relations(
        self,
        entity_id: Optional[str] = None,
        relation_type: Optional[RelationType] = None,
        source_entity_id: Optional[str] = None,
        target_entity_id: Optional[str] = None
    ) -> List[Relation]:
        """获取关系"""
        
        relations = []
        for relation in self.relation_cache.values():
            if entity_id and entity_id not in [relation.source_entity_id, relation.target_entity_id]:
                continue
            if relation_type and relation.relation_type != relation_type:
                continue
            if source_entity_id and relation.source_entity_id != source_entity_id:
                continue
            if target_entity_id and relation.target_entity_id != target_entity_id:
                continue
            relations.append(relation)
        
        return relations
    
    async def find_connections(self, query: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """查找连接关系"""
        # 首先查找匹配的实体
        entities = await self.find_entities_by_name(query)
        if not entities:
            return []
        
        connections = []
        
        for entity in entities:
            # 使用NetworkX查找路径
            try:
                # 查找与该实体相关的所有路径
                for other_entity in self.entity_cache.values():
                    if other_entity.id != entity.id:
                        try:
                            paths = list(nx.all_simple_paths(
                                self.graph,
                                entity.id,
                                other_entity.id,
                                cutoff=max_depth
                            ))
                            
                            for path in paths:
                                connections.append({
                                    'source': entity.name,
                                    'target': other_entity.name,
                                    'path': path,
                                    'path_length': len(path) - 1
                                })
                        except nx.NetworkXNoPath:
                            continue
            except Exception as e:
                print(f"路径查找错误: {e}")
                continue
        
        # 按路径长度排序
        connections.sort(key=lambda x: x['path_length'])
        return connections[:10]  # 返回前10个最短路径
    
    async def get_subgraph(self, entity_ids: List[str], max_depth: int = 2) -> Dict[str, Any]:
        """获取子图"""
        subgraph_entities = set(entity_ids)
        subgraph_relations = []
        
        # 扩展实体集合
        for depth in range(max_depth):
            new_entities = set()
            for relation in self.relation_cache.values():
                if relation.source_entity_id in subgraph_entities:
                    new_entities.add(relation.target_entity_id)
                    subgraph_relations.append(relation)
                elif relation.target_entity_id in subgraph_entities:
                    new_entities.add(relation.source_entity_id)
                    subgraph_relations.append(relation)
            
            subgraph_entities.update(new_entities)
        
        # 构建子图数据
        entities_data = []
        for entity_id in subgraph_entities:
            entity = self.entity_cache.get(entity_id)
            if entity:
                entities_data.append(asdict(entity))
        
        relations_data = []
        for relation in subgraph_relations:
            relations_data.append(asdict(relation))
        
        return {
            'entities': entities_data,
            'relations': relations_data,
            'entity_count': len(entities_data),
            'relation_count': len(relations_data)
        }
    
    async def reason_about_entity(self, entity_id: str) -> Dict[str, Any]:
        """对实体进行推理"""
        entity = self.entity_cache.get(entity_id)
        if not entity:
            return {}
        
        # 获取实体的所有关系
        entity_relations = await self.get_relations(entity_id=entity_id)
        
        # 分类关系
        outgoing_relations = [r for r in entity_relations if r.source_entity_id == entity_id]
        incoming_relations = [r for r in entity_relations if r.target_entity_id == entity_id]
        
        # 分析实体类型
        similar_entities = []
        for relation in outgoing_relations:
            if relation.relation_type == RelationType.SIMILAR_TO:
                target_entity = self.entity_cache.get(relation.target_entity_id)
                if target_entity:
                    similar_entities.append({
                        'entity': asdict(target_entity),
                        'confidence': relation.confidence * relation.weight
                    })
        
        # 分析属性
        inferred_properties = {}
        for relation in outgoing_relations:
            if relation.relation_type == RelationType.IS_A:
                target_entity = self.entity_cache.get(relation.target_entity_id)
                if target_entity:
                    inferred_properties.update(target_entity.properties)
        
        return {
            'entity': asdict(entity),
            'outgoing_relations': [asdict(r) for r in outgoing_relations],
            'incoming_relations': [asdict(r) for r in incoming_relations],
            'similar_entities': similar_entities,
            'inferred_properties': inferred_properties,
            'relation_count': len(entity_relations),
            'centrality_score': self._calculate_centrality(entity_id)
        }
    
    def _calculate_centrality(self, entity_id: str) -> float:
        """计算实体中心性分数"""
        try:
            # 度中心性
            degree_centrality = nx.degree_centrality(self.graph).get(entity_id, 0)
            
            # 介数中心性
            betweenness_centrality = nx.betweenness_centrality(self.graph).get(entity_id, 0)
            
            # 接近中心性
            try:
                closeness_centrality = nx.closeness_centrality(self.graph).get(entity_id, 0)
            except:
                closeness_centrality = 0
            
            # 综合中心性分数
            centrality_score = (degree_centrality + betweenness_centrality + closeness_centrality) / 3
            return centrality_score
        except:
            return 0.0
    
    async def update_entity(
        self,
        entity_id: str,
        properties: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
        confidence: Optional[float] = None
    ):
        """更新实体"""
        entity = self.entity_cache.get(entity_id)
        if not entity:
            return
        
        # 更新属性
        if properties:
            entity.properties.update(properties)
        if aliases:
            entity.aliases = aliases
        if confidence is not None:
            entity.confidence = confidence
        
        entity.updated_at = datetime.now()
        
        # 更新缓存和图
        self.entity_cache[entity_id] = entity
        self.graph.nodes[entity_id].update(asdict(entity))
        
        # 更新数据库
        row_data = self._entity_to_row(entity)
        self.conn.execute('''
            UPDATE entities 
            SET properties = ?, aliases = ?, confidence = ?, updated_at = ?
            WHERE id = ?
        ''', (
            row_data['properties'], row_data['aliases'], 
            row_data['confidence'], row_data['updated_at'], entity_id
        ))
        self.conn.commit()
    
    async def delete_entity(self, entity_id: str):
        """删除实体及其相关关系"""
        # 删除相关关系
        self.conn.execute('DELETE FROM relations WHERE source_entity_id = ? OR target_entity_id = ?', 
                         (entity_id, entity_id))
        
        # 删除实体
        self.conn.execute('DELETE FROM entities WHERE id = ?', (entity_id,))
        self.conn.commit()
        
        # 从缓存和图中移除
        if entity_id in self.entity_cache:
            del self.entity_cache[entity_id]
        if entity_id in self.graph:
            self.graph.remove_node(entity_id)
        
        # 清理关系缓存
        self.relation_cache = {
            rid: rel for rid, rel in self.relation_cache.items()
            if rel.source_entity_id != entity_id and rel.target_entity_id != entity_id
        }
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图统计信息"""
        stats = {
            'entity_count': len(self.entity_cache),
            'relation_count': len(self.relation_cache),
            'entity_types': {},
            'relation_types': {}
        }
        
        # 统计实体类型
        for entity in self.entity_cache.values():
            entity_type = entity.entity_type.value
            stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
        
        # 统计关系类型
        for relation in self.relation_cache.values():
            relation_type = relation.relation_type.value
            stats['relation_types'][relation_type] = stats['relation_types'].get(relation_type, 0) + 1
        
        # 图结构统计
        try:
            stats['is_connected'] = nx.is_weakly_connected(self.graph)
            stats['density'] = nx.density(self.graph)
            stats['average_clustering'] = nx.average_clustering(self.graph.to_undirected())
        except:
            stats['is_connected'] = False
            stats['density'] = 0.0
            stats['average_clustering'] = 0.0
        
        return stats
    
    async def close(self):
        """关闭知识图谱"""
        self.conn.close()
        print("知识图谱已关闭")