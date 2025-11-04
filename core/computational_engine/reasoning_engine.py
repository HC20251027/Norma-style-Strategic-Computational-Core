"""
知识推理引擎
基于知识图谱的逻辑推理、关联发现和知识发现
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import networkx as nx
from collections import defaultdict, deque


@dataclass
class ReasoningResult:
    """推理结果"""
    conclusion: str
    confidence: float
    evidence: List[Dict[str, Any]]
    reasoning_path: List[str]
    rule_applied: str
    timestamp: datetime


@dataclass
class Association:
    """关联信息"""
    source_entity: str
    target_entity: str
    association_type: str
    strength: float
    evidence_count: int
    context: str


class KnowledgeReasoningEngine:
    """知识推理引擎"""
    
    def __init__(self, knowledge_graph, config: Dict[str, Any]):
        self.knowledge_graph = knowledge_graph
        self.config = config
        
        # 推理规则
        self.reasoning_rules = self._load_reasoning_rules()
        
        # 推理缓存
        self.reasoning_cache: Dict[str, ReasoningResult] = {}
        
        # 关联发现缓存
        self.association_cache: Dict[str, List[Association]] = {}
        
        # 推理统计
        self.reasoning_stats = {
            'total_reasonings': 0,
            'successful_reasonings': 0,
            'rules_used': defaultdict(int),
            'avg_confidence': 0.0
        }
    
    def _load_reasoning_rules(self) -> Dict[str, Any]:
        """加载推理规则"""
        return {
            'transitivity': {
                'description': '传递性推理：如果A与B相关，B与C相关，则A与C相关',
                'rule': 'if has_relation(A, B, R) and has_relation(B, C, R) then has_relation(A, C, R)',
                'confidence': 0.8
            },
            'inheritance': {
                'description': '继承推理：如果A是B的子类，则A具有B的所有属性',
                'rule': 'if is_a(A, B) and has_property(B, P, V) then has_property(A, P, V)',
                'confidence': 0.9
            },
            'composition': {
                'description': '组合推理：如果A是B的一部分，则A具有B的某些属性',
                'rule': 'if part_of(A, B) and has_property(B, P, V) then has_property(A, P, V)',
                'confidence': 0.7
            },
            'similarity': {
                'description': '相似性推理：如果A与B相似，则它们具有相似的属性',
                'rule': 'if similar_to(A, B) and has_property(A, P, V) then has_property(B, P, V)',
                'confidence': 0.6
            },
            'causation': {
                'description': '因果推理：如果A导致B，则A的存在暗示B的可能存在',
                'rule': 'if causes(A, B) and exists(A) then likely_exists(B)',
                'confidence': 0.75
            },
            'temporal': {
                'description': '时间推理：基于时间关系进行推理',
                'rule': 'if occurs_at(A, T1) and occurs_at(B, T2) and T1 < T2 then A_might_influence_B',
                'confidence': 0.65
            }
        }
    
    async def initialize(self):
        """初始化推理引擎"""
        print("知识推理引擎初始化完成")
    
    async def reason(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]] = None,
        max_reasoning_depth: int = 3
    ) -> Dict[str, Any]:
        """执行推理"""
        
        reasoning_id = f"reasoning_{hash(query)}_{datetime.now().timestamp()}"
        
        # 查找相关的实体
        related_entities = await self.knowledge_graph.find_entities_by_name(query)
        
        if not related_entities:
            return {
                'reasoning_id': reasoning_id,
                'query': query,
                'conclusions': [],
                'reasoning_steps': [],
                'confidence': 0.0,
                'status': 'no_entities_found'
            }
        
        reasoning_steps = []
        conclusions = []
        
        # 对每个相关实体进行推理
        for entity in related_entities:
            # 实体推理
            entity_reasoning = await self._reason_about_entity(entity.id, max_reasoning_depth)
            reasoning_steps.extend(entity_reasoning.get('steps', []))
            conclusions.extend(entity_reasoning.get('conclusions', []))
            
            # 关系推理
            relation_reasoning = await self._reason_about_relations(entity.id, max_reasoning_depth)
            reasoning_steps.extend(relation_reasoning.get('steps', []))
            conclusions.extend(relation_reasoning.get('conclusions', []))
        
        # 合并和去重结论
        unique_conclusions = self._merge_conclusions(conclusions)
        
        # 计算总体置信度
        overall_confidence = self._calculate_overall_confidence(unique_conclusions)
        
        # 更新统计
        self.reasoning_stats['total_reasonings'] += 1
        if unique_conclusions:
            self.reasoning_stats['successful_reasonings'] += 1
        
        result = {
            'reasoning_id': reasoning_id,
            'query': query,
            'related_entities': [asdict(entity) for entity in related_entities],
            'conclusions': unique_conclusions,
            'reasoning_steps': reasoning_steps,
            'confidence': overall_confidence,
            'status': 'completed',
            'timestamp': datetime.now()
        }
        
        # 缓存结果
        self.reasoning_cache[reasoning_id] = ReasoningResult(
            conclusion=str(unique_conclusions),
            confidence=overall_confidence,
            evidence=reasoning_steps,
            reasoning_path=[step.get('rule', '') for step in reasoning_steps],
            rule_applied='multi_rule',
            timestamp=datetime.now()
        )
        
        return result
    
    async def _reason_about_entity(self, entity_id: str, max_depth: int) -> Dict[str, Any]:
        """对实体进行推理"""
        
        entity = await self.knowledge_graph.get_entity(entity_id)
        if not entity:
            return {'steps': [], 'conclusions': []}
        
        reasoning_steps = []
        conclusions = []
        
        # 获取实体的所有关系
        relations = await self.knowledge_graph.get_relations(entity_id=entity_id)
        
        # 应用继承规则
        inheritance_conclusions = await self._apply_inheritance_rule(entity, relations)
        conclusions.extend(inheritance_conclusions)
        
        # 应用组合规则
        composition_conclusions = await self._apply_composition_rule(entity, relations)
        conclusions.extend(composition_conclusions)
        
        # 应用相似性规则
        similarity_conclusions = await self._apply_similarity_rule(entity, relations)
        conclusions.extend(similarity_conclusions)
        
        # 构建推理步骤
        for conclusion in conclusions:
            reasoning_steps.append({
                'type': 'entity_reasoning',
                'entity': entity.name,
                'conclusion': conclusion,
                'rule': 'inheritance/composition/similarity',
                'confidence': conclusion.get('confidence', 0.5)
            })
        
        return {'steps': reasoning_steps, 'conclusions': conclusions}
    
    async def _reason_about_relations(self, entity_id: str, max_depth: int) -> Dict[str, Any]:
        """对关系进行推理"""
        
        reasoning_steps = []
        conclusions = []
        
        # 获取直接关系
        direct_relations = await self.knowledge_graph.get_relations(entity_id=entity_id)
        
        # 应用传递性规则
        for relation in direct_relations:
            # 查找关系链
            relation_chains = await self._find_relation_chains(
                relation.source_entity_id, 
                relation.target_entity_id, 
                relation.relation_type,
                max_depth
            )
            
            for chain in relation_chains:
                conclusion = {
                    'type': 'transitive_relation',
                    'source': relation.source_entity_id,
                    'target': relation.target_entity_id,
                    'intermediate': chain.get('intermediate_entities', []),
                    'confidence': 0.8 * relation.confidence,
                    'rule': 'transitivity'
                }
                conclusions.append(conclusion)
                
                reasoning_steps.append({
                    'type': 'relation_reasoning',
                    'relation': asdict(relation),
                    'chain': chain,
                    'conclusion': conclusion,
                    'rule': 'transitivity',
                    'confidence': conclusion['confidence']
                })
        
        return {'steps': reasoning_steps, 'conclusions': conclusions}
    
    async def _apply_inheritance_rule(self, entity, relations) -> List[Dict[str, Any]]:
        """应用继承规则"""
        conclusions = []
        
        # 查找IS_A关系
        isa_relations = [
            r for r in relations 
            if r.relation_type.value == 'is_a'
        ]
        
        for isa_relation in isa_relations:
            parent_entity = await self.knowledge_graph.get_entity(isa_relation.target_entity_id)
            if parent_entity:
                # 继承父实体的属性
                for prop_name, prop_value in parent_entity.properties.items():
                    if prop_name not in entity.properties:
                        conclusion = {
                            'type': 'inherited_property',
                            'entity': entity.name,
                            'property': prop_name,
                            'value': prop_value,
                            'source': parent_entity.name,
                            'confidence': 0.9 * isa_relation.confidence,
                            'rule': 'inheritance'
                        }
                        conclusions.append(conclusion)
        
        return conclusions
    
    async def _apply_composition_rule(self, entity, relations) -> List[Dict[str, Any]]:
        """应用组合规则"""
        conclusions = []
        
        # 查找PART_OF关系
        part_relations = [
            r for r in relations 
            if r.relation_type.value == 'part_of'
        ]
        
        for part_relation in part_relations:
            whole_entity = await self.knowledge_graph.get_entity(part_relation.target_entity_id)
            if whole_entity:
                # 部分继承整体的一些属性
                for prop_name, prop_value in whole_entity.properties.items():
                    if prop_name in ['size', 'weight', 'color', 'material']:
                        conclusion = {
                            'type': 'composed_property',
                            'entity': entity.name,
                            'property': prop_name,
                            'value': prop_value,
                            'whole': whole_entity.name,
                            'confidence': 0.7 * part_relation.confidence,
                            'rule': 'composition'
                        }
                        conclusions.append(conclusion)
        
        return conclusions
    
    async def _apply_similarity_rule(self, entity, relations) -> List[Dict[str, Any]]:
        """应用相似性规则"""
        conclusions = []
        
        # 查找SIMILAR_TO关系
        similar_relations = [
            r for r in relations 
            if r.relation_type.value == 'similar_to'
        ]
        
        for similar_relation in similar_relations:
            similar_entity = await self.knowledge_graph.get_entity(similar_relation.target_entity_id)
            if similar_entity:
                # 相似实体可能具有相似的属性
                for prop_name, prop_value in similar_entity.properties.items():
                    if prop_name not in entity.properties:
                        conclusion = {
                            'type': 'similar_property',
                            'entity': entity.name,
                            'property': prop_name,
                            'value': prop_value,
                            'similar_to': similar_entity.name,
                            'confidence': 0.6 * similar_relation.confidence,
                            'rule': 'similarity'
                        }
                        conclusions.append(conclusion)
        
        return conclusions
    
    async def _find_relation_chains(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """查找关系链"""
        
        chains = []
        
        # 使用NetworkX查找路径
        try:
            paths = list(nx.all_simple_paths(
                self.knowledge_graph.graph,
                source_id,
                target_id,
                cutoff=max_depth
            ))
            
            for path in paths:
                intermediate_entities = []
                for i in range(1, len(path) - 1):
                    entity = await self.knowledge_graph.get_entity(path[i])
                    if entity:
                        intermediate_entities.append(entity.name)
                
                chains.append({
                    'path': path,
                    'intermediate_entities': intermediate_entities,
                    'length': len(path) - 1
                })
        except nx.NetworkXNoPath:
            pass
        
        return chains
    
    def _merge_conclusions(self, conclusions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并和去重结论"""
        
        # 按结论类型和内容分组
        conclusion_groups = defaultdict(list)
        
        for conclusion in conclusions:
            key = f"{conclusion.get('type', '')}:{conclusion.get('entity', '')}:{conclusion.get('property', '')}"
            conclusion_groups[key].append(conclusion)
        
        # 合并每组中的结论
        merged_conclusions = []
        
        for key, group_conclusions in conclusion_groups.items():
            if len(group_conclusions) == 1:
                merged_conclusions.append(group_conclusions[0])
            else:
                # 合并多个结论
                best_conclusion = max(group_conclusions, key=lambda x: x.get('confidence', 0))
                best_conclusion['evidence_count'] = len(group_conclusions)
                best_conclusion['confidence'] = min(1.0, best_conclusion.get('confidence', 0) + 0.1)
                merged_conclusions.append(best_conclusion)
        
        return merged_conclusions
    
    def _calculate_overall_confidence(self, conclusions: List[Dict[str, Any]]) -> float:
        """计算总体置信度"""
        
        if not conclusions:
            return 0.0
        
        # 计算加权平均置信度
        total_confidence = sum(c.get('confidence', 0) for c in conclusions)
        return total_confidence / len(conclusions)
    
    async def discover_associations(
        self,
        entity_id: str,
        max_depth: int = 2,
        min_strength: float = 0.5
    ) -> List[Association]:
        """发现关联"""
        
        cache_key = f"{entity_id}_{max_depth}_{min_strength}"
        if cache_key in self.association_cache:
            return self.association_cache[cache_key]
        
        associations = []
        
        # 获取实体的所有关系
        relations = await self.knowledge_graph.get_relations(entity_id=entity_id)
        
        for relation in relations:
            # 计算关联强度
            strength = self._calculate_association_strength(relation)
            
            if strength >= min_strength:
                target_entity = await self.knowledge_graph.get_entity(relation.target_entity_id)
                if target_entity:
                    association = Association(
                        source_entity=entity_id,
                        target_entity=target_entity.name,
                        association_type=relation.relation_type.value,
                        strength=strength,
                        evidence_count=1,
                        context=f"通过{relation.relation_type.value}关系连接"
                    )
                    associations.append(association)
        
        # 查找间接关联
        indirect_associations = await self._find_indirect_associations(
            entity_id, max_depth, min_strength
        )
        associations.extend(indirect_associations)
        
        # 按强度排序
        associations.sort(key=lambda x: x.strength, reverse=True)
        
        # 缓存结果
        self.association_cache[cache_key] = associations
        
        return associations
    
    async def _find_indirect_associations(
        self,
        entity_id: str,
        max_depth: int,
        min_strength: float
    ) -> List[Association]:
        """查找间接关联"""
        
        associations = []
        
        # 使用NetworkX查找多跳路径
        try:
            for target_id in self.knowledge_graph.entity_cache.keys():
                if target_id != entity_id:
                    # 查找路径
                    paths = list(nx.all_simple_paths(
                        self.knowledge_graph.graph,
                        entity_id,
                        target_id,
                        cutoff=max_depth
                    ))
                    
                    for path in paths:
                        if len(path) > 2:  # 间接关联
                            # 计算路径强度
                            path_strength = self._calculate_path_strength(path)
                            
                            if path_strength >= min_strength:
                                target_entity = await self.knowledge_graph.get_entity(target_id)
                                if target_entity:
                                    association = Association(
                                        source_entity=entity_id,
                                        target_entity=target_entity.name,
                                        association_type='indirect',
                                        strength=path_strength,
                                        evidence_count=len(path) - 1,
                                        context=f"通过{len(path)-2}个中间节点连接"
                                    )
                                    associations.append(association)
        except Exception as e:
            print(f"间接关联查找错误: {e}")
        
        return associations
    
    def _calculate_association_strength(self, relation) -> float:
        """计算关联强度"""
        # 基于关系置信度和权重计算强度
        base_strength = relation.confidence * relation.weight
        
        # 根据关系类型调整
        type_modifiers = {
            'is_a': 1.2,
            'part_of': 1.0,
            'similar_to': 0.8,
            'related_to': 0.7,
            'causes': 0.9,
            'located_in': 0.8,
            'interacts_with': 0.6
        }
        
        modifier = type_modifiers.get(relation.relation_type.value, 1.0)
        return min(1.0, base_strength * modifier)
    
    def _calculate_path_strength(self, path: List[str]) -> float:
        """计算路径强度"""
        if len(path) < 3:
            return 0.0
        
        # 获取路径上的所有关系
        path_relations = []
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]
            
            # 查找源和目标之间的关系
            for relation in self.knowledge_graph.relation_cache.values():
                if (relation.source_entity_id == source_id and relation.target_entity_id == target_id) or \
                   (relation.source_entity_id == target_id and relation.target_entity_id == source_id):
                    path_relations.append(relation)
                    break
        
        if not path_relations:
            return 0.0
        
        # 计算路径强度（关系强度的乘积）
        path_strength = 1.0
        for relation in path_relations:
            path_strength *= self._calculate_association_strength(relation)
        
        # 路径越长，强度衰减越多
        decay_factor = 0.9 ** (len(path) - 2)
        return path_strength * decay_factor
    
    async def predict_properties(
        self,
        entity_id: str,
        property_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """预测实体属性"""
        
        entity = await self.knowledge_graph.get_entity(entity_id)
        if not entity:
            return {}
        
        predictions = {}
        
        # 基于相似实体的属性预测
        similar_predictions = await self._predict_from_similarity(entity_id)
        predictions.update(similar_predictions)
        
        # 基于父类属性的预测
        inheritance_predictions = await self._predict_from_inheritance(entity_id)
        predictions.update(inheritance_predictions)
        
        # 基于部分-整体关系的预测
        composition_predictions = await self._predict_from_composition(entity_id)
        predictions.update(composition_predictions)
        
        # 过滤属性类型
        if property_types:
            predictions = {
                k: v for k, v in predictions.items()
                if any(prop_type in k.lower() for prop_type in property_types)
            }
        
        return predictions
    
    async def _predict_from_similarity(self, entity_id: str) -> Dict[str, Any]:
        """基于相似性预测属性"""
        predictions = {}
        
        relations = await self.knowledge_graph.get_relations(entity_id=entity_id)
        similar_relations = [
            r for r in relations 
            if r.relation_type.value == 'similar_to'
        ]
        
        for relation in similar_relations:
            similar_entity = await self.knowledge_graph.get_entity(relation.target_entity_id)
            if similar_entity:
                for prop_name, prop_value in similar_entity.properties.items():
                    if prop_name not in predictions:
                        predictions[f"predicted_similar_{prop_name}"] = {
                            'value': prop_value,
                            'confidence': 0.6 * relation.confidence,
                            'source': similar_entity.name,
                            'method': 'similarity'
                        }
        
        return predictions
    
    async def _predict_from_inheritance(self, entity_id: str) -> Dict[str, Any]:
        """基于继承预测属性"""
        predictions = {}
        
        relations = await self.knowledge_graph.get_relations(entity_id=entity_id)
        isa_relations = [
            r for r in relations 
            if r.relation_type.value == 'is_a'
        ]
        
        for relation in isa_relations:
            parent_entity = await self.knowledge_graph.get_entity(relation.target_entity_id)
            if parent_entity:
                for prop_name, prop_value in parent_entity.properties.items():
                    if prop_name not in predictions:
                        predictions[f"predicted_inherited_{prop_name}"] = {
                            'value': prop_value,
                            'confidence': 0.9 * relation.confidence,
                            'source': parent_entity.name,
                            'method': 'inheritance'
                        }
        
        return predictions
    
    async def _predict_from_composition(self, entity_id: str) -> Dict[str, Any]:
        """基于组合关系预测属性"""
        predictions = {}
        
        relations = await self.knowledge_graph.get_relations(entity_id=entity_id)
        part_relations = [
            r for r in relations 
            if r.relation_type.value == 'part_of'
        ]
        
        for relation in part_relations:
            whole_entity = await self.knowledge_graph.get_entity(relation.target_entity_id)
            if whole_entity:
                for prop_name, prop_value in whole_entity.properties.items():
                    if prop_name in ['size', 'weight', 'color', 'material']:
                        if prop_name not in predictions:
                            predictions[f"predicted_composed_{prop_name}"] = {
                                'value': prop_value,
                                'confidence': 0.7 * relation.confidence,
                                'source': whole_entity.name,
                                'method': 'composition'
                            }
        
        return predictions
    
    async def get_reasoning_statistics(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        
        stats = self.reasoning_stats.copy()
        
        # 添加缓存统计
        stats['cache_size'] = len(self.reasoning_cache)
        stats['association_cache_size'] = len(self.association_cache)
        
        # 计算平均置信度
        if self.reasoning_stats['total_reasonings'] > 0:
            stats['success_rate'] = (
                self.reasoning_stats['successful_reasonings'] / 
                self.reasoning_stats['total_reasonings']
            )
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    async def clear_cache(self):
        """清理推理缓存"""
        self.reasoning_cache.clear()
        self.association_cache.clear()
        print("推理缓存已清理")