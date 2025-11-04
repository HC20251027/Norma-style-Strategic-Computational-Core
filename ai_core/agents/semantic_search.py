"""
语义搜索引擎
基于向量数据库的智能语义搜索和相似性匹配
"""

import asyncio
import re
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    collection: str
    highlights: List[str]
    context: str


class SemanticSearchEngine:
    """语义搜索引擎"""
    
    def __init__(self, vector_db, config: Dict[str, Any]):
        self.vector_db = vector_db
        self.config = config
        
        # 搜索配置
        self.default_top_k = config.get('default_top_k', 10)
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.max_context_length = config.get('max_context_length', 500)
        
        # 搜索历史和统计
        self.search_history: List[Dict[str, Any]] = []
        self.query_cache: Dict[str, List[SearchResult]] = {}
        self.popular_queries: Dict[str, int] = {}
        
        # 文本处理工具
        self.stop_words = set([
            '的', '了', '是', '在', '有', '和', '就', '不', '人', '都', '一', '一个',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        ])
    
    async def initialize(self):
        """初始化语义搜索引擎"""
        # 确保必要的集合存在
        collections_to_create = ['memories', 'knowledge', 'documents', 'conversations']
        
        for collection in collections_to_create:
            if collection not in await self.vector_db.list_collections():
                await self.vector_db.create_collection(collection, 384)
        
        print("语义搜索引擎初始化完成")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的中文和英文关键词提取
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = re.findall(r'\b\w+\b', text)
        
        # 过滤停用词和短词
        keywords = [
            word for word in words 
            if len(word) > 1 and word not in self.stop_words
        ]
        
        return keywords
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        # 简单的词汇重叠相似度
        words1 = set(self._extract_keywords(text1))
        words2 = set(self._extract_keywords(text2))
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """生成文本嵌入（简化版本）"""
        # 这里使用一个简化的方法，实际应用中应该使用预训练的嵌入模型
        # 如 sentence-transformers, OpenAI embeddings, 或其他embedding模型
        
        keywords = self._extract_keywords(text)
        
        # 创建基于关键词的简单向量表示
        vector = np.zeros(384)  # 假设向量维度为384
        
        # 为每个关键词分配一个向量位置
        for i, keyword in enumerate(keywords[:384]):
            # 使用哈希将关键词映射到向量位置
            hash_value = int(hashlib.md5(keyword.encode()).hexdigest(), 16)
            vector[hash_value % 384] += 1.0
        
        # 归一化向量
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
        
        return vector.tolist()
    
    async def search(
        self,
        query: str,
        collection: str = 'memories',
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        include_content: bool = True,
        filter_tags: Optional[List[str]] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[SearchResult]:
        """语义搜索"""
        
        top_k = top_k or self.default_top_k
        threshold = threshold or self.similarity_threshold
        
        # 检查查询缓存
        cache_key = f"{query}_{collection}_{top_k}_{threshold}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # 记录搜索历史
        search_record = {
            'query': query,
            'collection': collection,
            'timestamp': datetime.now(),
            'top_k': top_k
        }
        self.search_history.append(search_record)
        
        # 更新热门查询统计
        self.popular_queries[query] = self.popular_queries.get(query, 0) + 1
        
        # 生成查询向量
        query_vector = self._generate_embedding(query)
        
        # 向量搜索
        vector_results = await self.vector_db.search_similar(
            query_vector=query_vector,
            collection=collection,
            top_k=top_k * 2,  # 获取更多结果用于过滤和排序
            threshold=threshold * 0.8,  # 降低阈值以获得更多候选
            filter_tags=filter_tags
        )
        
        # 构建搜索结果
        search_results = []
        for result in vector_results:
            vector_item = await self.vector_db.get_vector(result['id'])
            if not vector_item:
                continue
            
            # 时间范围过滤
            if time_range:
                start_time, end_time = time_range
                if not (start_time <= vector_item.created_at <= end_time):
                    continue
            
            # 计算文本相似度
            text_similarity = self._calculate_text_similarity(query, vector_item.metadata.get('content', ''))
            
            # 综合评分
            vector_score = result['similarity_score']
            final_score = (vector_score * 0.7) + (text_similarity * 0.3)
            
            # 高亮匹配文本
            highlights = self._find_highlights(query, vector_item.metadata.get('content', ''))
            
            search_result = SearchResult(
                id=result['id'],
                content=vector_item.metadata.get('content', ''),
                score=final_score,
                metadata=vector_item.metadata,
                collection=collection,
                highlights=highlights,
                context=self._extract_context(vector_item.metadata.get('content', ''), query)
            )
            
            search_results.append(search_result)
        
        # 按分数排序并限制结果数量
        search_results.sort(key=lambda x: x.score, reverse=True)
        search_results = search_results[:top_k]
        
        # 缓存结果
        self.query_cache[cache_key] = search_results
        
        # 清理旧缓存（避免内存溢出）
        if len(self.query_cache) > 1000:
            # 保留最近的100个缓存条目
            cache_items = list(self.query_cache.items())
            self.query_cache = dict(cache_items[-100:])
        
        return search_results
    
    def _find_highlights(self, query: str, content: str) -> List[str]:
        """查找高亮文本片段"""
        highlights = []
        query_keywords = self._extract_keywords(query)
        
        for keyword in query_keywords:
            # 查找包含关键词的句子
            sentences = re.split(r'[.!?。！？]', content)
            for sentence in sentences:
                if keyword.lower() in sentence.lower():
                    # 截取包含关键词的片段
                    start = max(0, sentence.lower().find(keyword.lower()) - 20)
                    end = min(len(sentence), start + len(keyword) + 40)
                    highlight = sentence[start:end].strip()
                    if highlight and highlight not in highlights:
                        highlights.append(highlight)
        
        return highlights[:3]  # 最多返回3个高亮片段
    
    def _extract_context(self, content: str, query: str) -> str:
        """提取上下文"""
        if not content:
            return ""
        
        # 简单的上下文提取：返回内容的前N个字符
        max_length = self.max_context_length
        if len(content) <= max_length:
            return content
        
        # 尝试在查询关键词附近截取
        query_keywords = self._extract_keywords(query)
        for keyword in query_keywords:
            keyword_pos = content.lower().find(keyword.lower())
            if keyword_pos != -1:
                start = max(0, keyword_pos - max_length // 2)
                end = min(len(content), start + max_length)
                return content[start:end]
        
        # 如果没有找到关键词，返回开头部分
        return content[:max_length]
    
    async def fuzzy_search(
        self,
        query: str,
        collection: str = 'memories',
        max_distance: int = 2,
        top_k: int = 10
    ) -> List[SearchResult]:
        """模糊搜索"""
        # 这里实现一个简单的模糊搜索
        # 实际应用中可以使用更复杂的算法如编辑距离、模糊匹配等
        
        # 首先进行精确搜索
        exact_results = await self.search(query, collection, top_k * 2)
        
        # 如果精确搜索结果足够好，直接返回
        if len(exact_results) >= top_k and exact_results[0].score > 0.8:
            return exact_results[:top_k]
        
        # 进行模糊搜索
        fuzzy_results = []
        query_keywords = self._extract_keywords(query)
        
        # 为每个关键词搜索
        for keyword in query_keywords:
            keyword_results = await self.search(
                keyword, 
                collection, 
                top_k=5,
                threshold=0.3  # 降低阈值
            )
            
            for result in keyword_results:
                # 计算与原查询的相似度
                similarity = self._calculate_text_similarity(query, result.content)
                result.score = (result.score + similarity) / 2
                fuzzy_results.append(result)
        
        # 去重并排序
        unique_results = {}
        for result in fuzzy_results:
            if result.id not in unique_results or result.score > unique_results[result.id].score:
                unique_results[result.id] = result
        
        fuzzy_results = list(unique_results.values())
        fuzzy_results.sort(key=lambda x: x.score, reverse=True)
        
        return fuzzy_results[:top_k]
    
    async def multi_field_search(
        self,
        query: str,
        fields: Dict[str, float],
        collection: str = 'memories',
        top_k: int = 10
    ) -> List[SearchResult]:
        """多字段搜索"""
        
        # 生成查询向量
        query_vector = self._generate_embedding(query)
        
        # 在指定集合中搜索
        vector_results = await self.vector_db.search_similar(
            query_vector=query_vector,
            collection=collection,
            top_k=top_k * 2,
            threshold=0.5
        )
        
        search_results = []
        for result in vector_results:
            vector_item = await self.vector_db.get_vector(result['id'])
            if not vector_item:
                continue
            
            # 计算字段匹配分数
            field_scores = {}
            for field, weight in fields.items():
                field_value = vector_item.metadata.get(field, '')
                if field_value:
                    field_similarity = self._calculate_text_similarity(query, field_value)
                    field_scores[field] = field_similarity * weight
            
            # 计算加权平均分数
            if field_scores:
                weighted_score = sum(field_scores.values()) / sum(fields.values())
                final_score = (result['similarity_score'] * 0.6) + (weighted_score * 0.4)
            else:
                final_score = result['similarity_score']
            
            search_result = SearchResult(
                id=result['id'],
                content=vector_item.metadata.get('content', ''),
                score=final_score,
                metadata=vector_item.metadata,
                collection=collection,
                highlights=[],
                context=''
            )
            
            search_results.append(search_result)
        
        # 排序并返回结果
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results[:top_k]
    
    async def expand_query(self, query: str, expansion_terms: int = 5) -> List[str]:
        """查询扩展"""
        # 简单的查询扩展：基于搜索历史中的共现词汇
        
        expanded_terms = [query]
        
        # 从搜索历史中查找相关查询
        related_queries = []
        for search_record in self.search_history[-100:]:  # 最近100次搜索
            similarity = self._calculate_text_similarity(query, search_record['query'])
            if similarity > 0.3:  # 相似度阈值
                related_queries.append((search_record['query'], similarity))
        
        # 按相似度排序
        related_queries.sort(key=lambda x: x[1], reverse=True)
        
        # 添加相关查询
        for related_query, _ in related_queries[:expansion_terms]:
            if related_query not in expanded_terms:
                expanded_terms.append(related_query)
        
        return expanded_terms[:expansion_terms + 1]
    
    async def get_search_analytics(self) -> Dict[str, Any]:
        """获取搜索分析数据"""
        
        # 热门查询
        top_queries = sorted(
            self.popular_queries.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # 搜索趋势（最近7天）
        recent_searches = [
            record for record in self.search_history
            if record['timestamp'] > datetime.now() - timedelta(days=7)
        ]
        
        daily_searches = {}
        for record in recent_searches:
            date = record['timestamp'].date().isoformat()
            daily_searches[date] = daily_searches.get(date, 0) + 1
        
        # 集合使用统计
        collection_stats = {}
        for record in self.search_history:
            collection = record['collection']
            collection_stats[collection] = collection_stats.get(collection, 0) + 1
        
        return {
            'total_searches': len(self.search_history),
            'popular_queries': top_queries,
            'daily_searches': daily_searches,
            'collection_usage': collection_stats,
            'cache_size': len(self.query_cache)
        }
    
    async def clear_cache(self):
        """清理搜索缓存"""
        self.query_cache.clear()
        print("搜索缓存已清理")
    
    async def optimize_index(self, collection: str):
        """优化搜索索引"""
        # 这里可以添加索引优化逻辑
        # 如重新训练索引、清理无效数据等
        print(f"正在优化集合 {collection} 的索引...")
        
        # 获取集合统计
        stats = await self.vector_db.get_collection_stats(collection)
        if stats:
            print(f"集合 {collection} 统计: {stats}")
        
        print(f"集合 {collection} 的索引优化完成")