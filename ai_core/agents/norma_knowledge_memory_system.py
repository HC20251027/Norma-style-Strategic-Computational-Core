#!/usr/bin/env python3
"""
诺玛Agent知识库和记忆系统
基于Agno框架和LanceDb的向量数据库集成

功能模块:
1. 向量数据库集成（基于LanceDb）
2. 动态知识更新和学习
3. 上下文记忆管理
4. 个性化用户画像
5. 实时知识检索和RAG

作者: 皇
创建时间: 2025-11-01
版本: 1.0.0
"""

import asyncio
import json
import os
import tempfile
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# 第三方库
import lancedb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Agno框架
from agno.knowledge import Knowledge
from agno.memory import MemoryManager, UserMemory
from agno.media import Image, Audio, Video

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """知识库条目"""
    id: str
    content: str
    content_type: str  # text, image, audio, video
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    updated_at: datetime = None
    access_count: int = 0
    relevance_score: float = 0.0
    source: str = "unknown"
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    preferences: Dict[str, Any]
    interaction_history: List[Dict[str, Any]]
    knowledge_areas: List[str]
    communication_style: str
    expertise_level: Dict[str, float]
    learning_patterns: Dict[str, Any]
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class ContextMemory:
    """上下文记忆"""
    session_id: str
    user_id: str
    messages: List[Dict[str, Any]]
    context_summary: str
    key_topics: List[str]
    current_task: Optional[str] = None
    created_at: datetime = None
    last_accessed: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = self.created_at


class NormaVectorDatabase:
    """诺玛向量数据库管理器 - 基于LanceDb"""
    
    def __init__(self, db_path: str = None, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        初始化向量数据库
        
        Args:
            db_path: 数据库路径，如果为None则使用临时目录
            embedding_model: 文本嵌入模型名称
        """
        self.db_path = db_path or os.path.join(tempfile.gettempdir(), f"norma_knowledge_db_{uuid.uuid4().hex[:8]}")
        self.embedding_model_name = embedding_model
        
        # 初始化LanceDb连接
        self.db = lancedb.connect(self.db_path)
        
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # 知识库表
        self.knowledge_table = None
        self._initialize_tables()
        
        logger.info(f"诺玛向量数据库初始化完成: {self.db_path}")
    
    def _initialize_tables(self):
        """初始化数据库表"""
        try:
            # 知识库表
            schema = {
                "id": "string",
                "content": "string",
                "content_type": "string",
                "metadata": "string",  # JSON字符串
                "embedding": "vector",  # 1536维向量
                "created_at": "timestamp",
                "updated_at": "timestamp",
                "access_count": "int32",
                "relevance_score": "float32",
                "source": "string"
            }
            
            self.knowledge_table = self.db.create_table("knowledge", schema=schema)
            logger.info("知识库表创建成功")
            
        except Exception as e:
            # 表可能已存在，尝试获取
            try:
                self.knowledge_table = self.db.open_table("knowledge")
                logger.info("知识库表加载成功")
            except Exception as e2:
                logger.error(f"初始化知识库表失败: {e2}")
                raise
    
    def add_knowledge_entry(self, entry: KnowledgeEntry) -> str:
        """添加知识条目"""
        try:
            # 生成嵌入向量
            if entry.embedding is None:
                entry.embedding = self.embedding_model.encode(entry.content).tolist()
            
            # 准备数据
            data = {
                "id": entry.id,
                "content": entry.content,
                "content_type": entry.content_type,
                "metadata": json.dumps(entry.metadata, ensure_ascii=False),
                "embedding": entry.embedding,
                "created_at": entry.created_at,
                "updated_at": entry.updated_at,
                "access_count": entry.access_count,
                "relevance_score": entry.relevance_score,
                "source": entry.source
            }
            
            # 插入数据
            self.knowledge_table.add([data])
            logger.info(f"知识条目添加成功: {entry.id}")
            
            return entry.id
            
        except Exception as e:
            logger.error(f"添加知识条目失败: {e}")
            raise
    
    def search_knowledge(self, query: str, limit: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """搜索知识库"""
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # 向量搜索
            results = self.knowledge_table.search(query_embedding).limit(limit).to_pandas()
            
            # 过滤低相关性结果
            filtered_results = results[results['relevance_score'] >= threshold]
            
            # 转换为字典格式
            search_results = []
            for _, row in filtered_results.iterrows():
                result = {
                    "id": row['id'],
                    "content": row['content'],
                    "content_type": row['content_type'],
                    "metadata": json.loads(row['metadata']),
                    "relevance_score": row['relevance_score'],
                    "source": row['source'],
                    "created_at": row['created_at']
                }
                search_results.append(result)
            
            # 更新访问计数
            for result in search_results:
                self._update_access_count(result['id'])
            
            logger.info(f"知识库搜索完成: 查询='{query}', 结果数={len(search_results)}")
            return search_results
            
        except Exception as e:
            logger.error(f"知识库搜索失败: {e}")
            return []
    
    def _update_access_count(self, entry_id: str):
        """更新访问计数"""
        try:
            # 获取当前记录
            current_data = self.knowledge_table.to_pandas()
            entry_row = current_data[current_data['id'] == entry_id]
            
            if not entry_row.empty:
                new_count = entry_row['access_count'].iloc[0] + 1
                # 更新访问计数
                self.knowledge_table.update({
                    "access_count": new_count
                }).where(f"id = '{entry_id}'").execute()
                
        except Exception as e:
            logger.warning(f"更新访问计数失败: {e}")
    
    def update_knowledge_entry(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """更新知识条目"""
        try:
            # 如果内容更新，重新生成嵌入向量
            if 'content' in updates:
                updates['embedding'] = self.embedding_model.encode(updates['content']).tolist()
            
            # 添加更新时间
            updates['updated_at'] = datetime.now()
            
            # 更新数据
            self.knowledge_table.update(updates).where(f"id = '{entry_id}'").execute()
            
            logger.info(f"知识条目更新成功: {entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"更新知识条目失败: {e}")
            return False
    
    def delete_knowledge_entry(self, entry_id: str) -> bool:
        """删除知识条目"""
        try:
            self.knowledge_table.delete(f"id = '{entry_id}'").execute()
            logger.info(f"知识条目删除成功: {entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除知识条目失败: {e}")
            return False
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            data = self.knowledge_table.to_pandas()
            
            stats = {
                "total_entries": len(data),
                "content_types": data['content_type'].value_counts().to_dict(),
                "total_access_count": data['access_count'].sum(),
                "average_relevance_score": data['relevance_score'].mean(),
                "most_accessed": data.nlargest(5, 'access_count')[['id', 'content', 'access_count']].to_dict('records'),
                "recent_entries": data.nlargest(5, 'created_at')[['id', 'content', 'created_at']].to_dict('records')
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"获取知识库统计失败: {e}")
            return {}


class NormaKnowledgeManager:
    """诺玛知识管理器"""
    
    def __init__(self, vector_db: NormaVectorDatabase):
        """
        初始化知识管理器
        
        Args:
            vector_db: 向量数据库实例
        """
        self.vector_db = vector_db
        self.learning_cache = {}  # 学习缓存
        
        logger.info("诺玛知识管理器初始化完成")
    
    async def add_knowledge(self, content: str, content_type: str = "text", 
                           metadata: Dict[str, Any] = None, source: str = "user") -> str:
        """添加知识"""
        try:
            # 创建知识条目
            entry = KnowledgeEntry(
                id=str(uuid.uuid4()),
                content=content,
                content_type=content_type,
                metadata=metadata or {},
                source=source
            )
            
            # 计算初始相关性分数
            entry.relevance_score = self._calculate_relevance_score(content, metadata)
            
            # 添加到向量数据库
            entry_id = self.vector_db.add_knowledge_entry(entry)
            
            # 触发学习更新
            await self._trigger_learning_update(entry)
            
            logger.info(f"知识添加成功: {entry_id}")
            return entry_id
            
        except Exception as e:
            logger.error(f"添加知识失败: {e}")
            raise
    
    def _calculate_relevance_score(self, content: str, metadata: Dict[str, Any]) -> float:
        """计算相关性分数"""
        try:
            # 基于内容长度、关键词密度等计算基础分数
            base_score = min(len(content) / 1000.0, 1.0)  # 内容长度分数
            
            # 关键词密度分数
            keywords = metadata.get('keywords', []) if metadata else []
            keyword_density = len(keywords) / max(len(content.split()), 1)
            keyword_score = min(keyword_density * 10, 1.0)
            
            # 综合分数
            relevance_score = (base_score * 0.6 + keyword_score * 0.4)
            
            return min(relevance_score, 1.0)
            
        except Exception as e:
            logger.warning(f"计算相关性分数失败: {e}")
            return 0.5
    
    async def _trigger_learning_update(self, entry: KnowledgeEntry):
        """触发学习更新"""
        try:
            # 更新学习缓存
            cache_key = f"{entry.content_type}_{hashlib.md5(entry.content.encode()).hexdigest()[:8]}"
            
            if cache_key not in self.learning_cache:
                self.learning_cache[cache_key] = {
                    "entry_id": entry.id,
                    "content": entry.content,
                    "content_type": entry.content_type,
                    "last_updated": datetime.now(),
                    "learning_count": 0
                }
            
            # 增加学习计数
            self.learning_cache[cache_key]["learning_count"] += 1
            self.learning_cache[cache_key]["last_updated"] = datetime.now()
            
            # 如果学习次数达到阈值，更新相关性分数
            if self.learning_cache[cache_key]["learning_count"] >= 5:
                await self._update_relevance_score(entry.id)
            
        except Exception as e:
            logger.warning(f"触发学习更新失败: {e}")
    
    async def _update_relevance_score(self, entry_id: str):
        """更新相关性分数"""
        try:
            # 获取知识库统计
            stats = self.vector_db.get_knowledge_stats()
            
            # 重新计算相关性分数
            # 这里可以实现更复杂的算法，比如基于访问频率、用户反馈等
            new_score = min(stats.get("average_relevance_score", 0.5) + 0.1, 1.0)
            
            # 更新数据库
            self.vector_db.update_knowledge_entry(entry_id, {"relevance_score": new_score})
            
            logger.info(f"相关性分数更新成功: {entry_id} -> {new_score}")
            
        except Exception as e:
            logger.warning(f"更新相关性分数失败: {e}")
    
    async def search_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索知识"""
        try:
            results = self.vector_db.search_knowledge(query, limit)
            
            # 记录搜索历史（可以用于个性化）
            search_record = {
                "query": query,
                "timestamp": datetime.now(),
                "results_count": len(results),
                "top_result_relevance": results[0]["relevance_score"] if results else 0.0
            }
            
            # 这里可以添加到用户画像的学习模式中
            
            return results
            
        except Exception as e:
            logger.error(f"搜索知识失败: {e}")
            return []
    
    async def update_knowledge(self, entry_id: str, new_content: str, 
                              new_metadata: Dict[str, Any] = None) -> bool:
        """更新知识"""
        try:
            updates = {"content": new_content}
            if new_metadata:
                updates["metadata"] = new_metadata
            
            success = self.vector_db.update_knowledge_entry(entry_id, updates)
            
            if success:
                # 触发学习更新
                entry = KnowledgeEntry(
                    id=entry_id,
                    content=new_content,
                    content_type="text",
                    metadata=new_metadata or {}
                )
                await self._trigger_learning_update(entry)
            
            return success
            
        except Exception as e:
            logger.error(f"更新知识失败: {e}")
            return False
    
    async def delete_knowledge(self, entry_id: str) -> bool:
        """删除知识"""
        try:
            return self.vector_db.delete_knowledge_entry(entry_id)
            
        except Exception as e:
            logger.error(f"删除知识失败: {e}")
            return False


class NormaMemoryManager:
    """诺玛记忆管理器"""
    
    def __init__(self, memory_db_path: str = None):
        """
        初始化记忆管理器
        
        Args:
            memory_db_path: 记忆数据库路径
        """
        self.memory_db_path = memory_db_path or os.path.join(
            tempfile.gettempdir(), f"norma_memory_db_{uuid.uuid4().hex[:8]}"
        )
        
        # 初始化Agno记忆管理器
        self.agno_memory = MemoryManager()
        
        # 上下文记忆存储
        self.context_memories: Dict[str, ContextMemory] = {}
        
        # 用户画像存储
        self.user_profiles: Dict[str, UserProfile] = {}
        
        logger.info("诺玛记忆管理器初始化完成")
    
    async def store_context_memory(self, session_id: str, user_id: str, 
                                  message: Dict[str, Any]) -> str:
        """存储上下文记忆"""
        try:
            # 获取或创建上下文记忆
            if session_id not in self.context_memories:
                self.context_memories[session_id] = ContextMemory(
                    session_id=session_id,
                    user_id=user_id,
                    messages=[],
                    context_summary="",
                    key_topics=[]
                )
            
            context_memory = self.context_memories[session_id]
            
            # 添加消息
            context_memory.messages.append({
                **message,
                "timestamp": datetime.now().isoformat()
            })
            
            # 更新最后访问时间
            context_memory.last_accessed = datetime.now()
            
            # 定期生成上下文摘要
            if len(context_memory.messages) % 10 == 0:
                await self._generate_context_summary(session_id)
            
            # 更新用户画像
            await self._update_user_profile_from_context(user_id, message)
            
            logger.info(f"上下文记忆存储成功: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"存储上下文记忆失败: {e}")
            raise
    
    async def _generate_context_summary(self, session_id: str):
        """生成上下文摘要"""
        try:
            context_memory = self.context_memories.get(session_id)
            if not context_memory:
                return
            
            # 提取关键主题
            recent_messages = context_memory.messages[-20:]  # 最近20条消息
            all_content = " ".join([msg.get("content", "") for msg in recent_messages])
            
            # 简单的关键词提取（实际应用中可以使用更复杂的NLP技术）
            words = all_content.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # 只考虑长度大于3的词
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # 获取频率最高的词作为关键主题
            key_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            context_memory.key_topics = [topic[0] for topic in key_topics]
            
            # 生成摘要（简化版本）
            if recent_messages:
                context_memory.context_summary = f"会话包含{len(recent_messages)}条消息，涉及主题：{', '.join(context_memory.key_topics)}"
            
            logger.info(f"上下文摘要生成成功: {session_id}")
            
        except Exception as e:
            logger.warning(f"生成上下文摘要失败: {e}")
    
    async def _update_user_profile_from_context(self, user_id: str, message: Dict[str, Any]):
        """从上下文更新用户画像"""
        try:
            # 获取或创建用户画像
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile(
                    user_id=user_id,
                    preferences={},
                    interaction_history=[],
                    knowledge_areas=[],
                    communication_style="neutral",
                    expertise_level={},
                    learning_patterns={}
                )
            
            user_profile = self.user_profiles[user_id]
            
            # 添加交互历史
            user_profile.interaction_history.append({
                **message,
                "timestamp": datetime.now().isoformat()
            })
            
            # 分析通信风格（简化版本）
            content = message.get("content", "").lower()
            if "?" in content or "？" in content:
                user_profile.communication_style = "inquisitive"
            elif len(content) > 200:
                user_profile.communication_style = "detailed"
            elif len(content) < 50:
                user_profile.communication_style = "concise"
            
            # 更新专业知识水平（基于关键词）
            expertise_keywords = {
                "技术": ["代码", "编程", "系统", "架构", "算法"],
                "商业": ["市场", "营销", "销售", "客户", "业务"],
                "设计": ["界面", "用户体验", "视觉", "创意", "设计"],
                "数据": ["分析", "统计", "数据", "图表", "报告"]
            }
            
            for area, keywords in expertise_keywords.items():
                if any(keyword in content for keyword in keywords):
                    current_level = user_profile.expertise_level.get(area, 0.0)
                    user_profile.expertise_level[area] = min(current_level + 0.1, 1.0)
            
            # 更新最后修改时间
            user_profile.updated_at = datetime.now()
            
        except Exception as e:
            logger.warning(f"更新用户画像失败: {e}")
    
    async def get_context_memory(self, session_id: str) -> Optional[ContextMemory]:
        """获取上下文记忆"""
        try:
            context_memory = self.context_memories.get(session_id)
            if context_memory:
                context_memory.last_accessed = datetime.now()
            return context_memory
            
        except Exception as e:
            logger.error(f"获取上下文记忆失败: {e}")
            return None
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """获取用户画像"""
        try:
            return self.user_profiles.get(user_id)
            
        except Exception as e:
            logger.error(f"获取用户画像失败: {e}")
            return None
    
    async def search_context_memories(self, user_id: str, query: str = None) -> List[ContextMemory]:
        """搜索上下文记忆"""
        try:
            user_memories = [
                memory for memory in self.context_memories.values() 
                if memory.user_id == user_id
            ]
            
            if query:
                # 简单的文本匹配搜索
                filtered_memories = []
                for memory in user_memories:
                    if (query.lower() in memory.context_summary.lower() or 
                        any(query.lower() in topic.lower() for topic in memory.key_topics)):
                        filtered_memories.append(memory)
                return filtered_memories
            
            return user_memories
            
        except Exception as e:
            logger.error(f"搜索上下文记忆失败: {e}")
            return []
    
    async def clear_old_memories(self, days: int = 30):
        """清理旧记忆"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 清理旧的上下文记忆
            old_sessions = [
                session_id for session_id, memory in self.context_memories.items()
                if memory.last_accessed < cutoff_date
            ]
            
            for session_id in old_sessions:
                del self.context_memories[session_id]
            
            # 清理用户画像中的旧交互历史
            for user_profile in self.user_profiles.values():
                old_interactions = [
                    interaction for interaction in user_profile.interaction_history
                    if datetime.fromisoformat(interaction["timestamp"]) < cutoff_date
                ]
                user_profile.interaction_history = old_interactions
            
            logger.info(f"清理旧记忆完成，删除了{len(old_sessions)}个旧会话")
            
        except Exception as e:
            logger.error(f"清理旧记忆失败: {e}")


class NormaRAGSystem:
    """诺玛检索增强生成系统"""
    
    def __init__(self, knowledge_manager: NormaKnowledgeManager, 
                 memory_manager: NormaMemoryManager):
        """
        初始化RAG系统
        
        Args:
            knowledge_manager: 知识管理器
            memory_manager: 记忆管理器
        """
        self.knowledge_manager = knowledge_manager
        self.memory_manager = memory_manager
        
        logger.info("诺玛RAG系统初始化完成")
    
    async def retrieve_relevant_knowledge(self, query: str, user_id: str = None, 
                                        session_id: str = None, limit: int = 5) -> Dict[str, Any]:
        """检索相关知识"""
        try:
            # 1. 搜索知识库
            knowledge_results = await self.knowledge_manager.search_knowledge(query, limit)
            
            # 2. 获取上下文记忆
            context_info = {}
            if session_id:
                context_memory = await self.memory_manager.get_context_memory(session_id)
                if context_memory:
                    context_info = {
                        "context_summary": context_memory.context_summary,
                        "key_topics": context_memory.key_topics,
                        "recent_messages": context_memory.messages[-5:]  # 最近5条消息
                    }
            
            # 3. 获取用户画像
            user_profile = None
            if user_id:
                user_profile = await self.memory_manager.get_user_profile(user_id)
            
            # 4. 整合检索结果
            retrieval_result = {
                "query": query,
                "knowledge_results": knowledge_results,
                "context_info": context_info,
                "user_profile": asdict(user_profile) if user_profile else None,
                "retrieval_timestamp": datetime.now().isoformat(),
                "total_sources": len(knowledge_results)
            }
            
            logger.info(f"知识检索完成: 查询='{query}', 来源数={len(knowledge_results)}")
            return retrieval_result
            
        except Exception as e:
            logger.error(f"检索相关知识失败: {e}")
            return {
                "query": query,
                "knowledge_results": [],
                "context_info": {},
                "user_profile": None,
                "error": str(e)
            }
    
    async def generate_enhanced_response(self, query: str, user_id: str = None, 
                                       session_id: str = None) -> Dict[str, Any]:
        """生成增强响应"""
        try:
            # 1. 检索相关知识
            retrieval_result = await self.retrieve_relevant_knowledge(query, user_id, session_id)
            
            # 2. 构建增强提示
            enhanced_prompt = self._build_enhanced_prompt(query, retrieval_result)
            
            # 3. 生成响应（这里可以集成实际的LLM）
            response = await self._generate_llm_response(enhanced_prompt, retrieval_result)
            
            # 4. 存储交互记忆
            if session_id and user_id:
                await self.memory_manager.store_context_memory(
                    session_id, user_id, 
                    {"role": "user", "content": query}
                )
                await self.memory_manager.store_context_memory(
                    session_id, user_id,
                    {"role": "assistant", "content": response, "sources": retrieval_result["knowledge_results"]}
                )
            
            # 5. 返回增强响应
            enhanced_response = {
                "query": query,
                "response": response,
                "sources": retrieval_result["knowledge_results"],
                "context_used": bool(retrieval_result["context_info"]),
                "user_profile_applied": user_id is not None,
                "generation_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"增强响应生成完成: 查询='{query}'")
            return enhanced_response
            
        except Exception as e:
            logger.error(f"生成增强响应失败: {e}")
            return {
                "query": query,
                "response": f"抱歉，生成响应时出现错误: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def _build_enhanced_prompt(self, query: str, retrieval_result: Dict[str, Any]) -> str:
        """构建增强提示"""
        try:
            prompt_parts = [
                f"用户查询: {query}",
                "",
                "相关知识:",
            ]
            
            # 添加知识库结果
            for i, result in enumerate(retrieval_result["knowledge_results"], 1):
                prompt_parts.append(f"{i}. {result['content']} (来源: {result['source']})")
            
            # 添加上下文信息
            if retrieval_result["context_info"]:
                context = retrieval_result["context_info"]
                prompt_parts.extend([
                    "",
                    "对话上下文:",
                    f"摘要: {context.get('context_summary', '')}",
                    f"关键主题: {', '.join(context.get('key_topics', []))}"
                ])
            
            # 添加用户画像信息
            if retrieval_result["user_profile"]:
                profile = retrieval_result["user_profile"]
                prompt_parts.extend([
                    "",
                    "用户特征:",
                    f"沟通风格: {profile.get('communication_style', 'unknown')}",
                    f"专业知识: {profile.get('expertise_level', {})}"
                ])
            
            prompt_parts.extend([
                "",
                "请基于以上信息回答用户的问题，确保回答准确、相关且符合用户的沟通风格。"
            ])
            
            return "\n".join(prompt_parts)
            
        except Exception as e:
            logger.warning(f"构建增强提示失败: {e}")
            return query
    
    async def _generate_llm_response(self, prompt: str, retrieval_result: Dict[str, Any]) -> str:
        """生成LLM响应"""
        try:
            # 这里应该集成实际的LLM API调用
            # 为了演示，我们返回一个模拟响应
            
            knowledge_count = len(retrieval_result["knowledge_results"])
            context_used = bool(retrieval_result["context_info"])
            
            response = f"基于检索到的{knowledge_count}个知识源"
            if context_used:
                response += "和对话上下文"
            response += "，我为您分析如下：\n\n"
            response += prompt.split('请基于')[0] + "\n\n"
            response += "详细回答：\n"
            response += "这里应该是基于检索结果生成的详细回答。由于这是演示版本，实际应用中会调用真实的LLM API。\n\n"
            response += f"检索到的知识源提供了相关信息，{'结合对话上下文可以更好地理解您的需求。' if context_used else ''}"
            
            return response
            
        except Exception as e:
            logger.warning(f"生成LLM响应失败: {e}")
            return "抱歉，生成响应时遇到技术问题。请稍后重试。"


class NormaKnowledgeMemoryOrchestrator:
    """诺玛知识记忆系统编排器"""
    
    def __init__(self, db_path: str = None):
        """
        初始化编排器
        
        Args:
            db_path: 数据库路径
        """
        # 初始化各个组件
        self.vector_db = NormaVectorDatabase(db_path)
        self.knowledge_manager = NormaKnowledgeManager(self.vector_db)
        self.memory_manager = NormaMemoryManager()
        self.rag_system = NormaRAGSystem(self.knowledge_manager, self.memory_manager)
        
        logger.info("诺玛知识记忆系统编排器初始化完成")
    
    async def process_user_input(self, user_input: str, user_id: str = None, 
                               session_id: str = None) -> Dict[str, Any]:
        """处理用户输入"""
        try:
            # 1. 生成增强响应
            response = await self.rag_system.generate_enhanced_response(
                user_input, user_id, session_id
            )
            
            # 2. 添加新知识（如果用户提供了新信息）
            if self._is_knowledge_content(user_input):
                await self.knowledge_manager.add_knowledge(
                    user_input, 
                    content_type="text",
                    metadata={"source": "user_input", "user_id": user_id},
                    source="user"
                )
            
            # 3. 返回处理结果
            result = {
                "user_input": user_input,
                "response": response,
                "knowledge_added": self._is_knowledge_content(user_input),
                "processing_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"用户输入处理完成: {user_id or 'anonymous'}")
            return result
            
        except Exception as e:
            logger.error(f"处理用户输入失败: {e}")
            return {
                "user_input": user_input,
                "error": str(e),
                "processing_timestamp": datetime.now().isoformat()
            }
    
    def _is_knowledge_content(self, content: str) -> bool:
        """判断内容是否为知识内容"""
        # 简单的启发式判断
        knowledge_indicators = [
            "事实", "信息", "知识", "数据", "说明", "解释",
            "definition", "information", "fact", "data"
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in knowledge_indicators)
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        try:
            # 知识库统计
            knowledge_stats = self.vector_db.get_knowledge_stats()
            
            # 记忆统计
            memory_stats = {
                "context_memories_count": len(self.memory_manager.context_memories),
                "user_profiles_count": len(self.memory_manager.user_profiles),
                "total_interactions": sum(
                    len(profile.interaction_history) 
                    for profile in self.memory_manager.user_profiles.values()
                )
            }
            
            # RAG统计（模拟）
            rag_stats = {
                "total_queries_processed": len(self.memory_manager.context_memories),
                "average_sources_per_query": knowledge_stats.get("total_entries", 0) / max(len(self.memory_manager.context_memories), 1)
            }
            
            system_stats = {
                "knowledge_base": knowledge_stats,
                "memory_system": memory_stats,
                "rag_system": rag_stats,
                "system_uptime": datetime.now().isoformat()
            }
            
            return system_stats
            
        except Exception as e:
            logger.error(f"获取系统统计失败: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """清理系统资源"""
        try:
            # 清理旧记忆
            await self.memory_manager.clear_old_memories()
            
            # 清理学习缓存
            self.knowledge_manager.learning_cache.clear()
            
            logger.info("系统资源清理完成")
            
        except Exception as e:
            logger.error(f"清理系统资源失败: {e}")


# 工厂函数
def create_norma_knowledge_memory_system(db_path: str = None) -> NormaKnowledgeMemoryOrchestrator:
    """
    创建诺玛知识记忆系统实例
    
    Args:
        db_path: 数据库路径
        
    Returns:
        诺玛知识记忆系统编排器实例
    """
    return NormaKnowledgeMemoryOrchestrator(db_path)


# 使用示例
async def main():
    """主函数示例"""
    # 创建系统
    system = create_norma_knowledge_memory_system()
    
    # 添加一些示例知识
    await system.knowledge_manager.add_knowledge(
        "Python是一种高级编程语言，具有简洁的语法和强大的功能。",
        metadata={"keywords": ["Python", "编程", "语言"], "category": "技术"}
    )
    
    await system.knowledge_manager.add_knowledge(
        "机器学习是人工智能的一个分支，使计算机能够从数据中学习。",
        metadata={"keywords": ["机器学习", "人工智能", "数据"], "category": "技术"}
    )
    
    # 处理用户查询
    result = await system.process_user_input(
        "什么是Python？",
        user_id="demo_user",
        session_id="demo_session"
    )
    
    print("处理结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 获取系统统计
    stats = await system.get_system_stats()
    print("\n系统统计:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # 清理资源
    await system.cleanup()


if __name__ == "__main__":
    asyncio.run(main())