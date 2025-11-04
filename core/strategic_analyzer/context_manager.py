"""
上下文管理器

负责管理工具调用的上下文信息，包括会话状态、工具结果缓存、执行状态等
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict, deque
from datetime import datetime, timedelta
import asyncio

from ..models import ToolcallContext, ToolcallRequest


logger = logging.getLogger(__name__)


class ContextManager:
    """上下文管理器"""
    
    def __init__(self, max_history_size: int = 1000, cache_ttl: int = 3600):
        self.max_history_size = max_history_size
        self.cache_ttl = cache_ttl
        self.contexts: Dict[str, ToolcallContext] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self.context_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.global_state: Dict[str, Any] = {}
        
    async def get_or_create_context(self, request: ToolcallRequest) -> ToolcallContext:
        """获取或创建上下文"""
        async with self._get_session_lock(request.session_id):
            context_id = f"{request.session_id}:{request.request_id}"
            
            # 尝试获取现有上下文
            if context_id in self.contexts:
                context = self.contexts[context_id]
                context.update(**request.context)
                return context
            
            # 创建新上下文
            context = ToolcallContext(
                request_id=request.request_id,
                session_id=request.session_id,
                user_id=request.user_id,
                metadata=request.context.copy()
            )
            
            self.contexts[context_id] = context
            self.context_history[request.session_id].append(context)
            
            # 清理过期上下文
            await self._cleanup_expired_contexts()
            
            logger.info(f"创建新上下文: {context_id}")
            return context
    
    async def update_context(
        self, 
        context: ToolcallContext, 
        **updates
    ) -> ToolcallContext:
        """更新上下文"""
        context_id = f"{context.session_id}:{context.request_id}"
        
        async with self._get_session_lock(context.session_id):
            # 更新上下文
            context.update(**updates)
            
            # 记录更新历史
            self.context_history[context.session_id].append(context)
            
            logger.debug(f"更新上下文: {context_id}")
            return context
    
    async def add_conversation_history(
        self, 
        context: ToolcallContext, 
        message: Dict[str, Any]
    ):
        """添加对话历史"""
        async with self._get_session_lock(context.session_id):
            context.conversation_history.append({
                **message,
                "timestamp": datetime.now().isoformat()
            })
            
            # 保持历史记录在合理范围内
            if len(context.conversation_history) > self.max_history_size:
                context.conversation_history = context.conversation_history[-self.max_history_size//2:]
            
            logger.debug(f"添加对话历史: {len(context.conversation_history)}条记录")
    
    async def cache_tool_result(
        self, 
        context: ToolcallContext, 
        tool_id: str, 
        result: Any,
        cache_key: Optional[str] = None
    ):
        """缓存工具执行结果"""
        async with self._get_session_lock(context.session_id):
            if cache_key is None:
                cache_key = self._generate_cache_key(tool_id, result)
            
            cache_entry = {
                "result": result,
                "timestamp": datetime.now(),
                "tool_id": tool_id
            }
            
            context.tool_results_cache[cache_key] = cache_entry
            
            # 清理过期缓存
            await self._cleanup_expired_cache(context)
            
            logger.debug(f"缓存工具结果: {cache_key}")
    
    async def get_cached_result(
        self, 
        context: ToolcallContext, 
        tool_id: str, 
        cache_key: Optional[str] = None
    ) -> Optional[Any]:
        """获取缓存的工具结果"""
        if cache_key is None:
            return None
        
        cache_entry = context.tool_results_cache.get(cache_key)
        if not cache_entry:
            return None
        
        # 检查是否过期
        if self._is_cache_expired(cache_entry["timestamp"]):
            del context.tool_results_cache[cache_key]
            return None
        
        logger.debug(f"获取缓存结果: {cache_key}")
        return cache_entry["result"]
    
    async def update_execution_state(
        self, 
        context: ToolcallContext, 
        state_updates: Dict[str, Any]
    ):
        """更新执行状态"""
        async with self._get_session_lock(context.session_id):
            context.execution_state.update(state_updates)
            
            logger.debug(f"更新执行状态: {state_updates}")
    
    async def get_execution_state(
        self, 
        context: ToolcallContext, 
        key: str, 
        default: Any = None
    ) -> Any:
        """获取执行状态"""
        return context.execution_state.get(key, default)
    
    async def set_available_tools(
        self, 
        context: ToolcallContext, 
        tools: List[str]
    ):
        """设置可用工具列表"""
        async with self._get_session_lock(context.session_id):
            context.available_tools = tools.copy()
            
            logger.debug(f"设置可用工具: {len(tools)}个")
    
    async def get_available_tools(self, context: ToolcallContext) -> List[str]:
        """获取可用工具列表"""
        return context.available_tools.copy()
    
    async def add_metadata(
        self, 
        context: ToolcallContext, 
        key: str, 
        value: Any
    ):
        """添加元数据"""
        async with self._get_session_lock(context.session_id):
            context.metadata[key] = value
            
            logger.debug(f"添加元数据: {key} = {value}")
    
    async def get_metadata(
        self, 
        context: ToolcallContext, 
        key: str, 
        default: Any = None
    ) -> Any:
        """获取元数据"""
        return context.metadata.get(key, default)
    
    async def get_session_contexts(self, session_id: str) -> List[ToolcallContext]:
        """获取会话的所有上下文"""
        session_contexts = []
        
        for context in self.contexts.values():
            if context.session_id == session_id:
                session_contexts.append(context)
        
        # 按创建时间排序
        session_contexts.sort(key=lambda x: x.created_at)
        
        return session_contexts
    
    async def get_conversation_summary(
        self, 
        context: ToolcallContext, 
        max_messages: int = 10
    ) -> List[Dict[str, Any]]:
        """获取对话摘要"""
        history = context.conversation_history
        
        if len(history) <= max_messages:
            return history
        
        # 返回最近的消息
        return history[-max_messages:]
    
    async def find_similar_contexts(
        self, 
        context: ToolcallContext, 
        threshold: float = 0.7
    ) -> List[ToolcallContext]:
        """查找相似的上下文"""
        similar_contexts = []
        current_metadata = context.metadata
        
        for other_context in self.contexts.values():
            if other_context.request_id == context.request_id:
                continue
            
            similarity = self._calculate_context_similarity(current_metadata, other_context.metadata)
            
            if similarity > threshold:
                similar_contexts.append(other_context)
        
        # 按相似度排序
        similar_contexts.sort(
            key=lambda x: self._calculate_context_similarity(current_metadata, x.metadata), 
            reverse=True
        )
        
        return similar_contexts[:5]  # 返回最相似的5个
    
    async def merge_contexts(
        self, 
        target_context: ToolcallContext, 
        source_context: ToolcallContext
    ) -> ToolcallContext:
        """合并上下文"""
        async with self._get_session_lock(target_context.session_id):
            # 合并对话历史
            target_context.conversation_history.extend(source_context.conversation_history)
            
            # 合并工具结果缓存
            target_context.tool_results_cache.update(source_context.tool_results_cache)
            
            # 合并执行状态
            target_context.execution_state.update(source_context.execution_state)
            
            # 合并元数据
            target_context.metadata.update(source_context.metadata)
            
            # 合并可用工具列表
            all_tools = set(target_context.available_tools) | set(source_context.available_tools)
            target_context.available_tools = list(all_tools)
            
            logger.info(f"合并上下文: {source_context.request_id} -> {target_context.request_id}")
            
            return target_context
    
    async def cleanup_session(self, session_id: str):
        """清理会话数据"""
        async with self._get_session_lock(session_id):
            # 删除该会话的所有上下文
            contexts_to_remove = [
                context_id for context_id, context in self.contexts.items()
                if context.session_id == session_id
            ]
            
            for context_id in contexts_to_remove:
                del self.contexts[context_id]
            
            # 清空历史记录
            if session_id in self.context_history:
                self.context_history[session_id].clear()
            
            logger.info(f"清理会话数据: {session_id}, 删除了{len(contexts_to_remove)}个上下文")
    
    async def get_context_statistics(self) -> Dict[str, Any]:
        """获取上下文统计信息"""
        total_contexts = len(self.contexts)
        session_count = len(set(context.session_id for context in self.contexts.values()))
        
        # 统计各会话的上下文数量
        session_stats = defaultdict(int)
        total_history_messages = 0
        total_cache_entries = 0
        
        for context in self.contexts.values():
            session_stats[context.session_id] += 1
            total_history_messages += len(context.conversation_history)
            total_cache_entries += len(context.tool_results_cache)
        
        return {
            "total_contexts": total_contexts,
            "active_sessions": session_count,
            "session_distribution": dict(session_stats),
            "average_contexts_per_session": total_contexts / max(session_count, 1),
            "total_history_messages": total_history_messages,
            "total_cache_entries": total_cache_entries,
            "average_cache_entries_per_context": total_cache_entries / max(total_contexts, 1)
        }
    
    async def export_context(self, context: ToolcallContext) -> Dict[str, Any]:
        """导出上下文数据"""
        return {
            "context": context.to_dict(),
            "conversation_summary": await self.get_conversation_summary(context),
            "available_tools": context.available_tools,
            "metadata": context.metadata
        }
    
    async def import_context(self, context_data: Dict[str, Any]) -> ToolcallContext:
        """导入上下文数据"""
        context_dict = context_data["context"]
        
        # 重建上下文对象
        context = ToolcallContext(
            request_id=context_dict["request_id"],
            session_id=context_dict["session_id"],
            user_id=context_dict["user_id"],
            conversation_history=context_data.get("conversation_summary", []),
            available_tools=context_data.get("available_tools", []),
            metadata=context_data.get("metadata", {})
        )
        
        # 存储上下文
        context_id = f"{context.session_id}:{context.request_id}"
        self.contexts[context_id] = context
        
        logger.info(f"导入上下文: {context_id}")
        
        return context
    
    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """获取会话锁"""
        if session_id not in self.session_locks:
            self.session_locks[session_id] = asyncio.Lock()
        return self.session_locks[session_id]
    
    def _generate_cache_key(self, tool_id: str, result: Any) -> str:
        """生成缓存键"""
        import hashlib
        
        # 简化的缓存键生成
        content = f"{tool_id}:{str(result)[:100]}"  # 限制长度
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_expired(self, timestamp: datetime) -> bool:
        """检查缓存是否过期"""
        return datetime.now() - timestamp > timedelta(seconds=self.cache_ttl)
    
    async def _cleanup_expired_contexts(self):
        """清理过期的上下文"""
        # 这里可以实现更复杂的过期清理逻辑
        # 目前简化处理
        pass
    
    async def _cleanup_expired_cache(self, context: ToolcallContext):
        """清理过期的缓存"""
        expired_keys = []
        
        for cache_key, cache_entry in context.tool_results_cache.items():
            if self._is_cache_expired(cache_entry["timestamp"]):
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del context.tool_results_cache[key]
        
        if expired_keys:
            logger.debug(f"清理过期缓存: {len(expired_keys)}条")
    
    def _calculate_context_similarity(
        self, 
        metadata1: Dict[str, Any], 
        metadata2: Dict[str, Any]
    ) -> float:
        """计算上下文相似度"""
        if not metadata1 or not metadata2:
            return 0.0
        
        # 简单的Jaccard相似度
        keys1 = set(metadata1.keys())
        keys2 = set(metadata2.keys())
        
        intersection = keys1.intersection(keys2)
        union = keys1.union(keys2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    async def persist_contexts(self, file_path: str):
        """持久化上下文到文件"""
        contexts_data = []
        
        for context in self.contexts.values():
            contexts_data.append(await self.export_context(context))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(contexts_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"持久化上下文到文件: {file_path}")
    
    async def load_contexts(self, file_path: str):
        """从文件加载上下文"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                contexts_data = json.load(f)
            
            for context_data in contexts_data:
                await self.import_context(context_data)
            
            logger.info(f"从文件加载上下文: {file_path}, {len(contexts_data)}个上下文")
            
        except FileNotFoundError:
            logger.warning(f"上下文文件不存在: {file_path}")
        except Exception as e:
            logger.error(f"加载上下文失败: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy",
            "total_contexts": len(self.contexts),
            "active_sessions": len(self.session_locks),
            "memory_usage": {
                "contexts": len(self.contexts),
                "history_entries": sum(len(history) for history in self.context_history.values())
            },
            "cache_stats": {
                "total_cache_entries": sum(
                    len(context.tool_results_cache) 
                    for context in self.contexts.values()
                )
            }
        }