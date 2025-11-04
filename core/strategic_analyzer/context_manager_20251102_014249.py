#!/usr/bin/env python3
"""
上下文管理器
负责管理对话上下文、会话状态和相关信息

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass, asdict

from ..utils.logger import NormaLogger

@dataclass
class ContextEntry:
    """上下文条目"""
    key: str
    value: Any
    timestamp: datetime
    ttl: Optional[int] = None  # 生存时间（秒）
    access_count: int = 0
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl,
            "access_count": self.access_count
        }

class SessionContext:
    """会话上下文类"""
    
    def __init__(self, session_id: str, max_size: int = 1000):
        self.session_id = session_id
        self.max_size = max_size
        self.contexts: Dict[str, ContextEntry] = {}
        self.access_order: deque = deque()  # 访问顺序，用于LRU
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置上下文值"""
        
        # 检查是否需要清理过期条目
        self._cleanup_expired()
        
        # 如果已存在，更新访问顺序
        if key in self.contexts:
            self.access_order.remove(key)
        
        # 创建新的上下文条目
        entry = ContextEntry(
            key=key,
            value=value,
            timestamp=datetime.now(),
            ttl=ttl
        )
        
        self.contexts[key] = entry
        self.access_order.append(key)
        
        # 如果超过最大大小，移除最久未使用的条目
        if len(self.contexts) > self.max_size:
            self._evict_lru()
        
        self.last_accessed = datetime.now()
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取上下文值"""
        
        if key not in self.contexts:
            return default
        
        entry = self.contexts[key]
        
        # 检查是否过期
        if entry.is_expired():
            self.delete(key)
            return default
        
        # 更新访问统计
        entry.access_count += 1
        self.access_order.remove(key)
        self.access_order.append(key)
        self.last_accessed = datetime.now()
        
        return entry.value
    
    def delete(self, key: str) -> bool:
        """删除上下文条目"""
        if key in self.contexts:
            del self.contexts[key]
            if key in self.access_order:
                self.access_order.remove(key)
            return True
        return False
    
    def has(self, key: str) -> bool:
        """检查键是否存在"""
        if key not in self.contexts:
            return False
        
        entry = self.contexts[key]
        if entry.is_expired():
            self.delete(key)
            return False
        
        return True
    
    def keys(self) -> List[str]:
        """获取所有键"""
        self._cleanup_expired()
        return list(self.contexts.keys())
    
    def items(self) -> List[tuple]:
        """获取所有键值对"""
        self._cleanup_expired()
        return [(key, entry.value) for key, entry in self.contexts.items() if not entry.is_expired()]
    
    def clear(self) -> None:
        """清空所有上下文"""
        self.contexts.clear()
        self.access_order.clear()
    
    def size(self) -> int:
        """获取当前大小"""
        self._cleanup_expired()
        return len(self.contexts)
    
    def _cleanup_expired(self) -> None:
        """清理过期的条目"""
        expired_keys = [
            key for key, entry in self.contexts.items() 
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            self.delete(key)
    
    def _evict_lru(self) -> None:
        """移除最久未使用的条目"""
        if not self.access_order:
            return
        
        # 移除最久未使用的条目
        while self.access_order and len(self.contexts) >= self.max_size:
            lru_key = self.access_order.popleft()
            if lru_key in self.contexts:
                del self.contexts[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        self._cleanup_expired()
        
        total_access = sum(entry.access_count for entry in self.contexts.values())
        
        return {
            "session_id": self.session_id,
            "size": len(self.contexts),
            "max_size": self.max_size,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "total_accesses": total_access,
            "average_accesses": total_access / len(self.contexts) if self.contexts else 0
        }

class ContextManager:
    """上下文管理器"""
    
    def __init__(self, max_context_length: int = 50):
        """初始化上下文管理器
        
        Args:
            max_context_length: 最大上下文长度
        """
        self.max_context_length = max_context_length
        self.logger = NormaLogger("context_manager")
        
        # 会话上下文存储
        self.session_contexts: Dict[str, SessionContext] = {}
        
        # 全局上下文
        self.global_context = SessionContext("global", max_size=200)
        
        # 上下文模板
        self.context_templates = {
            "system_status": {
                "cpu_usage": None,
                "memory_usage": None,
                "network_status": None,
                "security_level": None,
                "last_update": None
            },
            "user_profile": {
                "user_id": None,
                "permissions": [],
                "preferences": {},
                "session_history": []
            },
            "conversation_state": {
                "current_topic": None,
                "message_count": 0,
                "last_intent": None,
                "context_stack": []
            }
        }
        
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """初始化上下文管理器"""
        try:
            self.logger.info("初始化上下文管理器...")
            
            # 初始化全局上下文
            await self._initialize_global_context()
            
            self.is_initialized = True
            self.logger.info("上下文管理器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"上下文管理器初始化失败: {e}")
            return False
    
    async def _initialize_global_context(self) -> None:
        """初始化全局上下文"""
        
        # 设置系统信息
        self.global_context.set(
            "system_info",
            {
                "name": "诺玛·劳恩斯",
                "version": "1.0.0",
                "type": "卡塞尔学院主控计算机AI系统",
                "created": "1990",
                "capabilities": [
                    "系统状态监控",
                    "网络安全扫描", 
                    "龙族血统分析",
                    "安全检查",
                    "多模态感知"
                ]
            }
        )
        
        # 设置默认用户权限
        self.global_context.set(
            "default_permissions",
            [
                "system_status_query",
                "blood_analysis",
                "security_check",
                "multimodal_access"
            ]
        )
        
        # 设置上下文模板
        for template_name, template_data in self.context_templates.items():
            self.global_context.set(f"template_{template_name}", template_data)
    
    async def create_session_context(self, session_id: str) -> SessionContext:
        """创建会话上下文"""
        
        if session_id in self.session_contexts:
            return self.session_contexts[session_id]
        
        session_context = SessionContext(session_id, self.max_context_length)
        self.session_contexts[session_id] = session_context
        
        # 初始化会话上下文
        await self._initialize_session_context(session_context)
        
        self.logger.info(f"创建会话上下文: {session_id}")
        return session_context
    
    async def _initialize_session_context(self, session_context: SessionContext) -> None:
        """初始化会话上下文"""
        
        session_id = session_context.session_id
        
        # 设置会话基本信息
        session_context.set(
            "session_info",
            {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "status": "active"
            }
        )
        
        # 设置对话状态
        session_context.set(
            "conversation_state",
            {
                "current_topic": None,
                "message_count": 0,
                "last_intent": None,
                "context_stack": []
            }
        )
        
        # 设置用户信息
        session_context.set(
            "user_info",
            {
                "user_id": None,
                "permissions": self.global_context.get("default_permissions", []),
                "preferences": {},
                "session_count": 1
            }
        )
    
    def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """获取会话上下文"""
        return self.session_contexts.get(session_id)
    
    async def set_context(
        self,
        session_id: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """设置上下文值"""
        
        try:
            # 获取或创建会话上下文
            session_context = self.get_session_context(session_id)
            if not session_context:
                session_context = await self.create_session_context(session_id)
            
            # 设置上下文值
            session_context.set(key, value, ttl)
            
            self.logger.debug(f"设置上下文 {session_id}:{key}")
            return True
            
        except Exception as e:
            self.logger.error(f"设置上下文失败 {session_id}:{key} - {e}")
            return False
    
    async def get_context(
        self,
        session_id: str,
        key: str,
        default: Any = None
    ) -> Any:
        """获取上下文值"""
        
        try:
            session_context = self.get_session_context(session_id)
            if not session_context:
                return default
            
            value = session_context.get(key, default)
            self.logger.debug(f"获取上下文 {session_id}:{key}")
            return value
            
        except Exception as e:
            self.logger.error(f"获取上下文失败 {session_id}:{key} - {e}")
            return default
    
    async def delete_context(self, session_id: str, key: str) -> bool:
        """删除上下文值"""
        
        try:
            session_context = self.get_session_context(session_id)
            if not session_context:
                return False
            
            result = session_context.delete(key)
            self.logger.debug(f"删除上下文 {session_id}:{key}")
            return result
            
        except Exception as e:
            self.logger.error(f"删除上下文失败 {session_id}:{key} - {e}")
            return False
    
    async def clear_session_context(self, session_id: str) -> bool:
        """清空会话上下文"""
        
        try:
            if session_id in self.session_contexts:
                self.session_contexts[session_id].clear()
                del self.session_contexts[session_id]
                
                self.logger.info(f"清空会话上下文: {session_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"清空会话上下文失败 {session_id}:{e}")
            return False
    
    async def update_conversation_state(
        self,
        session_id: str,
        message_count: Optional[int] = None,
        current_topic: Optional[str] = None,
        last_intent: Optional[str] = None,
        context_update: Optional[Dict[str, Any]] = None
    ) -> bool:
        """更新对话状态"""
        
        try:
            conversation_state = await self.get_context(
                session_id,
                "conversation_state",
                {}
            )
            
            if message_count is not None:
                conversation_state["message_count"] = message_count
            
            if current_topic is not None:
                conversation_state["current_topic"] = current_topic
            
            if last_intent is not None:
                conversation_state["last_intent"] = last_intent
            
            if context_update:
                conversation_state.update(context_update)
            
            # 更新上下文
            await self.set_context(session_id, "conversation_state", conversation_state)
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新对话状态失败 {session_id}:{e}")
            return False
    
    async def get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """获取完整的对话上下文"""
        
        try:
            session_context = self.get_session_context(session_id)
            if not session_context:
                return {}
            
            # 获取会话信息
            session_info = session_context.get("session_info", {})
            
            # 获取对话状态
            conversation_state = session_context.get("conversation_state", {})
            
            # 获取用户信息
            user_info = session_context.get("user_info", {})
            
            # 获取最近的上下文历史
            recent_contexts = {}
            for key, value in session_context.items():
                if key.startswith("recent_") or key in ["conversation_state", "user_info", "session_info"]:
                    recent_contexts[key] = value
            
            return {
                "session_id": session_id,
                "session_info": session_info,
                "conversation_state": conversation_state,
                "user_info": user_info,
                "recent_contexts": recent_contexts,
                "context_count": session_context.size(),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取对话上下文失败 {session_id}:{e}")
            return {}
    
    async def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """获取上下文摘要"""
        
        try:
            session_context = self.get_session_context(session_id)
            if not session_context:
                return {"session_id": session_id, "status": "not_found"}
            
            conversation_state = session_context.get("conversation_state", {})
            user_info = session_context.get("user_info", {})
            
            return {
                "session_id": session_id,
                "status": "active",
                "message_count": conversation_state.get("message_count", 0),
                "current_topic": conversation_state.get("current_topic"),
                "last_intent": conversation_state.get("last_intent"),
                "user_permissions": user_info.get("permissions", []),
                "context_size": session_context.size(),
                "created_at": session_context.created_at.isoformat(),
                "last_accessed": session_context.last_accessed.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取上下文摘要失败 {session_id}:{e}")
            return {"session_id": session_id, "status": "error", "error": str(e)}
    
    async def get_all_sessions_summary(self) -> Dict[str, Any]:
        """获取所有会话摘要"""
        
        try:
            summaries = []
            total_contexts = 0
            
            for session_id, session_context in self.session_contexts.items():
                summary = await self.get_context_summary(session_id)
                summaries.append(summary)
                total_contexts += session_context.size()
            
            return {
                "total_sessions": len(self.session_contexts),
                "total_contexts": total_contexts,
                "sessions": summaries,
                "global_context_size": self.global_context.size(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取所有会话摘要失败: {e}")
            return {"error": str(e)}
    
    async def cleanup_expired_contexts(self) -> int:
        """清理过期的上下文"""
        
        cleaned_count = 0
        
        try:
            # 清理会话上下文
            for session_id, session_context in list(self.session_contexts.items()):
                before_size = session_context.size()
                session_context._cleanup_expired()
                after_size = session_context.size()
                
                cleaned = before_size - after_size
                if cleaned > 0:
                    cleaned_count += cleaned
                    self.logger.info(f"清理会话 {session_id} 过期上下文: {cleaned} 条")
            
            # 清理全局上下文
            before_size = self.global_context.size()
            self.global_context._cleanup_expired()
            after_size = self.global_context.size()
            
            cleaned = before_size - after_size
            if cleaned > 0:
                cleaned_count += cleaned
                self.logger.info(f"清理全局过期上下文: {cleaned} 条")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"清理过期上下文失败: {e}")
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        
        try:
            active_sessions = len(self.session_contexts)
            total_contexts = sum(ctx.size() for ctx in self.session_contexts.values())
            
            return {
                "component": "context_manager",
                "status": "healthy" if self.is_initialized else "stopped",
                "active_sessions": active_sessions,
                "total_contexts": total_contexts,
                "global_context_size": self.global_context.size(),
                "max_context_length": self.max_context_length,
                "initialized": self.is_initialized
            }
            
        except Exception as e:
            return {
                "component": "context_manager",
                "status": "error",
                "error": str(e)
            }