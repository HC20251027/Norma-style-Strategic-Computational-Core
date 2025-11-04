#!/usr/bin/env python3
"""
诺玛Agent主入口和初始化系统
负责整个对话系统的初始化、配置管理和生命周期控制

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator
from pathlib import Path

from .conversation_engine import ConversationEngine
from .context_manager import ContextManager
from .memory_manager import MemoryManager
from .agui_integration import AGUIIntegration
from .event_system import EventSystem
from .multimodal_interface import MultimodalInterface
from ..utils.logger import NormaLogger

class NormaAgent:
    """诺玛Agent主类 - 核心对话能力控制器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化诺玛Agent
        
        Args:
            config_path: 配置文件路径
        """
        self.agent_id = str(uuid.uuid4())
        self.version = "1.0.0"
        self.start_time = datetime.now()
        
        # 初始化日志系统
        self.logger = NormaLogger("norma_agent")
        
        # 核心组件
        self.conversation_engine: Optional[ConversationEngine] = None
        self.context_manager: Optional[ContextManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.agui_integration: Optional[AGUIIntegration] = None
        self.event_system: Optional[EventSystem] = None
        self.multimodal_interface: Optional[MultimodalInterface] = None
        
        # 状态管理
        self.is_initialized = False
        self.is_running = False
        self.current_session_id: Optional[str] = None
        
        # 配置
        self.config = self._load_config(config_path)
        
        # 会话管理
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"诺玛Agent {self.version} 正在初始化...")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            "agent": {
                "name": "诺玛·劳恩斯",
                "version": "1.0.0",
                "personality": "卡塞尔学院主控计算机AI系统",
                "max_conversation_length": 100,
                "context_window": 50
            },
            "features": {
                "multimodal": True,
                "agui_protocol": True,
                "real_time_events": True,
                "memory_management": True,
                "context_awareness": True
            },
            "performance": {
                "max_concurrent_sessions": 10,
                "response_timeout": 30,
                "event_buffer_size": 1000
            },
            "logging": {
                "level": "INFO",
                "enable_file_logging": True,
                "log_file_path": "/workspace/norma_agent/data/agent.log"
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 合并配置
                    self._deep_merge(default_config, user_config)
            except Exception as e:
                self.logger.warning(f"配置文件加载失败，使用默认配置: {e}")
        
        return default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """深度合并配置字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    async def initialize(self) -> bool:
        """初始化所有核心组件"""
        try:
            self.logger.info("开始初始化诺玛Agent核心组件...")
            
            # 1. 初始化事件系统
            self.event_system = EventSystem()
            await self.event_system.initialize()
            self.logger.info("✓ 事件系统初始化完成")
            
            # 2. 初始化上下文管理器
            self.context_manager = ContextManager(
                max_context_length=self.config["agent"]["context_window"]
            )
            await self.context_manager.initialize()
            self.logger.info("✓ 上下文管理器初始化完成")
            
            # 3. 初始化记忆管理器
            self.memory_manager = MemoryManager(
                conversation_limit=self.config["agent"]["max_conversation_length"]
            )
            await self.memory_manager.initialize()
            self.logger.info("✓ 记忆管理器初始化完成")
            
            # 4. 初始化对话引擎
            self.conversation_engine = ConversationEngine(
                context_manager=self.context_manager,
                memory_manager=self.memory_manager,
                event_system=self.event_system
            )
            await self.conversation_engine.initialize()
            self.logger.info("✓ 对话引擎初始化完成")
            
            # 5. 初始化AG-UI集成
            if self.config["features"]["agui_protocol"]:
                self.agui_integration = AGUIIntegration(
                    agent=self,
                    event_system=self.event_system
                )
                await self.agui_integration.initialize()
                self.logger.info("✓ AG-UI集成初始化完成")
            
            # 6. 初始化多模态接口
            if self.config["features"]["multimodal"]:
                self.multimodal_interface = MultimodalInterface(
                    agent=self,
                    event_system=self.event_system
                )
                await self.multimodal_interface.initialize()
                self.logger.info("✓ 多模态接口初始化完成")
            
            # 7. 发送系统启动事件
            await self.event_system.publish_event(
                "system.start",
                {
                    "agent_id": self.agent_id,
                    "version": self.version,
                    "start_time": self.start_time.isoformat(),
                    "features": list(self.config["features"].keys())
                }
            )
            
            self.is_initialized = True
            self.logger.info("诺玛Agent初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"诺玛Agent初始化失败: {e}")
            return False
    
    async def start(self) -> bool:
        """启动诺玛Agent"""
        if not self.is_initialized:
            self.logger.error("Agent尚未初始化，请先调用initialize()")
            return False
        
        try:
            self.logger.info("启动诺玛Agent...")
            
            # 启动各个组件
            if self.conversation_engine:
                await self.conversation_engine.start()
            
            if self.event_system:
                await self.event_system.start()
            
            if self.agui_integration:
                await self.agui_integration.start()
            
            if self.multimodal_interface:
                await self.multimodal_interface.start()
            
            self.is_running = True
            
            # 发送系统就绪事件
            await self.event_system.publish_event(
                "system.ready",
                {
                    "agent_id": self.agent_id,
                    "status": "running",
                    "active_sessions": len(self.active_sessions),
                    "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
                }
            )
            
            self.logger.info("诺玛Agent启动完成")
            return True
            
        except Exception as e:
            self.logger.error(f"诺玛Agent启动失败: {e}")
            return False
    
    async def stop(self) -> bool:
        """停止诺玛Agent"""
        if not self.is_running:
            self.logger.warning("Agent未在运行状态")
            return True
        
        try:
            self.logger.info("停止诺玛Agent...")
            
            # 关闭所有活跃会话
            await self._close_all_sessions()
            
            # 停止各个组件
            if self.multimodal_interface:
                await self.multimodal_interface.stop()
            
            if self.agui_integration:
                await self.agui_integration.stop()
            
            if self.conversation_engine:
                await self.conversation_engine.stop()
            
            if self.event_system:
                await self.event_system.stop()
            
            self.is_running = False
            
            # 发送系统停止事件
            await self.event_system.publish_event(
                "system.stop",
                {
                    "agent_id": self.agent_id,
                    "stop_time": datetime.now().isoformat(),
                    "total_uptime": (datetime.now() - self.start_time).total_seconds()
                }
            )
            
            self.logger.info("诺玛Agent已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"诺玛Agent停止失败: {e}")
            return False
    
    async def create_session(self, session_config: Optional[Dict[str, Any]] = None) -> str:
        """创建新的对话会话"""
        session_id = str(uuid.uuid4())
        
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "config": session_config or {},
            "message_count": 0,
            "last_activity": datetime.now().isoformat()
        }
        
        self.active_sessions[session_id] = session_data
        self.current_session_id = session_id
        
        # 初始化会话上下文
        if self.context_manager:
            await self.context_manager.create_session_context(session_id)
        
        # 发送会话创建事件
        await self.event_system.publish_event(
            "session.created",
            {
                "session_id": session_id,
                "session_config": session_config
            }
        )
        
        self.logger.info(f"创建新会话: {session_id}")
        return session_id
    
    async def close_session(self, session_id: str) -> bool:
        """关闭指定会话"""
        if session_id not in self.active_sessions:
            self.logger.warning(f"会话不存在: {session_id}")
            return False
        
        try:
            # 清理会话资源
            if self.context_manager:
                await self.context_manager.clear_session_context(session_id)
            
            if self.memory_manager:
                await self.memory_manager.archive_session(session_id)
            
            # 更新会话状态
            self.active_sessions[session_id]["status"] = "closed"
            self.active_sessions[session_id]["closed_at"] = datetime.now().isoformat()
            
            # 从活跃会话中移除
            del self.active_sessions[session_id]
            
            # 如果关闭的是当前会话，清除当前会话ID
            if self.current_session_id == session_id:
                self.current_session_id = None
            
            # 发送会话关闭事件
            await self.event_system.publish_event(
                "session.closed",
                {"session_id": session_id}
            )
            
            self.logger.info(f"会话已关闭: {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"关闭会话失败 {session_id}: {e}")
            return False
    
    async def _close_all_sessions(self) -> None:
        """关闭所有活跃会话"""
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.close_session(session_id)
    
    async def process_message(
        self, 
        message: str, 
        session_id: Optional[str] = None,
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """处理用户消息并生成回复"""
        
        # 确保有活跃会话
        if not session_id:
            if not self.current_session_id:
                session_id = await self.create_session()
            else:
                session_id = self.current_session_id
        elif session_id not in self.active_sessions:
            await self.create_session({"restored": True})
            session_id = self.current_session_id
        
        # 更新会话活动时间
        self.active_sessions[session_id]["last_activity"] = datetime.now().isoformat()
        self.active_sessions[session_id]["message_count"] += 1
        
        try:
            # 通过对话引擎处理消息
            if self.conversation_engine:
                async for response_chunk in self.conversation_engine.process_message(
                    message=message,
                    session_id=session_id,
                    message_type=message_type,
                    metadata=metadata
                ):
                    yield response_chunk
            else:
                yield "抱歉，对话引擎尚未初始化。"
                
        except Exception as e:
            self.logger.error(f"消息处理失败: {e}")
            yield f"处理消息时发生错误: {str(e)}"
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        status = {
            "agent_id": self.agent_id,
            "version": self.version,
            "status": "running" if self.is_running else "stopped",
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_duration(uptime),
            "initialized": self.is_initialized,
            "current_session": self.current_session_id,
            "active_sessions": len(self.active_sessions),
            "components": {
                "conversation_engine": self.conversation_engine is not None,
                "context_manager": self.context_manager is not None,
                "memory_manager": self.memory_manager is not None,
                "agui_integration": self.agui_integration is not None,
                "event_system": self.event_system is not None,
                "multimodal_interface": self.multimodal_interface is not None
            },
            "features": self.config["features"],
            "start_time": self.start_time.isoformat()
        }
        
        return status
    
    def _format_duration(self, seconds: float) -> str:
        """格式化持续时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}小时")
        if minutes > 0:
            parts.append(f"{minutes}分钟")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}秒")
        
        return "".join(parts)
    
    def get_session_info(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """获取会话信息"""
        if session_id:
            return self.active_sessions.get(session_id, {})
        else:
            return {
                "current_session": self.current_session_id,
                "active_sessions": list(self.active_sessions.keys()),
                "total_active": len(self.active_sessions)
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # 检查各个组件
        components = [
            ("conversation_engine", self.conversation_engine),
            ("context_manager", self.context_manager),
            ("memory_manager", self.memory_manager),
            ("agui_integration", self.agui_integration),
            ("event_system", self.event_system),
            ("multimodal_interface", self.multimodal_interface)
        ]
        
        for component_name, component in components:
            try:
                if component:
                    if hasattr(component, 'health_check'):
                        component_health = await component.health_check()
                        health_status["components"][component_name] = component_health
                    else:
                        health_status["components"][component_name] = {"status": "ok"}
                else:
                    health_status["components"][component_name] = {"status": "not_initialized"}
            except Exception as e:
                health_status["components"][component_name] = {
                    "status": "error",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
        
        return health_status

# 全局诺玛Agent实例
_global_agent: Optional[NormaAgent] = None

async def get_agent() -> NormaAgent:
    """获取全局诺玛Agent实例"""
    global _global_agent
    if _global_agent is None:
        _global_agent = NormaAgent()
        await _global_agent.initialize()
        await _global_agent.start()
    return _global_agent

async def shutdown_agent():
    """关闭全局诺玛Agent实例"""
    global _global_agent
    if _global_agent:
        await _global_agent.stop()
        _global_agent = None