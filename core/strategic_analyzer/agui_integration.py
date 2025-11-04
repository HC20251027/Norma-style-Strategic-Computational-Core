#!/usr/bin/env python3
"""
AG-UI集成模块
实现AG-UI协议支持，包括事件系统、流式通信和工具注册

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable
from enum import Enum
from dataclasses import dataclass, asdict

from ..utils.logger import NormaLogger

class AGUIEventType(Enum):
    """AG-UI事件类型枚举"""
    # 系统事件
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    SYSTEM_ERROR = "system.error"
    SYSTEM_STATUS = "system.status"
    
    # 运行事件
    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    RUN_ERROR = "run.error"
    
    # 工具事件
    TOOL_CALL_STARTED = "tool.call_started"
    TOOL_CALL_FINISHED = "tool.call_finished"
    TOOL_CALL_ERROR = "tool.call_error"
    
    # 消息事件
    MESSAGE_RECEIVED = "message.received"
    MESSAGE_SENT = "message.sent"
    MESSAGE_STREAM = "message.stream"
    
    # 对话事件
    CONVERSATION_START = "conversation.start"
    CONVERSATION_END = "conversation.end"
    CONVERSATION_ERROR = "conversation.error"
    
    # 状态事件
    STATUS_UPDATE = "status.update"
    PROGRESS_UPDATE = "progress.update"
    HEARTBEAT = "heartbeat"

class AGUIEvent:
    """AG-UI事件类"""
    
    def __init__(
        self,
        event_type: AGUIEventType,
        run_id: str,
        data: Dict[str, Any],
        source: str = "norma_agent",
        timestamp: Optional[datetime] = None
    ):
        self.id = str(uuid.uuid4())
        self.type = event_type
        self.run_id = run_id
        self.data = data
        self.source = source
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type.value,
            "run_id": self.run_id,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(',', ':'))

class AGUITool:
    """AG-UI工具类"""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.call_count = 0
        self.last_called = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "call_count": self.call_count,
            "last_called": self.last_called.isoformat() if self.last_called else None
        }

class AGUIStreamManager:
    """AG-UI流式管理器"""
    
    def __init__(self):
        self.active_streams: Dict[str, Dict[str, Any]] = {}
        self.stream_handlers: Dict[str, Callable] = {}
        self.logger = NormaLogger("agui_stream")
    
    async def create_stream(
        self,
        run_id: str,
        stream_type: str = "default"
    ) -> str:
        """创建流"""
        
        stream_id = str(uuid.uuid4())
        
        self.active_streams[stream_id] = {
            "stream_id": stream_id,
            "run_id": run_id,
            "type": stream_type,
            "created_at": datetime.now(),
            "status": "active",
            "event_count": 0
        }
        
        self.logger.info(f"创建AG-UI流: {stream_id}")
        return stream_id
    
    async def send_event(
        self,
        stream_id: str,
        event: AGUIEvent
    ) -> bool:
        """发送事件到流"""
        
        try:
            if stream_id not in self.active_streams:
                self.logger.warning(f"流不存在: {stream_id}")
                return False
            
            # 更新流统计
            self.active_streams[stream_id]["event_count"] += 1
            self.active_streams[stream_id]["last_event"] = event
            
            # 发送给处理器
            if stream_id in self.stream_handlers:
                handler = self.stream_handlers[stream_id]
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            
            self.logger.debug(f"发送事件到流 {stream_id}: {event.type.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"发送事件失败: {e}")
            return False
    
    async def close_stream(self, stream_id: str) -> bool:
        """关闭流"""
        
        try:
            if stream_id in self.active_streams:
                self.active_streams[stream_id]["status"] = "closed"
                self.active_streams[stream_id]["closed_at"] = datetime.now()
                
                # 移除处理器
                if stream_id in self.stream_handlers:
                    del self.stream_handlers[stream_id]
                
                self.logger.info(f"关闭AG-UI流: {stream_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"关闭流失败: {e}")
            return False
    
    def register_stream_handler(self, stream_id: str, handler: Callable) -> None:
        """注册流处理器"""
        self.stream_handlers[stream_id] = handler
    
    def get_stream_stats(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取流统计"""
        return self.active_streams.get(stream_id)

class AGUIIntegration:
    """AG-UI集成类"""
    
    def __init__(self, agent, event_system):
        """初始化AG-UI集成
        
        Args:
            agent: 诺玛Agent实例
            event_system: 事件系统
        """
        self.agent = agent
        self.event_system = event_system
        self.logger = NormaLogger("agui_integration")
        
        # 核心组件
        self.stream_manager = AGUIStreamManager()
        self.tools: Dict[str, AGUITool] = {}
        
        # 状态
        self.is_initialized = False
        self.is_running = False
        self.run_counter = 0
        
        # 配置
        self.config = {
            "enable_streaming": True,
            "enable_tools": True,
            "enable_heartbeat": True,
            "heartbeat_interval": 30,  # 秒
            "max_concurrent_runs": 10
        }
        
        # 活跃运行
        self.active_runs: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> bool:
        """初始化AG-UI集成"""
        try:
            self.logger.info("初始化AG-UI集成...")
            
            # 注册默认工具
            await self._register_default_tools()
            
            # 设置事件监听
            await self._setup_event_listeners()
            
            self.is_initialized = True
            self.logger.info("AG-UI集成初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"AG-UI集成初始化失败: {e}")
            return False
    
    async def _register_default_tools(self) -> None:
        """注册默认工具"""
        
        # 系统状态工具
        self.register_tool(
            name="get_system_status",
            description="获取诺玛系统状态信息",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            handler=self._handle_get_system_status
        )
        
        # 网络扫描工具
        self.register_tool(
            name="scan_network",
            description="网络安全扫描（仅限学院内部）",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            handler=self._handle_scan_network
        )
        
        # 血统分析工具
        self.register_tool(
            name="dragon_blood_analysis",
            description="龙族血统分析",
            parameters={
                "type": "object",
                "properties": {
                    "student_name": {
                        "type": "string",
                        "description": "学生姓名（可选）"
                    }
                },
                "required": []
            },
            handler=self._handle_dragon_blood_analysis
        )
        
        # 安全检查工具
        self.register_tool(
            name="security_check",
            description="系统安全检查",
            parameters={
                "type": "object",
                "properties": {},
                "required": []
            },
            handler=self._handle_security_check
        )
        
        # 系统日志工具
        self.register_tool(
            name="get_system_logs",
            description="获取系统日志",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "日志条数限制",
                        "default": 10
                    }
                },
                "required": []
            },
            handler=self._handle_get_system_logs
        )
        
        # 多模态感知工具
        self.register_tool(
            name="multimodal_perception",
            description="多模态感知操作",
            parameters={
                "type": "object",
                "properties": {
                    "input_type": {
                        "type": "string",
                        "enum": ["image", "audio", "video", "text"],
                        "description": "输入类型"
                    },
                    "input_data": {
                        "type": "string",
                        "description": "输入数据（URL或base64）"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["object_detection", "face_recognition", "ocr", "emotion_analysis", "scene_detection"],
                        "description": "分析类型"
                    }
                },
                "required": ["input_type", "input_data"]
            },
            handler=self._handle_multimodal_perception
        )
        
        self.logger.info(f"注册了 {len(self.tools)} 个AG-UI工具")
    
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable
    ) -> None:
        """注册工具"""
        
        tool = AGUITool(name, description, parameters, handler)
        self.tools[name] = tool
        
        self.logger.info(f"注册AG-UI工具: {name}")
    
    async def _setup_event_listeners(self) -> None:
        """设置事件监听器"""
        
        # 监听系统事件
        await self.event_system.add_event_listener("system.*", self._handle_system_event)
        
        # 监听对话事件
        await self.event_system.add_event_listener("conversation.*", self._handle_conversation_event)
        
        # 监听工具事件
        await self.event_system.add_event_listener("tool.*", self._handle_tool_event)
    
    async def start(self) -> bool:
        """启动AG-UI集成"""
        if not self.is_initialized:
            self.logger.error("AG-UI集成尚未初始化")
            return False
        
        try:
            self.logger.info("启动AG-UI集成...")
            
            # 启动心跳
            if self.config["enable_heartbeat"]:
                asyncio.create_task(self._heartbeat_loop())
            
            self.is_running = True
            self.logger.info("AG-UI集成启动完成")
            return True
            
        except Exception as e:
            self.logger.error(f"AG-UI集成启动失败: {e}")
            return False
    
    async def stop(self) -> bool:
        """停止AG-UI集成"""
        if not self.is_running:
            return True
        
        try:
            self.logger.info("停止AG-UI集成...")
            
            # 关闭所有活跃流
            for stream_id in list(self.stream_manager.active_streams.keys()):
                await self.stream_manager.close_stream(stream_id)
            
            self.is_running = False
            self.logger.info("AG-UI集成已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"AG-UI集成停止失败: {e}")
            return False
    
    async def create_run(self, request_data: Dict[str, Any]) -> str:
        """创建新的运行"""
        
        self.run_counter += 1
        run_id = f"run_{self.run_counter}_{int(datetime.now().timestamp())}"
        
        self.active_runs[run_id] = {
            "run_id": run_id,
            "created_at": datetime.now(),
            "status": "created",
            "request_data": request_data,
            "events": []
        }
        
        self.logger.info(f"创建AG-UI运行: {run_id}")
        return run_id
    
    async def process_run(
        self,
        run_id: str,
        request_data: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """处理运行请求"""
        
        try:
            # 更新运行状态
            if run_id in self.active_runs:
                self.active_runs[run_id]["status"] = "running"
                self.active_runs[run_id]["started_at"] = datetime.now()
            
            # 创建流
            stream_id = await self.stream_manager.create_stream(run_id)
            
            # 发送运行开始事件
            start_event = AGUIEvent(
                AGUIEventType.RUN_STARTED,
                run_id,
                {
                    "message": request_data.get("message", ""),
                    "message_type": request_data.get("message_type", "text"),
                    "timestamp": datetime.now().isoformat()
                }
            )
            await self.stream_manager.send_event(stream_id, start_event)
            
            # 处理消息
            message = request_data.get("message", "")
            message_type = request_data.get("message_type", "text")
            
            async for response_chunk in self.agent.process_message(
                message=message,
                session_id=request_data.get("session_id"),
                message_type=message_type,
                metadata=request_data.get("metadata", {})
            ):
                # 发送消息流事件
                message_event = AGUIEvent(
                    AGUIEventType.MESSAGE_STREAM,
                    run_id,
                    {
                        "content": response_chunk,
                        "chunk_index": len(response_chunk),
                        "timestamp": datetime.now().isoformat()
                    }
                )
                await self.stream_manager.send_event(stream_id, message_event)
                
                yield response_chunk
            
            # 发送运行完成事件
            finish_event = AGUIEvent(
                AGUIEventType.RUN_FINISHED,
                run_id,
                {
                    "status": "completed",
                    "duration": "计算中...",
                    "timestamp": datetime.now().isoformat()
                }
            )
            await self.stream_manager.send_event(stream_id, finish_event)
            
            # 更新运行状态
            if run_id in self.active_runs:
                self.active_runs[run_id]["status"] = "completed"
                self.active_runs[run_id]["completed_at"] = datetime.now()
            
            # 关闭流
            await self.stream_manager.close_stream(stream_id)
            
        except Exception as e:
            self.logger.error(f"处理运行失败 {run_id}: {e}")
            
            # 发送错误事件
            error_event = AGUIEvent(
                AGUIEventType.RUN_ERROR,
                run_id,
                {
                    "error_code": "PROCESSING_ERROR",
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # 尝试发送到流
            if run_id in self.active_runs:
                stream_id = list(self.stream_manager.active_streams.keys())[0]  # 简化处理
                await self.stream_manager.send_event(stream_id, error_event)
            
            yield f"处理请求时发生错误: {str(e)}"
    
    async def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        run_id: str
    ) -> Dict[str, Any]:
        """调用工具"""
        
        try:
            if tool_name not in self.tools:
                raise ValueError(f"工具不存在: {tool_name}")
            
            tool = self.tools[tool_name]
            
            # 发送工具调用开始事件
            start_event = AGUIEvent(
                AGUIEventType.TOOL_CALL_STARTED,
                run_id,
                {
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # 更新工具统计
            tool.call_count += 1
            tool.last_called = datetime.now()
            
            # 调用工具处理器
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(parameters)
            else:
                result = tool.handler(parameters)
            
            # 发送工具调用完成事件
            finish_event = AGUIEvent(
                AGUIEventType.TOOL_CALL_FINISHED,
                run_id,
                {
                    "tool_name": tool_name,
                    "result": result,
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            self.logger.info(f"工具调用成功: {tool_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"工具调用失败 {tool_name}: {e}")
            
            # 发送工具调用错误事件
            error_event = AGUIEvent(
                AGUIEventType.TOOL_CALL_ERROR,
                run_id,
                {
                    "tool_name": tool_name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            raise
    
    def get_tools_registry(self) -> Dict[str, Any]:
        """获取工具注册信息"""
        
        return {
            "tools": [tool.to_dict() for tool in self.tools.values()],
            "agent_info": {
                "name": "诺玛·劳恩斯",
                "version": "1.0.0",
                "description": "卡塞尔学院主控计算机AI系统",
                "capabilities": [
                    "系统状态监控",
                    "网络安全扫描",
                    "龙族血统分析",
                    "安全检查",
                    "多模态感知",
                    "AG-UI协议支持"
                ]
            }
        }
    
    def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """获取运行状态"""
        return self.active_runs.get(run_id)
    
    def get_all_runs_status(self) -> Dict[str, Any]:
        """获取所有运行状态"""
        
        return {
            "total_runs": len(self.active_runs),
            "active_runs": len([r for r in self.active_runs.values() if r["status"] == "running"]),
            "runs": list(self.active_runs.values())
        }
    
    # 工具处理器方法
    async def _handle_get_system_status(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """处理系统状态查询"""
        return await self.agent.get_system_status()
    
    async def _handle_scan_network(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """处理网络扫描"""
        # 模拟网络扫描结果
        return {
            "scan_type": "内部网络安全性扫描",
            "target_range": "192.168.1.0/24 (学院内部网络)",
            "scan_results": {
                "active_hosts": 15,
                "open_ports": [22, 80, 443, 8080],
                "security_status": "正常",
                "vulnerabilities": "无高危漏洞发现"
            },
            "recommendations": [
                "建议定期更新系统补丁",
                "建议加强SSH访问控制",
                "建议启用防火墙规则"
            ],
            "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    async def _handle_dragon_blood_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """处理血统分析"""
        student_name = parameters.get("student_name")
        
        if student_name:
            return {
                "student_name": student_name,
                "bloodline_type": "S级混血种",
                "purity_level": "95.2%",
                "abilities": "黄金瞳、言灵·君焰、时间零",
                "status": "稳定",
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            return {
                "total_registered_students": 4,
                "bloodline_distribution": {
                    "S级混血种": 1,
                    "A级混血种": 3
                },
                "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    async def _handle_security_check(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """处理安全检查"""
        return {
            "firewall_status": "已启用",
            "antivirus_status": "运行中",
            "intrusion_detection": "正常监控",
            "failed_login_attempts": 0,
            "suspicious_activities": "无",
            "last_security_scan": "2025-10-31 10:30:00",
            "security_score": 95,
            "recommendations": [
                "系统安全状态良好",
                "建议继续保持当前安全配置"
            ]
        }
    
    async def _handle_get_system_logs(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """处理系统日志查询"""
        limit = parameters.get("limit", 10)
        
        # 模拟系统日志
        logs = []
        for i in range(min(limit, 5)):
            logs.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "level": "INFO",
                "module": "system",
                "message": f"系统日志条目 {i+1}",
                "details": f"详细信息 {i+1}"
            })
        
        return {
            "log_count": len(logs),
            "logs": logs,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    async def _handle_multimodal_perception(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """处理多模态感知"""
        input_type = parameters.get("input_type")
        analysis_type = parameters.get("analysis_type")
        
        return {
            "input_type": input_type,
            "analysis_type": analysis_type,
            "status": "processing",
            "result": {
                "confidence": 0.85,
                "description": f"{analysis_type}分析结果",
                "details": "多模态分析完成"
            },
            "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    # 事件处理器
    async def _handle_system_event(self, event_data: Dict[str, Any]) -> None:
        """处理系统事件"""
        self.logger.debug(f"处理系统事件: {event_data}")
    
    async def _handle_conversation_event(self, event_data: Dict[str, Any]) -> None:
        """处理对话事件"""
        self.logger.debug(f"处理对话事件: {event_data}")
    
    async def _handle_tool_event(self, event_data: Dict[str, Any]) -> None:
        """处理工具事件"""
        self.logger.debug(f"处理工具事件: {event_data}")
    
    async def _heartbeat_loop(self) -> None:
        """心跳循环"""
        while self.is_running:
            try:
                # 发送心跳事件
                heartbeat_event = AGUIEvent(
                    AGUIEventType.HEARTBEAT,
                    "system",
                    {
                        "agent_id": self.agent.agent_id,
                        "timestamp": datetime.now().isoformat(),
                        "active_runs": len(self.active_runs)
                    }
                )
                
                # 发送到所有活跃流
                for stream_id in self.stream_manager.active_streams:
                    await self.stream_manager.send_event(stream_id, heartbeat_event)
                
                await asyncio.sleep(self.config["heartbeat_interval"])
                
            except Exception as e:
                self.logger.error(f"心跳循环错误: {e}")
                await asyncio.sleep(self.config["heartbeat_interval"])
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        
        return {
            "component": "agui_integration",
            "status": "healthy" if self.is_running else "stopped",
            "initialized": self.is_initialized,
            "tools_count": len(self.tools),
            "active_runs": len(self.active_runs),
            "active_streams": len(self.stream_manager.active_streams),
            "config": self.config
        }