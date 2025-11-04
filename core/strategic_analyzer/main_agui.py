#!/usr/bin/env python3
"""
诺玛·劳恩斯AI系统 - FastAPI后端服务（AG-UI集成版本）
集成AG-UI协议，提供标准化的Agent用户交互接口

作者: 皇
创建时间: 2025-10-31
AG-UI集成: 2025-10-31
"""

import os
import sys
import json
import asyncio
import sqlite3
import psutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Generator, AsyncGenerator
from contextlib import asynccontextmanager
from enum import Enum
from dataclasses import dataclass
from functools import wraps
from collections import defaultdict
from time import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field
import uvicorn

# =============================================================================
# 生产环境配置
# =============================================================================

class ProductionConfig:
    """生产环境配置类"""
    
    # 环境设置
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # 服务器配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8002"))
    
    # 安全配置
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
    
    # API密钥配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # 数据库配置
    DATABASE_PATH = os.getenv("DATABASE_PATH", "/workspace/data/norma_database.db")
    LOGS_PATH = os.getenv("LOGS_PATH", "/workspace/data/norma_logs.db")
    
    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO" if not DEBUG else "DEBUG")
    LOG_FILE = os.getenv("LOG_FILE", "/workspace/logs/norma_production.log")
    
    # 限流配置
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = os.getenv("RATE_LIMIT_WINDOW", "1 minute")
    
    # 安全配置
    FORCE_HTTPS = os.getenv("FORCE_HTTPS", "false").lower() == "true"
    TRUSTED_HOSTS = os.getenv("TRUSTED_HOSTS", "*").split(",")
    
    # WebSocket配置
    WS_MAX_CONNECTIONS = int(os.getenv("WS_MAX_CONNECTIONS", "1000"))
    WS_TIMEOUT = int(os.getenv("WS_TIMEOUT", "300"))

# 初始化配置
config = ProductionConfig()

# =============================================================================
# 日志配置
# =============================================================================

def setup_logging():
    """配置日志系统"""
    # 创建日志目录
    Path("/workspace/logs").mkdir(parents=True, exist_ok=True)
    
    # 配置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 创建日志记录器
    logger = logging.getLogger("norma_ai")
    logger.setLevel(getattr(logging, config.LOG_LEVEL))
    
    # 文件处理器
    file_handler = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    file_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, config.LOG_LEVEL))
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化日志
logger = setup_logging()

# =============================================================================
# 安全装饰器
# =============================================================================

def rate_limit_key_func(request: Request) -> str:
    """限流键函数"""
    # 获取客户端IP
    try:
        client_ip = get_remote_address(request)
    except:
        # 如果slowapi不可用，使用备用方法
        client_ip = request.client.host if request.client else "unknown"
    
    # 如果有API密钥，使用API密钥作为限流键
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"
    
    return f"ip:{client_ip}"

def rate_limit(requests: int = None, window: str = None):
    """限流装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not config.RATE_LIMIT_ENABLED:
                return await func(*args, **kwargs)
            # 限流逻辑由SlowAPI中间件处理
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# =============================================================================
# 连接管理器（增强版）
# =============================================================================

class EnhancedConnectionManager:
    """增强版连接管理器"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.system_status_subscribers: List[WebSocket] = []
        self.chat_subscribers: List[WebSocket] = []
        self.connection_stats = defaultdict(int)
        self.connection_times = defaultdict(float)
    
    async def connect(self, websocket: WebSocket, connection_type: str = "general"):
        """连接WebSocket"""
        try:
            # 检查连接数限制
            if len(self.active_connections) >= config.WS_MAX_CONNECTIONS:
                await websocket.close(code=1013, reason="服务器连接数已满")
                logger.warning(f"WebSocket连接被拒绝: 连接数已达上限 ({config.WS_MAX_CONNECTIONS})")
                return
            
            await websocket.accept()
            
            if connection_type == "system_status":
                self.system_status_subscribers.append(websocket)
            elif connection_type == "chat":
                self.chat_subscribers.append(websocket)
            else:
                self.active_connections.append(websocket)
            
            # 记录连接统计
            self.connection_stats[connection_type] += 1
            self.connection_times[websocket] = time()
            
            logger.info(f"WebSocket连接建立: {connection_type}, 当前连接数: {len(self.active_connections)}")
        except Exception as e:
            logger.error(f"WebSocket连接失败: {e}")
            try:
                await websocket.close(code=1011, reason="连接建立失败")
            except:
                pass
    
    def disconnect(self, websocket: WebSocket):
        """断开WebSocket连接"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.system_status_subscribers:
            self.system_status_subscribers.remove(websocket)
        if websocket in self.chat_subscribers:
            self.chat_subscribers.remove(websocket)
        
        # 清理连接时间记录
        if websocket in self.connection_times:
            del self.connection_times[websocket]
        
        logger.info(f"WebSocket连接断开, 当前连接数: {len(self.active_connections)}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计"""
        return {
            "total_connections": len(self.active_connections),
            "system_status_subscribers": len(self.system_status_subscribers),
            "chat_subscribers": len(self.chat_subscribers),
            "connection_stats": dict(self.connection_stats),
            "max_connections": config.WS_MAX_CONNECTIONS
        }

# 添加代码路径
sys.path.append('/workspace/code')
sys.path.append('/workspace')

# 尝试导入诺玛模块
NormaCoreAgent = None
NormaAdvancedFeaturesClass = None

try:
    from norma_core_agent import NormaCoreAgent, create_demo_data
    print("诺玛核心模块导入成功")
except ImportError as e:
    print(f"警告: 诺玛核心模块导入失败: {e}")

try:
    # 创建一个简化的高级功能类
    class SimplifiedAdvancedFeatures:
        def demo_search_functionality(self, query: str):
            return {
                "search_query": query,
                "search_engine": "DuckDuckGo (模拟)",
                "results": [
                    {
                        "title": f"关于 '{query}' 的研究资料",
                        "url": "https://example.com/research",
                        "snippet": "这是关于该主题的详细研究内容...",
                        "reliability": "高"
                    }
                ],
                "total_results": 1,
                "search_time": "0.3秒",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        def demo_multi_agent_collaboration(self):
            return {
                "agents": [
                    {
                        "name": "搜索专家",
                        "status": "活跃",
                        "tasks_completed": 156,
                        "accuracy": 94.5,
                        "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    },
                    {
                        "name": "文档专家",
                        "status": "活跃",
                        "tasks_completed": 89,
                        "accuracy": 97.2,
                        "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    },
                    {
                        "name": "知识专家",
                        "status": "活跃",
                        "tasks_completed": 203,
                        "accuracy": 96.8,
                        "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                ],
                "collaboration_mode": "协调模式",
                "total_tasks": 448,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    NormaAdvancedFeaturesClass = SimplifiedAdvancedFeatures
    print("简化高级功能模块创建成功")
except ImportError as e:
    print(f"警告: 诺玛高级功能模块导入失败: {e}")

# =============================================================================
# AG-UI 协议相关定义
# =============================================================================

class EventType(str, Enum):
    """AG-UI事件类型枚举"""
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_ERROR = "run_error"
    STEP_STARTED = "step_started"
    STEP_FINISHED = "step_finished"
    TEXT_MESSAGE_START = "text_message_start"
    TEXT_MESSAGE_CONTENT = "text_message_content"
    TEXT_MESSAGE_END = "text_message_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_ARGS = "tool_call_args"
    TOOL_CALL_END = "tool_call_end"
    TOOL_CALL_RESULT = "tool_call_result"
    STATE_SNAPSHOT = "state_snapshot"
    STATE_DELTA = "state_delta"
    MESSAGES_SNAPSHOT = "messages_snapshot"
    RAW = "raw"
    CUSTOM = "custom"

class MessageRole(str, Enum):
    """消息角色枚举"""
    DEVELOPER = "developer"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"

# AG-UI 基础事件模型
class BaseEvent(BaseModel):
    """AG-UI基础事件模型"""
    type: EventType
    timestamp: datetime = Field(default_factory=datetime.now)
    raw_event: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# 生命周期事件
class RunStartedEvent(BaseEvent):
    type: EventType = EventType.RUN_STARTED
    run_id: str

class RunFinishedEvent(BaseEvent):
    type: EventType = EventType.RUN_FINISHED
    run_id: str
    result: Optional[Any] = None

class RunErrorEvent(BaseEvent):
    type: EventType = EventType.RUN_ERROR
    run_id: str
    error: str
    details: Optional[Dict[str, Any]] = None

class StepStartedEvent(BaseEvent):
    type: EventType = EventType.STEP_STARTED
    run_id: str
    step_id: str
    step_name: str

class StepFinishedEvent(BaseEvent):
    type: EventType = EventType.STEP_FINISHED
    run_id: str
    step_id: str
    step_name: str
    result: Optional[Any] = None

# 文本消息事件
class TextMessageStartEvent(BaseEvent):
    type: EventType = EventType.TEXT_MESSAGE_START
    run_id: str
    message_id: str
    role: MessageRole

class TextMessageContentEvent(BaseEvent):
    type: EventType = EventType.TEXT_MESSAGE_CONTENT
    run_id: str
    message_id: str
    content: str

class TextMessageEndEvent(BaseEvent):
    type: EventType = EventType.TEXT_MESSAGE_END
    run_id: str
    message_id: str
    role: MessageRole

# 工具调用事件
class ToolCallStartEvent(BaseEvent):
    type: EventType = EventType.TOOL_CALL_START
    run_id: str
    tool_call_id: str
    tool_name: str

class ToolCallArgsEvent(BaseEvent):
    type: EventType = EventType.TOOL_CALL_ARGS
    run_id: str
    tool_call_id: str
    args: Dict[str, Any]

class ToolCallEndEvent(BaseEvent):
    type: EventType = EventType.TOOL_CALL_END
    run_id: str
    tool_call_id: str
    tool_name: str

class ToolCallResultEvent(BaseEvent):
    type: EventType = EventType.TOOL_CALL_RESULT
    run_id: str
    tool_call_id: str
    result: Any

# 状态管理事件
class StateSnapshotEvent(BaseEvent):
    type: EventType = EventType.STATE_SNAPSHOT
    run_id: str
    state: Dict[str, Any]

class StateDeltaEvent(BaseEvent):
    type: EventType = EventType.STATE_DELTA
    run_id: str
    delta: Dict[str, Any]

class MessagesSnapshotEvent(BaseEvent):
    type: EventType = EventType.MESSAGES_SNAPSHOT
    run_id: str
    messages: List[Dict[str, Any]]

# 特殊事件
class RawEvent(BaseEvent):
    type: EventType = EventType.RAW
    data: Any

class CustomEvent(BaseEvent):
    type: EventType = EventType.CUSTOM
    name: str
    data: Any

# 联合类型定义
AGUIEvent = Union[
    RunStartedEvent, RunFinishedEvent, RunErrorEvent,
    StepStartedEvent, StepFinishedEvent,
    TextMessageStartEvent, TextMessageContentEvent, TextMessageEndEvent,
    ToolCallStartEvent, ToolCallArgsEvent, ToolCallEndEvent, ToolCallResultEvent,
    StateSnapshotEvent, StateDeltaEvent, MessagesSnapshotEvent,
    RawEvent, CustomEvent
]

# AG-UI 输入模型
class Message(BaseModel):
    """AG-UI消息模型"""
    role: MessageRole
    content: str
    id: Optional[str] = None

class FunctionCall(BaseModel):
    """函数调用模型"""
    name: str
    arguments: str

class ToolCall(BaseModel):
    """工具调用模型"""
    id: str
    type: str = "function"
    function: FunctionCall

class Context(BaseModel):
    """上下文模型"""
    description: str
    value: Any

class Tool(BaseModel):
    """工具模型"""
    name: str
    description: str
    parameters: Dict[str, Any]

class RunAgentInput(BaseModel):
    """AG-UI代理运行输入模型"""
    thread_id: str
    run_id: str
    state: Optional[Any] = None
    messages: List[Message] = Field(default_factory=list)
    tools: List[Tool] = Field(default_factory=list)
    context: List[Context] = Field(default_factory=list)
    forwarded_props: Optional[Any] = None

# 事件编码器
class EventEncoder:
    """AG-UI事件编码器"""
    
    def __init__(self, accept: Optional[str] = None):
        self.accept = accept
    
    def encode(self, event: AGUIEvent) -> str:
        """将事件编码为SSE格式"""
        try:
            event_dict = event.dict()
            # 确保时间戳格式正确
            if 'timestamp' in event_dict and isinstance(event_dict['timestamp'], datetime):
                event_dict['timestamp'] = event_dict['timestamp'].isoformat()
            return f"data: {json.dumps(event_dict, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"事件编码失败: {e}")
            # 返回错误事件
            error_event = {
                "type": "run_error",
                "timestamp": datetime.now().isoformat(),
                "error": f"事件编码失败: {str(e)}"
            }
            return f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

# =============================================================================
# 原有数据模型（保持兼容性）
# =============================================================================

class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None

class SystemStatusResponse(BaseModel):
    system_name: str
    version: str
    uptime: str
    status: str
    cpu_usage: str
    memory_usage: str
    memory_available: str
    network_status: str
    security_level: str
    timestamp: str

class BloodAnalysisRequest(BaseModel):
    student_name: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    type: Optional[str] = "general"

# =============================================================================
# WebSocket连接管理器（保持原有功能）
# =============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.system_status_subscribers: List[WebSocket] = []
        self.chat_subscribers: List[WebSocket] = []

    async def connect(self, websocket: WebSocket, connection_type: str = "general"):
        await websocket.accept()
        if connection_type == "system_status":
            self.system_status_subscribers.append(websocket)
        elif connection_type == "chat":
            self.chat_subscribers.append(websocket)
        else:
            self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.system_status_subscribers:
            self.system_status_subscribers.remove(websocket)
        if websocket in self.chat_subscribers:
            self.chat_subscribers.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """发送个人消息"""
        try:
            await websocket.send_text(message)
        except WebSocketDisconnect:
            logger.info("WebSocket连接已断开，移除连接")
            self.disconnect(websocket)
        except Exception as e:
            logger.error(f"发送WebSocket消息失败: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str, connection_type: str = "general"):
        """广播消息"""
        if connection_type == "system_status":
            connections = self.system_status_subscribers
        elif connection_type == "chat":
            connections = self.chat_subscribers
        else:
            connections = self.active_connections
        
        disconnected_connections = []
        for connection in connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                disconnected_connections.append(connection)
            except Exception as e:
                logger.error(f"广播消息失败: {e}")
                disconnected_connections.append(connection)
        
        # 清理断开的连接
        for connection in disconnected_connections:
            self.disconnect(connection)

manager = ConnectionManager()

# =============================================================================
# 诺玛系统服务类（扩展AG-UI支持）
# =============================================================================

class NormaSystemService:
    def __init__(self):
        self.core_agent = NormaCoreAgent() if NormaCoreAgent else None
        self.advanced_features = NormaAdvancedFeaturesClass() if NormaAdvancedFeaturesClass else None
        self.db_path = config.DATABASE_PATH
        self.logs_path = config.LOGS_PATH
        self.ensure_directories()
        self.initialize_data()
        
        logger.info(f"诺玛系统服务初始化完成 - 数据库路径: {self.db_path}")

    def ensure_directories(self):
        """确保必要目录存在"""
        directories = [
            "/workspace/data",
            "/workspace/data/knowledge_base",
            "/workspace/data/vector_db",
            "/workspace/data/dragon_blood",
            "/workspace/data/security_logs",
            "/workspace/backend/uploads",
            "/workspace/logs"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info("目录结构检查完成")

    def initialize_data(self):
        """初始化演示数据"""
        if NormaCoreAgent:
            try:
                create_demo_data()
                logger.info("演示数据初始化成功")
            except Exception as e:
                logger.error(f"演示数据初始化失败: {e}")
        else:
            logger.warning("诺玛核心模块未加载，使用模拟数据")

    # =============================================================================
    # AG-UI 代理运行方法
    # =============================================================================
    
    async def run_agent_agui(self, input_data: RunAgentInput) -> AsyncGenerator[AGUIEvent, None]:
        """AG-UI代理运行方法，返回事件流"""
        run_id = input_data.run_id
        
        try:
            logger.info(f"开始AG-UI代理运行: run_id={run_id}, thread_id={input_data.thread_id}")
            
            # 1. 发送运行开始事件
            yield RunStartedEvent(run_id=run_id)
            
            # 2. 发送状态快照
            current_state = {
                "thread_id": input_data.thread_id,
                "messages_count": len(input_data.messages),
                "tools_available": len(input_data.tools),
                "context_count": len(input_data.context)
            }
            yield StateSnapshotEvent(run_id=run_id, state=current_state)
            
            # 3. 发送消息快照
            messages_snapshot = [
                {"role": msg.role.value, "content": msg.content, "id": msg.id}
                for msg in input_data.messages
            ]
            yield MessagesSnapshotEvent(run_id=run_id, messages=messages_snapshot)
            
            # 4. 处理用户消息
            if input_data.messages:
                latest_message = input_data.messages[-1]
                if latest_message.role == MessageRole.USER:
                    message_id = latest_message.id or f"msg_{run_id}_{len(input_data.messages)}"
                    
                    # 文本消息开始
                    yield TextMessageStartEvent(
                        run_id=run_id,
                        message_id=message_id,
                        role=latest_message.role
                    )
                    
                    # 处理消息内容
                    content = latest_message.content
                    
                    # 模拟诺玛系统处理
                    if self.core_agent:
                        try:
                            response = await self.core_agent.process_user_query(content)
                        except Exception as e:
                            response = f"处理消息时发生错误: {str(e)}"
                    else:
                        response = f"诺玛系统回复: 您发送的消息是 '{content}'。当前系统正在初始化中，请稍候再试。"
                    
                    # 发送文本内容
                    yield TextMessageContentEvent(
                        run_id=run_id,
                        message_id=message_id,
                        content=response
                    )
                    
                    # 文本消息结束
                    yield TextMessageEndEvent(
                        run_id=run_id,
                        message_id=message_id,
                        role=MessageRole.ASSISTANT
                    )
            
            # 5. 模拟工具调用（如果有工具）
            if input_data.tools:
                for tool in input_data.tools[:2]:  # 限制工具调用数量
                    tool_call_id = f"tool_{run_id}_{tool.name}"
                    
                    # 工具调用开始
                    yield ToolCallStartEvent(
                        run_id=run_id,
                        tool_call_id=tool_call_id,
                        tool_name=tool.name
                    )
                    
                    # 工具参数
                    yield ToolCallArgsEvent(
                        run_id=run_id,
                        tool_call_id=tool_call_id,
                        args=tool.parameters
                    )
                    
                    # 模拟工具执行
                    await asyncio.sleep(0.5)  # 模拟执行时间
                    
                    # 工具调用结束
                    yield ToolCallEndEvent(
                        run_id=run_id,
                        tool_call_id=tool_call_id,
                        tool_name=tool.name
                    )
                    
                    # 工具结果
                    tool_result = {
                        "status": "success",
                        "message": f"工具 {tool.name} 执行完成",
                        "timestamp": datetime.now().isoformat()
                    }
                    yield ToolCallResultEvent(
                        run_id=run_id,
                        tool_call_id=tool_call_id,
                        result=tool_result
                    )
            
            # 6. 发送最终状态
            final_state = {
                "status": "completed",
                "messages_processed": len(input_data.messages),
                "tools_executed": len(input_data.tools),
                "timestamp": datetime.now().isoformat()
            }
            yield StateDeltaEvent(run_id=run_id, delta=final_state)
            
            # 7. 发送运行完成事件
            yield RunFinishedEvent(
                run_id=run_id,
                result={
                    "status": "success",
                    "response": "诺玛系统处理完成",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            logger.info(f"AG-UI代理运行完成: run_id={run_id}")
            
        except Exception as e:
            # 发送错误事件
            logger.error(f"AG-UI代理运行错误: run_id={run_id}, error={e}", exc_info=True)
            yield RunErrorEvent(
                run_id=run_id,
                error=str(e),
                details={"type": type(e).__name__, "module": __name__}
            )

    async def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        try:
            status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "3.1.2-agui-production",
                "environment": config.ENVIRONMENT,
                "agui_enabled": True,
                "websocket_enabled": True,
                "security_config": {
                    "cors_enabled": True,
                    "rate_limiting": config.RATE_LIMIT_ENABLED,
                    "https_redirect": config.FORCE_HTTPS
                },
                "system_info": {
                    "cpu_usage": f"{psutil.cpu_percent()}%",
                    "memory_usage": f"{psutil.virtual_memory().percent}%",
                    "disk_usage": f"{psutil.disk_usage('/').percent}%"
                },
                "connection_stats": manager.get_connection_stats() if 'manager' in globals() else {}
            }
            
            logger.debug("健康状态检查完成")
            return status
        except Exception as e:
            logger.error(f"健康状态检查失败: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # =============================================================================
    # 原有方法（保持兼容性）
    # =============================================================================
    
    async def get_system_status(self) -> SystemStatusResponse:
        """获取系统状态"""
        try:
            if self.core_agent:
                info = self.core_agent.tools.get_system_info()
                return SystemStatusResponse(**info)
            else:
                # 模拟数据
                return SystemStatusResponse(
                    system_name="诺玛·劳恩斯",
                    version="3.1.2-agui (1990-2025)",
                    uptime="35年3个月",
                    status="正常运行",
                    cpu_usage="15.3%",
                    memory_usage="42.7%",
                    memory_available="32GB",
                    network_status="活跃",
                    security_level="高",
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"系统状态获取失败: {str(e)}")

    async def process_chat_message(self, message: str) -> str:
        """处理聊天消息"""
        try:
            if self.core_agent:
                return await self.core_agent.process_user_query(message)
            else:
                return f"诺玛系统回复: 您发送的消息是 '{message}'。当前系统正在初始化中，请稍候再试。"
        except Exception as e:
            return f"处理消息时发生错误: {str(e)}"

    async def get_blood_analysis(self, student_name: Optional[str] = None) -> Dict[str, Any]:
        """获取血统分析"""
        try:
            if self.core_agent:
                return self.core_agent.tools.dragon_blood_analysis(student_name)
            else:
                # 模拟数据
                if student_name:
                    return {
                        "student_name": student_name,
                        "bloodline_type": "A级混血种",
                        "purity_level": "87.3%",
                        "abilities": "黄金瞳、言灵·君焰",
                        "status": "稳定",
                        "last_analysis": datetime.now().strftime("%Y-%m-%d")
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
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"血统分析失败: {str(e)}")

    async def get_system_logs(self, limit: int = 10) -> Dict[str, Any]:
        """获取系统日志"""
        try:
            if self.core_agent:
                return self.core_agent.tools.get_system_logs(limit)
            else:
                # 模拟数据
                logs = []
                for i in range(limit):
                    logs.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "level": ["INFO", "WARNING", "ERROR"][i % 3],
                        "module": ["系统", "网络", "安全"][i % 3],
                        "message": f"示例日志消息 {i+1}",
                        "details": f"详细描述 {i+1}"
                    })
                return {
                    "log_count": len(logs),
                    "logs": logs,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"系统日志获取失败: {str(e)}")

    async def search_knowledge_base(self, query: str) -> Dict[str, Any]:
        """搜索知识库"""
        try:
            if self.advanced_features:
                return self.advanced_features.demo_search_functionality(query)
            else:
                # 模拟搜索结果
                return {
                    "search_query": query,
                    "search_engine": "DuckDuckGo (模拟)",
                    "results": [
                        {
                            "title": f"关于 '{query}' 的研究资料",
                            "url": "https://example.com/research",
                            "snippet": "这是关于该主题的详细研究内容...",
                            "reliability": "高"
                        }
                    ],
                    "total_results": 1,
                    "search_time": "0.3秒",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"知识库搜索失败: {str(e)}")

    async def get_security_status(self) -> Dict[str, Any]:
        """获取安全状态"""
        try:
            if self.core_agent:
                return self.core_agent.tools.security_check()
            else:
                # 模拟数据
                return {
                    "firewall_status": "已启用",
                    "antivirus_status": "运行中",
                    "intrusion_detection": "正常监控",
                    "failed_login_attempts": 0,
                    "suspicious_activities": "无",
                    "last_security_scan": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "security_score": 95,
                    "recommendations": [
                        "系统安全状态良好",
                        "建议继续保持当前安全配置"
                    ]
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"安全状态获取失败: {str(e)}")

    async def get_multi_agent_status(self) -> Dict[str, Any]:
        """获取多智能体状态"""
        try:
            if self.advanced_features:
                return self.advanced_features.demo_multi_agent_collaboration()
            else:
                # 模拟数据
                return {
                    "agents": [
                        {
                            "name": "搜索专家",
                            "status": "活跃",
                            "tasks_completed": 156,
                            "accuracy": 94.5,
                            "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        },
                        {
                            "name": "文档专家",
                            "status": "活跃",
                            "tasks_completed": 89,
                            "accuracy": 97.2,
                            "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        },
                        {
                            "name": "知识专家",
                            "status": "活跃",
                            "tasks_completed": 203,
                            "accuracy": 96.8,
                            "last_activity": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    ],
                    "collaboration_mode": "协调模式",
                    "total_tasks": 448,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"多智能体状态获取失败: {str(e)}")

# =============================================================================
# 全局异常处理器
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    logger.warning(f"HTTP异常: {exc.status_code} - {exc.detail}")
    return {
        "error": {
            "code": f"HTTP_{exc.status_code}",
            "message": str(exc.detail),
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    }

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return {
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "服务器内部错误" if config.ENVIRONMENT == "production" else str(exc),
            "type": type(exc).__name__,
            "timestamp": datetime.now().isoformat()
        }
    }

# =============================================================================
# 全局服务实例
# =============================================================================

# 初始化限流器
try:
    limiter = Limiter(key_func=rate_limit_key_func)
    RATE_LIMIT_AVAILABLE = True
except Exception as e:
    logger.warning(f"限流器初始化失败: {e}，将禁用限流功能")
    limiter = None
    RATE_LIMIT_AVAILABLE = False

# 创建全局服务实例
norma_service = NormaSystemService()

# 创建增强版连接管理器
manager = EnhancedConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    logger.info("正在启动诺玛·劳恩斯AI系统后端服务（AG-UI集成版本）...")
    logger.info(f"运行环境: {config.ENVIRONMENT}")
    logger.info(f"调试模式: {config.DEBUG}")
    logger.info(f"服务器地址: {config.HOST}:{config.PORT}")
    
    # 初始化数据库连接
    try:
        Path(config.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"数据库路径: {config.DATABASE_PATH}")
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
    
    yield
    
    # 关闭时执行
    logger.info("诺玛·劳恩斯AI系统后端服务已关闭")

# =============================================================================
# FastAPI应用创建（生产环境配置）
# =============================================================================

app = FastAPI(
    title="诺玛·劳恩斯AI系统API（AG-UI集成版）",
    description="卡塞尔学院主控计算机AI系统后端服务，支持AG-UI协议 - 生产环境",
    version="3.1.2-agui-production",
    lifespan=lifespan,
    debug=config.DEBUG
)

# 添加限流异常处理器
if RATE_LIMIT_AVAILABLE and limiter:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 配置CORS（生产环境安全配置）
cors_origins = config.CORS_ORIGINS if "*" not in config.CORS_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "*",  # 允许所有头部
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Accept",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
    ],
    expose_headers=["*"],
    max_age=86400,  # 24小时
)

# 配置HTTPS重定向（生产环境）
if config.FORCE_HTTPS:
    app.add_middleware(HTTPSRedirectMiddleware)

# 配置信任主机（生产环境安全）
if config.TRUSTED_HOSTS and "*" not in config.TRUSTED_HOSTS:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=config.TRUSTED_HOSTS
    )

# 配置限流中间件
if config.RATE_LIMIT_ENABLED and RATE_LIMIT_AVAILABLE:
    @app.middleware("http")
    async def add_rate_limit_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(config.RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Window"] = config.RATE_LIMIT_WINDOW
        return response

# =============================================================================
# API路由
# =============================================================================

@app.get("/")
async def root():
    """根路径端点"""
    logger.info("根路径访问")
    return {
        "message": "诺玛·劳恩斯AI系统API服务（AG-UI集成版）",
        "version": "3.1.2-agui-production",
        "environment": config.ENVIRONMENT,
        "status": "running",
        "agui_enabled": True,
        "security_features": {
            "cors_enabled": True,
            "rate_limiting": config.RATE_LIMIT_ENABLED,
            "https_redirect": config.FORCE_HTTPS
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# AG-UI 健康检查端点
@app.get("/status")
@limiter.limit(f"{config.RATE_LIMIT_REQUESTS}/{config.RATE_LIMIT_WINDOW}") if RATE_LIMIT_AVAILABLE and limiter else lambda request: None
async def get_health_status(request: Request):
    """AG-UI健康检查端点（带限流）"""
    logger.debug("健康检查请求")
    return await norma_service.get_health_status()

# AG-UI 主入口端点
@app.post("/agui")
@limiter.limit(f"{config.RATE_LIMIT_REQUESTS}/{config.RATE_LIMIT_WINDOW}") if RATE_LIMIT_AVAILABLE and limiter else lambda request: None
async def run_agent_agui(request: Request):
    """AG-UI主入口端点，接收RunAgentInput并流式传输事件（带限流）"""
    try:
        logger.info(f"AG-UI请求 - 客户端IP: {get_remote_address(request) if RATE_LIMIT_AVAILABLE else 'unknown'}")
        
        # 解析输入数据
        input_data = await request.json()
        run_input = RunAgentInput(**input_data)
        
        # 创建事件编码器
        encoder = EventEncoder()
        
        # 生成事件流
        async def event_generator():
            async for event in norma_service.run_agent_agui(run_input):
                encoded_event = encoder.encode(event)
                yield encoded_event
        
        # 返回流式响应
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用Nginx缓冲
                "Access-Control-Allow-Origin": cors_origins[0] if cors_origins != ["*"] else "*",
                "Access-Control-Allow-Credentials": "true",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS,PATCH",
                "Access-Control-Expose-Headers": "*",
            }
        )
        
    except Exception as e:
        logger.error(f"AG-UI请求处理失败: {e}", exc_info=True)
        error_response = {
            "error": {
                "code": "AGUI_REQUEST_ERROR",
                "message": f"AG-UI请求处理失败: {str(e)}",
                "type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }
        }
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=400, detail=error_response)

# 原有API端点（保持兼容性，添加限流和日志）
@app.get("/api/system/status", response_model=SystemStatusResponse)
@limiter.limit(f"{config.RATE_LIMIT_REQUESTS}/{config.RATE_LIMIT_WINDOW}") if RATE_LIMIT_AVAILABLE and limiter else lambda request: None
async def get_system_status(request: Request):
    """获取系统状态（带限流）"""
    logger.debug("系统状态请求")
    return await norma_service.get_system_status()

@app.post("/api/chat/message")
@limiter.limit(f"{config.RATE_LIMIT_REQUESTS}/{config.RATE_LIMIT_WINDOW}") if RATE_LIMIT_AVAILABLE and limiter else lambda request: None
async def chat_message(request: Request, message: ChatMessage):
    """处理聊天消息（带限流）"""
    logger.info(f"聊天消息: {message.message[:50]}...")
    response = await norma_service.process_chat_message(message.message)
    return {
        "response": response,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/api/bloodline/analyze")
async def analyze_bloodline(request: BloodAnalysisRequest):
    """血统分析"""
    return await norma_service.get_blood_analysis(request.student_name)

@app.get("/api/system/logs")
async def get_logs(limit: int = 10):
    """获取系统日志"""
    return await norma_service.get_system_logs(limit)

@app.post("/api/knowledge/search")
async def search_knowledge(request: SearchRequest):
    """搜索知识库"""
    return await norma_service.search_knowledge_base(request.query)

@app.get("/api/security/status")
async def get_security_status():
    """获取安全状态"""
    return await norma_service.get_security_status()

@app.get("/api/agents/status")
async def get_agents_status():
    """获取多智能体状态"""
    return await norma_service.get_multi_agent_status()

# =============================================================================
# WebSocket端点（增强版）
# =============================================================================

@app.websocket("/ws/system")
async def websocket_system_status(websocket: WebSocket):
    """系统状态WebSocket连接（增强版）"""
    await manager.connect(websocket, "system_status")
    try:
        while True:
            # 定期发送系统状态更新
            status = await norma_service.get_system_status()
            await manager.send_personal_message(
                json.dumps({
                    "type": "system_status",
                    "data": status.dict(),
                    "timestamp": datetime.now().isoformat()
                }),
                websocket
            )
            await asyncio.sleep(5)  # 每5秒更新一次
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"系统状态WebSocket错误: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """聊天WebSocket连接（增强版）"""
    await manager.connect(websocket, "chat")
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "message":
                # 处理聊天消息
                response = await norma_service.process_chat_message(message_data.get("message", ""))
                
                # 发送回复
                await manager.send_personal_message(
                    json.dumps({
                        "type": "response",
                        "message": response,
                        "timestamp": datetime.now().isoformat()
                    }),
                    websocket
                )
                
                # 广播给其他聊天用户
                await manager.broadcast(
                    json.dumps({
                        "type": "chat_message",
                        "user": "用户",
                        "message": message_data.get("message", ""),
                        "timestamp": datetime.now().isoformat()
                    }),
                    "chat"
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"聊天WebSocket错误: {e}")
        manager.disconnect(websocket)

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """日志流WebSocket连接（增强版）"""
    await manager.connect(websocket)
    try:
        while True:
            # 定期发送日志更新
            logs = await norma_service.get_system_logs(5)
            await manager.send_personal_message(
                json.dumps({
                    "type": "logs_update",
                    "data": logs,
                    "timestamp": datetime.now().isoformat()
                }),
                websocket
            )
            await asyncio.sleep(3)  # 每3秒更新一次
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"日志WebSocket错误: {e}")
        manager.disconnect(websocket)

# =============================================================================
# 标准WebSocket端点（AG-UI兼容）
# =============================================================================

@app.websocket("/ws")
async def websocket_agui_standard(websocket: WebSocket):
    """AG-UI标准WebSocket端点"""
    await manager.connect(websocket, "agui_standard")
    try:
        logger.info("AG-UI标准WebSocket连接已建立")
        
        # 发送连接成功事件
        await manager.send_personal_message(
            json.dumps({
                "type": "connection_established",
                "protocol": "AG-UI",
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "message": "AG-UI WebSocket连接成功"
            }),
            websocket
        )
        
        # 处理客户端消息
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                message_type = message.get("type", "unknown")
                
                if message_type == "ping":
                    # 心跳响应
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )
                elif message_type == "get_status":
                    # 发送当前状态
                    status = await norma_service.get_health_status()
                    await manager.send_personal_message(
                        json.dumps({
                            "type": "status_response",
                            "data": status,
                            "timestamp": datetime.now().isoformat()
                        }),
                        websocket
                    )
                elif message_type == "chat_message":
                    # 处理聊天消息
                    user_input = message.get("data", {}).get("message", "")
                    if user_input:
                        # 异步处理用户输入
                        asyncio.create_task(process_chat_message(user_input, websocket))
                else:
                    logger.warning(f"未知的消息类型: {message_type}")
                    
            except json.JSONDecodeError:
                logger.error("WebSocket消息格式错误")
            except Exception as e:
                logger.error(f"处理WebSocket消息错误: {e}")
                
    except WebSocketDisconnect:
        logger.info("AG-UI标准WebSocket连接断开")
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"AG-UI标准WebSocket错误: {e}")
        manager.disconnect(websocket)

async def process_chat_message(user_input: str, websocket: WebSocket):
    """处理聊天消息"""
    try:
        # 发送消息接收确认
        await manager.send_personal_message(
            json.dumps({
                "type": "message_received",
                "data": {"message": user_input},
                "timestamp": datetime.now().isoformat()
            }),
            websocket
        )
        
        # 这里可以集成诺玛AI处理逻辑
        # 暂时返回模拟响应
        response = f"收到消息: {user_input} (处理中...)"
        
        await manager.send_personal_message(
            json.dumps({
                "type": "chat_response",
                "data": {"message": response},
                "timestamp": datetime.now().isoformat()
            }),
            websocket
        )
        
    except Exception as e:
        logger.error(f"处理聊天消息错误: {e}")
        await manager.send_personal_message(
            json.dumps({
                "type": "error",
                "data": {"message": f"处理消息时出错: {str(e)}"},
                "timestamp": datetime.now().isoformat()
            }),
            websocket
        )

# =============================================================================
# 文件上传端点（保持原有功能）
# =============================================================================

@app.post("/api/upload/document")
async def upload_document(file: UploadFile = File(...)):
    """上传文档"""
    try:
        # 保存文件
        file_path = f"/workspace/backend/uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "message": "文件上传成功",
            "filename": file.filename,
            "size": len(content),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

# 静态文件服务
if Path("/workspace/backend/uploads").exists():
    app.mount("/uploads", StaticFiles(directory="/workspace/backend/uploads"), name="uploads")

# =============================================================================
# 应用启动配置
# =============================================================================

if __name__ == "__main__":
    # 配置启动参数
    uvicorn_config = {
        "app": "main_agui:app",
        "host": config.HOST,
        "port": config.PORT,
        "log_level": config.LOG_LEVEL.lower(),
        "access_log": True,
        "loop": "uvloop" if config.ENVIRONMENT == "production" else "auto"
    }
    
    # 生产环境额外配置
    if config.ENVIRONMENT == "production":
        uvicorn_config.update({
            "workers": 1,  # 单worker避免内存问题
            "reload": False,
            "use_colors": False
        })
        logger.info("生产环境启动配置")
    else:
        uvicorn_config.update({
            "reload": config.DEBUG,
            "use_colors": True
        })
        logger.info("开发环境启动配置")
    
    # 启动服务器
    logger.info(f"启动诺玛AI系统服务器: {config.HOST}:{config.PORT}")
    logger.info(f"环境: {config.ENVIRONMENT}, 调试: {config.DEBUG}")
    
    try:
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        logger.info("服务器被用户中断")
    except Exception as e:
        logger.error(f"服务器启动失败: {e}")
        raise