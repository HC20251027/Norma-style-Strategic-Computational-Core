#!/usr/bin/env python3
"""
诺玛AI系统AG-UI配置文件
包含AG-UI协议支持、事件类型定义、工具注册和配置参数

作者: 皇
创建时间: 2025-10-31 12:56:50
版本: 1.0.0
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# AG-UI协议版本
AGUI_PROTOCOL_VERSION = "1.0.0"
AGUI_AGENT_VERSION = "3.1.2"

class AGUIEventType(Enum):
    """AG-UI事件类型枚举"""
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_FINISHED = "tool_call_finished"
    MESSAGE = "message"
    ERROR = "error"
    PROGRESS = "progress"
    STATUS = "status"
    HEARTBEAT = "heartbeat"

class AGUIErrorCode(Enum):
    """AG-UI错误代码枚举"""
    PROCESSING_ERROR = "PROCESSING_ERROR"
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    SYSTEM_ERROR = "SYSTEM_ERROR"

@dataclass
class AGUIEvent:
    """AG-UI基础事件类"""
    type: AGUIEventType
    run_id: str
    timestamp: str
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "type": self.type.value,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "data": self.data
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

@dataclass
class AGUITool:
    """AG-UI工具定义类"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]
    examples: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AGUIAgentInfo:
    """AG-UI智能体信息类"""
    name: str
    version: str
    description: str
    capabilities: List[str]
    supported_protocols: List[str]
    max_concurrent_runs: int = 10
    timeout_seconds: int = 300
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AGUIEventEncoder:
    """AG-UI事件编码器"""
    
    @staticmethod
    def encode(event: AGUIEvent) -> str:
        """编码事件为JSON字符串"""
        return event.to_json() + "\n"
    
    @staticmethod
    def encode_event(type: AGUIEventType, run_id: str, data: Dict[str, Any]) -> str:
        """便捷方法：直接编码事件"""
        event = AGUIEvent(
            type=type,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            data=data
        )
        return AGUIEventEncoder.encode(event)
    
    @staticmethod
    def create_run_started_event(run_id: str, agent_name: str, query: str) -> AGUIEvent:
        """创建运行开始事件"""
        return AGUIEvent(
            type=AGUIEventType.RUN_STARTED,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            data={
                "agent_name": agent_name,
                "query": query,
                "protocol_version": AGUI_PROTOCOL_VERSION,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @staticmethod
    def create_tool_call_event(tool_name: str, parameters: Dict[str, Any], run_id: str) -> AGUIEvent:
        """创建工具调用事件"""
        return AGUIEvent(
            type=AGUIEventType.TOOL_CALL_STARTED,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            data={
                "tool_name": tool_name,
                "parameters": parameters,
                "status": "started"
            }
        )
    
    @staticmethod
    def create_message_event(content: str, role: str = "assistant", run_id: str = "") -> AGUIEvent:
        """创建消息事件"""
        return AGUIEvent(
            type=AGUIEventType.MESSAGE,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            data={
                "content": content,
                "role": role,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @staticmethod
    def create_error_event(error_code: AGUIErrorCode, error_message: str, run_id: str) -> AGUIEvent:
        """创建错误事件"""
        return AGUIEvent(
            type=AGUIEventType.ERROR,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            data={
                "error_code": error_code.value,
                "error_message": error_message,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    @staticmethod
    def create_run_finished_event(run_id: str, status: str = "completed") -> AGUIEvent:
        """创建运行完成事件"""
        return AGUIEvent(
            type=AGUIEventType.RUN_FINISHED,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            data={
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
        )

class NormaAGUIConfig:
    """诺玛AG-UI配置管理器"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.agent_info = self._create_agent_info()
        self.tools = self._create_tools_registry()
        self.streaming_config = self._create_streaming_config()
        self.error_handling_config = self._create_error_handling_config()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("norma_agui_config")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_agent_info(self) -> AGUIAgentInfo:
        """创建智能体信息"""
        return AGUIAgentInfo(
            name="诺玛·劳恩斯",
            version=AGUI_AGENT_VERSION,
            description="卡塞尔学院主控计算机AI系统，支持AG-UI协议",
            capabilities=[
                "Agent-User Interaction Protocol支持",
                "流式事件输出",
                "AG-UI兼容的工具注册",
                "错误处理和日志记录",
                "标准化响应格式",
                "系统状态监控",
                "网络安全扫描",
                "龙族血统分析",
                "安全检查",
                "日志分析"
            ],
            supported_protocols=[
                "AG-UI-1.0",
                "WebSocket",
                "HTTP-SSE",
                "JSON-RPC"
            ],
            max_concurrent_runs=5,
            timeout_seconds=600
        )
    
    def _create_tools_registry(self) -> List[AGUITool]:
        """创建工具注册表"""
        tools = [
            AGUITool(
                name="get_system_info",
                description="获取诺玛系统基本信息，包括CPU、内存、网络状态等",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                required=[],
                examples=[
                    {"description": "获取完整系统状态", "parameters": {}}
                ]
            ),
            AGUITool(
                name="scan_network",
                description="对学院内部网络进行安全扫描（仅限内部网络）",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                required=[],
                examples=[
                    {"description": "执行内部网络安全扫描", "parameters": {}}
                ]
            ),
            AGUITool(
                name="dragon_blood_analysis",
                description="分析学生的龙族血统信息",
                parameters={
                    "type": "object",
                    "properties": {
                        "student_name": {
                            "type": "string",
                            "description": "学生姓名（可选，不指定则返回全体统计）"
                        }
                    },
                    "required": []
                },
                required=[],
                examples=[
                    {"description": "分析特定学生血统", "parameters": {"student_name": "路明非"}},
                    {"description": "获取全体学生血统统计", "parameters": {}}
                ]
            ),
            AGUITool(
                name="security_check",
                description="执行系统安全检查和分析",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                required=[],
                examples=[
                    {"description": "执行完整安全检查", "parameters": {}}
                ]
            ),
            AGUITool(
                name="get_system_logs",
                description="获取系统日志记录",
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "日志条数限制（默认10条）",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 100
                        },
                        "level": {
                            "type": "string",
                            "description": "日志级别过滤",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]
                        }
                    },
                    "required": []
                },
                required=[],
                examples=[
                    {"description": "获取最近10条日志", "parameters": {"limit": 10}},
                    {"description": "获取错误日志", "parameters": {"level": "ERROR", "limit": 20}}
                ]
            ),
            AGUITool(
                name="get_network_connections",
                description="获取当前网络连接状态",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                required=[],
                examples=[
                    {"description": "获取网络连接状态", "parameters": {}}
                ]
            )
        ]
        
        return tools
    
    def _create_streaming_config(self) -> Dict[str, Any]:
        """创建流式配置"""
        return {
            "enabled": True,
            "chunk_size": 200,
            "chunk_delay": 0.05,
            "heartbeat_interval": 30,
            "buffer_size": 1024,
            "compression": False,
            "retry_attempts": 3,
            "retry_delay": 1.0
        }
    
    def _create_error_handling_config(self) -> Dict[str, Any]:
        """创建错误处理配置"""
        return {
            "log_errors": True,
            "return_error_details": True,
            "error_codes": {
                "PROCESSING_ERROR": {
                    "description": "处理请求时发生错误",
                    "retryable": True
                },
                "TOOL_EXECUTION_ERROR": {
                    "description": "工具执行失败",
                    "retryable": False
                },
                "VALIDATION_ERROR": {
                    "description": "请求参数验证失败",
                    "retryable": False
                },
                "TIMEOUT_ERROR": {
                    "description": "请求超时",
                    "retryable": True
                },
                "AUTHENTICATION_ERROR": {
                    "description": "身份验证失败",
                    "retryable": False
                },
                "SYSTEM_ERROR": {
                    "description": "系统内部错误",
                    "retryable": True
                }
            },
            "max_retries": 3,
            "timeout_seconds": 300
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """获取智能体信息"""
        return self.agent_info.to_dict()
    
    def get_tools_registry(self) -> List[Dict[str, Any]]:
        """获取工具注册表"""
        return [tool.to_dict() for tool in self.tools]
    
    def get_streaming_config(self) -> Dict[str, Any]:
        """获取流式配置"""
        return self.streaming_config
    
    def get_error_handling_config(self) -> Dict[str, Any]:
        """获取错误处理配置"""
        return self.error_handling_config
    
    def get_full_config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return {
            "agent_info": self.get_agent_info(),
            "tools": self.get_tools_registry(),
            "streaming": self.get_streaming_config(),
            "error_handling": self.get_error_handling_config(),
            "protocol_version": AGUI_PROTOCOL_VERSION,
            "generated_at": datetime.now().isoformat()
        }
    
    def validate_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """验证AG-UI请求"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # 检查必需字段
        required_fields = ["run_id", "message"]
        for field in required_fields:
            if field not in request:
                validation_result["valid"] = False
                validation_result["errors"].append(f"缺少必需字段: {field}")
        
        # 检查run_id格式
        if "run_id" in request:
            run_id = request["run_id"]
            if not isinstance(run_id, str) or len(run_id) == 0:
                validation_result["valid"] = False
                validation_result["errors"].append("run_id必须是非空字符串")
        
        # 检查message字段
        if "message" in request:
            message = request["message"]
            if not isinstance(message, str):
                validation_result["valid"] = False
                validation_result["errors"].append("message必须是字符串")
            elif len(message) > 10000:
                validation_result["warnings"].append("message较长，可能影响响应时间")
        
        return validation_result
    
    def create_heartbeat_event(self, run_id: str) -> AGUIEvent:
        """创建心跳事件"""
        return AGUIEvent(
            type=AGUIEventType.HEARTBEAT,
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            data={
                "status": "alive",
                "timestamp": datetime.now().isoformat()
            }
        )

# 全局配置实例
agui_config = NormaAGUIConfig()

# 便捷函数
def get_agui_config() -> NormaAGUIConfig:
    """获取AG-UI配置实例"""
    return agui_config

def create_agui_event(type: str, run_id: str, data: Dict[str, Any]) -> str:
    """创建AG-UI事件的便捷函数"""
    event_type = AGUIEventType(type)
    event = AGUIEvent(
        type=event_type,
        run_id=run_id,
        timestamp=datetime.now().isoformat(),
        data=data
    )
    return AGUIEventEncoder.encode(event)

def validate_agui_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """验证AG-UI请求的便捷函数"""
    return agui_config.validate_request(request)

if __name__ == "__main__":
    print("诺玛AI系统AG-UI配置")
    print("="*50)
    
    # 显示完整配置
    config = agui_config.get_full_config()
    
    print(f"Agent名称: {config['agent_info']['name']}")
    print(f"版本: {config['agent_info']['version']}")
    print(f"协议版本: {config['protocol_version']}")
    print(f"工具数量: {len(config['tools'])}")
    print(f"流式支持: {config['streaming']['enabled']}")
    
    print("\n可用工具:")
    for tool in config['tools']:
        print(f"  - {tool['name']}: {tool['description']}")
    
    print("\n配置已生成并可用于AG-UI协议支持")
    
    # 保存配置到文件
    config_file = "/workspace/code/norma_agent_agui_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"\n配置已保存到: {config_file}")