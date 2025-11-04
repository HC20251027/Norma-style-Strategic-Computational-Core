"""
语音流水线数据模型

定义流水线处理过程中使用的数据结构和枚举
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid


class ProcessingStage(Enum):
    """处理阶段枚举"""
    RECEIVED = "received"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TOOL_MAPPING = "text_tool_mapping"
    TOOL_EXECUTION = "tool_execution"
    RESULT_PROCESSING = "result_processing"
    TEXT_TO_SPEECH = "text_to_speech"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStatus(Enum):
    """流水线状态枚举"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorType(Enum):
    """错误类型枚举"""
    SPEECH_RECOGNITION_ERROR = "speech_recognition_error"
    TEXT_PARSING_ERROR = "text_parsing_error"
    TOOL_MAPPING_ERROR = "tool_mapping_error"
    TOOL_EXECUTION_ERROR = "tool_execution_error"
    TEXT_TO_SPEECH_ERROR = "text_to_speech_error"
    PIPELINE_ERROR = "pipeline_error"
    SYSTEM_ERROR = "system_error"


@dataclass
class PipelineRequest:
    """流水线请求数据类"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    audio_data: Optional[bytes] = None
    audio_format: str = "wav"
    sample_rate: int = 16000
    language: str = "zh-CN"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # 扩展配置
    voice_config: Dict[str, Any] = field(default_factory=dict)
    tool_config: Dict[str, Any] = field(default_factory=dict)
    output_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeechToTextResult:
    """语音转文本结果"""
    text: str
    confidence: float
    language: str
    duration: float
    segments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """工具调用数据类"""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionResult:
    """工具执行结果"""
    call_id: str
    success: bool
    result: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TextToSpeechResult:
    """文本转语音结果"""
    audio_data: bytes
    format: str = "wav"
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PipelineState:
    """流水线状态数据类"""
    request_id: str
    status: PipelineStatus
    current_stage: ProcessingStage
    progress: float = 0.0
    error_message: Optional[str] = None
    error_type: Optional[ErrorType] = None
    
    # 处理结果
    speech_text: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    execution_results: List[ToolExecutionResult] = field(default_factory=list)
    final_text: Optional[str] = None
    response_audio: Optional[bytes] = None
    
    # 元数据
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_times: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResponse:
    """流水线响应数据类"""
    request_id: str
    success: bool
    status: PipelineStatus
    audio_response: Optional[bytes] = None
    text_response: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PipelineConfig:
    """流水线配置数据类"""
    # 语音识别配置
    speech_recognition: Dict[str, Any] = field(default_factory=lambda: {
        "engine": "whisper",
        "language": "zh-CN",
        "confidence_threshold": 0.7
    })
    
    # 文本处理配置
    text_processing: Dict[str, Any] = field(default_factory=lambda: {
        "intent_recognition": True,
        "entity_extraction": True,
        "context_awareness": True
    })
    
    # 工具映射配置
    tool_mapping: Dict[str, Any] = field(default_factory=lambda: {
        "max_tools_per_request": 5,
        "timeout_seconds": 30,
        "retry_attempts": 3
    })
    
    # 语音合成配置
    text_to_speech: Dict[str, Any] = field(default_factory=lambda: {
        "engine": "pyttsx3",
        "voice": "zh",
        "rate": 150,
        "volume": 0.8
    })
    
    # 性能配置
    performance: Dict[str, Any] = field(default_factory=lambda: {
        "max_concurrent_requests": 10,
        "cache_results": True,
        "enable_monitoring": True
    })
    
    # 错误处理配置
    error_handling: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "retry_delay": 1.0,
        "fallback_enabled": True
    })


@dataclass
class PipelineMetrics:
    """流水线性能指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_processing_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    error_counts: Dict[ErrorType, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)