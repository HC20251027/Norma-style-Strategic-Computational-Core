"""
工具定义和数据模型
"""

from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import uuid


class ToolCategory(Enum):
    """工具分类枚举"""
    SYSTEM_MONITOR = "system_monitor"
    NETWORK_SECURITY = "network_security"
    BLOODLINE_ANALYSIS = "bloodline_analysis"
    DATA_PROCESSING = "data_processing"
    FILE_MANAGEMENT = "file_management"
    API_INTEGRATION = "api_integration"
    WEB_SCRAPING = "web_scraping"
    DATABASE_OPERATIONS = "database_operations"
    SECURITY_ANALYSIS = "security_analysis"
    CUSTOM = "custom"


class ToolStatus(Enum):
    """工具状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"


class SecurityLevel(Enum):
    """安全级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    validation: Optional[Callable] = None
    options: Optional[List[Any]] = None


@dataclass
class ToolDefinition:
    """工具定义"""
    id: str
    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    author: str = "Unknown"
    parameters: List[ToolParameter] = field(default_factory=list)
    return_type: str = "dict"
    timeout: int = 30
    retry_count: int = 3
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    status: ToolStatus = ToolStatus.ACTIVE
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "version": self.version,
            "author": self.author,
            "parameters": [
                {
                    "name": param.name,
                    "type": param.type,
                    "description": param.description,
                    "required": param.required,
                    "default": param.default,
                    "options": param.options
                }
                for param in self.parameters
            ],
            "return_type": self.return_type,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "security_level": self.security_level.value,
            "status": self.status.value,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolDefinition':
        """从字典创建工具定义"""
        parameters = [
            ToolParameter(
                name=param["name"],
                type=param["type"],
                description=param["description"],
                required=param.get("required", True),
                default=param.get("default"),
                options=param.get("options")
            )
            for param in data.get("parameters", [])
        ]
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            category=ToolCategory(data["category"]),
            version=data.get("version", "1.0.0"),
            author=data.get("author", "Unknown"),
            parameters=parameters,
            return_type=data.get("return_type", "dict"),
            timeout=data.get("timeout", 30),
            retry_count=data.get("retry_count", 3),
            security_level=SecurityLevel(data.get("security_level", "medium")),
            status=ToolStatus(data.get("status", "active")),
            tags=data.get("tags", []),
            dependencies=data.get("dependencies", []),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        )


@dataclass
class ToolExecutionContext:
    """工具执行上下文"""
    tool_id: str
    parameters: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timeout: Optional[int] = None
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolExecutionResult:
    """工具执行结果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    tool_id: str = ""
    request_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time": self.execution_time,
            "tool_id": self.tool_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }