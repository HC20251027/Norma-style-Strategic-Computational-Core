#!/usr/bin/env python3
"""
多智能体协作系统共享类型定义
避免循环导入问题

作者: 皇
创建时间: 2025-10-31
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


class AgentStatus(Enum):
    """智能体状态枚举"""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class AgentCapability(Enum):
    """智能体能力枚举"""
    TASK_EXECUTION = "task_execution"
    DATA_PROCESSING = "data_processing"
    DATA_ANALYSIS = "data_analysis"
    MACHINE_LEARNING = "machine_learning"
    VISUALIZATION = "visualization"
    COMMUNICATION = "communication"
    COORDINATION = "coordination"
    MANAGEMENT = "management"
    SUPERVISION = "supervision"
    REPORTING = "reporting"
    VALIDATION = "validation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"


class CollaborationType(Enum):
    """协作类型枚举"""
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    PIPELINE = "pipeline"
    ENSEMBLE = "ensemble"
    CONSENSUS = "consensus"


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING = "waiting"


class LoadBalancingStrategy(Enum):
    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAPABILITY_BASED = "capability_based"
    WEIGHTED = "weighted"


@dataclass
class AgentInfo:
    """智能体信息"""
    id: str
    name: str
    capabilities: List[str]
    status: AgentStatus
    current_load: float = 0.0
    performance_score: float = 1.0
    last_heartbeat: Optional[float] = None


@dataclass
class TaskInfo:
    """任务信息"""
    id: str
    name: str
    description: str
    priority: int
    dependencies: List[str]
    required_capabilities: List[str]
    estimated_duration: int
    status: TaskStatus
    assigned_agent: Optional[str] = None
    created_time: Optional[float] = None


@dataclass
class MessageInfo:
    """消息信息"""
    id: str
    type: str
    content: Any
    sender: str
    receiver: Optional[str]
    timestamp: float
    topic: Optional[str] = None


# 导出所有类型
__all__ = [
    'AgentStatus',
    'AgentCapability', 
    'CollaborationType',
    'TaskStatus',
    'LoadBalancingStrategy',
    'AgentInfo',
    'TaskInfo',
    'MessageInfo'
]