"""
权限管理模块
提供多级安全策略和权限控制
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import json
import uuid

from ..core.models import ToolDefinition, SecurityLevel


logger = logging.getLogger(__name__)


class PermissionType(Enum):
    """权限类型"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    DELETE = "delete"


class UserRole(Enum):
    """用户角色"""
    GUEST = "guest"
    USER = "user"
    OPERATOR = "operator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class SecurityPolicy(Enum):
    """安全策略"""
    STRICT = "strict"  # 严格模式
    MODERATE = "moderate"  # 中等模式
    LENIENT = "lenient"  # 宽松模式


@dataclass
class User:
    """用户"""
    id: str
    username: str
    role: UserRole
    permissions: Set[PermissionType] = field(default_factory=set)
    allowed_tools: Set[str] = field(default_factory=set)
    denied_tools: Set[str] = field(default_factory=set)
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    last_active: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityRule:
    """安全规则"""
    id: str
    name: str
    description: str
    rule_type: str  # "tool_access", "parameter_validation", "rate_limit", etc.
    conditions: Dict[str, Any]
    actions: Dict[str, Any]
    priority: int = 1
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AccessLog:
    """访问日志"""
    id: str
    user_id: str
    tool_id: str
    action: PermissionType
    result: bool  # 是否允许
    timestamp: datetime = field(default_factory=datetime.now)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class PermissionManager:
    """权限管理器"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.security_rules: Dict[str, SecurityRule] = {}
        self.access_logs: List[AccessLog] = []
        self.rate_limits: Dict[str, Dict[str, Any]] = {}  # 用户->工具->限制
        self.failed_attempts: Dict[str, List[datetime]] = {}  # 失败尝试记录
        self.blocked_users: Set[str] = set()
        self.security_policy = SecurityPolicy.MODERATE
        
        # 初始化默认用户和规则
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """初始化默认用户和安全规则"""
        # 创建默认管理员用户
        admin_user = User(
            id="admin",
            username="admin",
            role=UserRole.SUPER_ADMIN,
            permissions={perm for perm in PermissionType},
            security_level=SecurityLevel.CRITICAL
        )
        self.users["admin"] = admin_user
        
        # 创建默认操作员用户
        operator_user = User(
            id="operator",
            username="operator",
            role=UserRole.OPERATOR,
            permissions={PermissionType.READ, PermissionType.EXECUTE},
            security_level=SecurityLevel.HIGH
        )
        self.users["operator"] = operator_user
        
        # 创建默认普通用户
        normal_user = User(
            id="user",
            username="user",
            role=UserRole.USER,
            permissions={PermissionType.READ},
            security_level=SecurityLevel.MEDIUM
        )
        self.users["user"] = normal_user
        
        # 添加默认安全规则
        self._add_default_rules()
    
    def _add_default_rules(self):
        """添加默认安全规则"""
        # 高危工具访问规则
        high_risk_rule = SecurityRule(
            id="high_risk_tool_access",
            name="高危工具访问控制",
            description="限制对高危工具的访问",
            rule_type="tool_access",
            conditions={
                "security_level": "critical",
                "user_role": ["guest", "user"]
            },
            actions={
                "deny": True,
                "log": True,
                "notify": True
            },
            priority=1
        )
        self.security_rules[high_risk_rule.id] = high_risk_rule
        
        # 参数验证规则
        param_validation_rule = SecurityRule(
            id="parameter_validation",
            name="参数验证",
            description="验证工具参数的安全性",
            rule_type="parameter_validation",
            conditions={
                "parameter_count": ">10",
                "parameter_size": ">10000"
            },
            actions={
                "validate": True,
                "sanitize": True
            },
            priority=2
        )
        self.security_rules[param_validation_rule.id] = param_validation_rule
        
        # 频率限制规则
        rate_limit_rule = SecurityRule(
            id="rate_limit",
            name="频率限制",
            description="限制工具调用频率",
            rule_type="rate_limit",
            conditions={
                "requests_per_minute": ">60"
            },
            actions={
                "throttle": True,
                "log": True
            },
            priority=3
        )
        self.security_rules[rate_limit_rule.id] = rate_limit_rule
    
    async def check_permission(self, tool_id: str, parameters: Dict[str, Any], 
                              user_id: str = "user") -> bool:
        """检查权限"""
        try:
            # 检查用户是否存在且激活
            user = self.users.get(user_id)
            if not user or not user.is_active:
                await self._log_access(user_id, tool_id, PermissionType.EXECUTE, False, 
                                     {"reason": "用户不存在或未激活"})
                return False
            
            # 检查用户是否被封禁
            if user_id in self.blocked_users:
                await self._log_access(user_id, tool_id, PermissionType.EXECUTE, False,
                                     {"reason": "用户被封禁"})
                return False
            
            # 检查工具访问权限
            if not self._check_tool_access(user, tool_id):
                await self._log_access(user_id, tool_id, PermissionType.EXECUTE, False,
                                     {"reason": "工具访问被拒绝"})
                return False
            
            # 检查安全规则
            if not await self._check_security_rules(user, tool_id, parameters):
                await self._log_access(user_id, tool_id, PermissionType.EXECUTE, False,
                                     {"reason": "安全规则检查失败"})
                return False
            
            # 检查频率限制
            if not await self._check_rate_limit(user_id, tool_id):
                await self._log_access(user_id, tool_id, PermissionType.EXECUTE, False,
                                     {"reason": "频率限制触发"})
                return False
            
            # 记录成功访问
            await self._log_access(user_id, tool_id, PermissionType.EXECUTE, True)
            
            # 更新用户最后活跃时间
            user.last_active = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"权限检查异常: {e}")
            return False
    
    def _check_tool_access(self, user: User, tool_id: str) -> bool:
        """检查工具访问权限"""
        # 检查明确拒绝的工具
        if tool_id in user.denied_tools:
            return False
        
        # 检查明确允许的工具
        if tool_id in user.allowed_tools:
            return True
        
        # 基于角色的默认权限
        role_permissions = {
            UserRole.GUEST: {"cpu_monitor", "memory_monitor"},
            UserRole.USER: {"cpu_monitor", "memory_monitor", "disk_monitor"},
            UserRole.OPERATOR: {"cpu_monitor", "memory_monitor", "disk_monitor", "network_monitor", "port_scanner"},
            UserRole.ADMIN: set(),  # 允许所有
            UserRole.SUPER_ADMIN: set()  # 允许所有
        }
        
        default_allowed = role_permissions.get(user.role, set())
        return tool_id in default_allowed or user.role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]
    
    async def _check_security_rules(self, user: User, tool_id: str, parameters: Dict[str, Any]) -> bool:
        """检查安全规则"""
        # 获取工具定义
        tool = global_registry.get_tool(tool_id)
        if not tool:
            return False
        
        # 按优先级排序规则
        sorted_rules = sorted(
            self.security_rules.values(),
            key=lambda x: x.priority,
            reverse=True
        )
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            # 检查规则条件
            if self._evaluate_rule_conditions(rule.conditions, user, tool, parameters):
                # 执行规则动作
                if not self._execute_rule_actions(rule.actions, user, tool, parameters):
                    return False
        
        return True
    
    def _evaluate_rule_conditions(self, conditions: Dict[str, Any], user: User, 
                                 tool: ToolDefinition, parameters: Dict[str, Any]) -> bool:
        """评估规则条件"""
        for key, value in conditions.items():
            if key == "security_level":
                if not self._check_security_level(user.security_level, tool.security_level):
                    return False
            
            elif key == "user_role":
                if isinstance(value, list) and user.role.value not in value:
                    return False
                elif isinstance(value, str) and user.role.value != value:
                    return False
            
            elif key == "parameter_count":
                param_count = len(parameters)
                if not self._evaluate_comparison(param_count, value):
                    return False
            
            elif key == "parameter_size":
                param_size = len(json.dumps(parameters))
                if not self._evaluate_comparison(param_size, value):
                    return False
            
            elif key == "requests_per_minute":
                current_rate = self._get_request_rate(user.id)
                if not self._evaluate_comparison(current_rate, value):
                    return False
        
        return True
    
    def _execute_rule_actions(self, actions: Dict[str, Any], user: User, 
                             tool: ToolDefinition, parameters: Dict[str, Any]) -> bool:
        """执行规则动作"""
        for action, value in actions.items():
            if action == "deny" and value:
                return False
            
            elif action == "log" and value:
                logger.info(f"安全规则触发: 用户 {user.username} 尝试访问工具 {tool.name}")
            
            elif action == "notify" and value:
                # 发送通知（实际实现中应该集成通知系统）
                logger.warning(f"高危操作通知: 用户 {user.username} 尝试访问高危工具 {tool.name}")
        
        return True
    
    def _check_security_level(self, user_level: SecurityLevel, tool_level: SecurityLevel) -> bool:
        """检查安全级别"""
        level_hierarchy = {
            SecurityLevel.LOW: 1,
            SecurityLevel.MEDIUM: 2,
            SecurityLevel.HIGH: 3,
            SecurityLevel.CRITICAL: 4
        }
        
        user_level_num = level_hierarchy.get(user_level, 1)
        tool_level_num = level_hierarchy.get(tool_level, 1)
        
        # 用户级别必须大于等于工具级别
        return user_level_num >= tool_level_num
    
    def _evaluate_comparison(self, value: Any, condition: str) -> bool:
        """评估比较条件"""
        try:
            # 解析条件字符串，如 ">10", ">=5", "==3" 等
            operators = [">=", "<=", ">", "<", "==", "!="]
            
            for op in operators:
                if op in condition:
                    threshold = float(condition.replace(op, ""))
                    
                    if op == ">":
                        return value > threshold
                    elif op == ">=":
                        return value >= threshold
                    elif op == "<":
                        return value < threshold
                    elif op == "<=":
                        return value <= threshold
                    elif op == "==":
                        return value == threshold
                    elif op == "!=":
                        return value != threshold
            
            return False
            
        except (ValueError, TypeError):
            return False
    
    async def _check_rate_limit(self, user_id: str, tool_id: str) -> bool:
        """检查频率限制"""
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=1)
        
        # 获取用户的访问日志
        recent_logs = [
            log for log in self.access_logs
            if (log.user_id == user_id and 
                log.tool_id == tool_id and 
                log.timestamp > window_start)
        ]
        
        # 检查是否超过限制
        max_requests = self._get_rate_limit(user_id, tool_id)
        if len(recent_logs) >= max_requests:
            # 记录失败尝试
            if user_id not in self.failed_attempts:
                self.failed_attempts[user_id] = []
            
            self.failed_attempts[user_id].append(current_time)
            
            # 检查是否需要封禁
            failed_attempts = [
                attempt for attempt in self.failed_attempts.get(user_id, [])
                if attempt > current_time - timedelta(minutes=15)
            ]
            
            if len(failed_attempts) >= 5:  # 15分钟内5次失败
                self.blocked_users.add(user_id)
                logger.warning(f"用户 {user_id} 因频繁失败被封禁")
            
            return False
        
        return True
    
    def _get_rate_limit(self, user_id: str, tool_id: str) -> int:
        """获取频率限制"""
        user = self.users.get(user_id)
        if not user:
            return 10  # 默认限制
        
        # 基于角色的默认限制
        role_limits = {
            UserRole.GUEST: 5,
            UserRole.USER: 20,
            UserRole.OPERATOR: 50,
            UserRole.ADMIN: 100,
            UserRole.SUPER_ADMIN: 1000
        }
        
        return role_limits.get(user.role, 20)
    
    def _get_request_rate(self, user_id: str) -> float:
        """获取请求频率（每分钟）"""
        current_time = datetime.now()
        window_start = current_time - timedelta(minutes=1)
        
        recent_requests = [
            log for log in self.access_logs
            if (log.user_id == user_id and 
                log.timestamp > window_start)
        ]
        
        return len(recent_requests)
    
    async def _log_access(self, user_id: str, tool_id: str, action: PermissionType, 
                         result: bool, details: Dict[str, Any] = None):
        """记录访问日志"""
        log_entry = AccessLog(
            id=str(uuid.uuid4()),
            user_id=user_id,
            tool_id=tool_id,
            action=action,
            result=result,
            details=details or {}
        )
        
        self.access_logs.append(log_entry)
        
        # 限制日志大小
        if len(self.access_logs) > 10000:
            self.access_logs = self.access_logs[-5000:]  # 保留最近5000条
    
    def add_user(self, user: User) -> bool:
        """添加用户"""
        try:
            self.users[user.id] = user
            logger.info(f"用户添加成功: {user.username}")
            return True
        except Exception as e:
            logger.error(f"用户添加失败: {e}")
            return False
    
    def remove_user(self, user_id: str) -> bool:
        """移除用户"""
        if user_id in self.users:
            del self.users[user_id]
            logger.info(f"用户移除成功: {user_id}")
            return True
        return False
    
    def update_user_permissions(self, user_id: str, permissions: Set[PermissionType]) -> bool:
        """更新用户权限"""
        if user_id in self.users:
            self.users[user_id].permissions = permissions
            logger.info(f"用户权限更新成功: {user_id}")
            return True
        return False
    
    def add_security_rule(self, rule: SecurityRule) -> bool:
        """添加安全规则"""
        try:
            self.security_rules[rule.id] = rule
            logger.info(f"安全规则添加成功: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"安全规则添加失败: {e}")
            return False
    
    def remove_security_rule(self, rule_id: str) -> bool:
        """移除安全规则"""
        if rule_id in self.security_rules:
            del self.security_rules[rule_id]
            logger.info(f"安全规则移除成功: {rule_id}")
            return True
        return False
    
    def get_access_logs(self, user_id: Optional[str] = None, tool_id: Optional[str] = None, 
                       limit: int = 100) -> List[Dict[str, Any]]:
        """获取访问日志"""
        logs = self.access_logs
        
        if user_id:
            logs = [log for log in logs if log.user_id == user_id]
        
        if tool_id:
            logs = [log for log in logs if log.tool_id == tool_id]
        
        # 按时间倒序排列
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        return [
            {
                "id": log.id,
                "user_id": log.user_id,
                "tool_id": log.tool_id,
                "action": log.action.value,
                "result": log.result,
                "timestamp": log.timestamp.isoformat(),
                "details": log.details
            }
            for log in logs[:limit]
        ]
    
    def get_security_statistics(self) -> Dict[str, Any]:
        """获取安全统计"""
        current_time = datetime.now()
        last_24h = current_time - timedelta(hours=24)
        
        recent_logs = [log for log in self.access_logs if log.timestamp > last_24h]
        successful_accesses = [log for log in recent_logs if log.result]
        failed_accesses = [log for log in recent_logs if not log.result]
        
        return {
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() if u.is_active]),
            "blocked_users": len(self.blocked_users),
            "total_rules": len(self.security_rules),
            "enabled_rules": len([r for r in self.security_rules.values() if r.enabled]),
            "recent_accesses_24h": len(recent_logs),
            "successful_accesses_24h": len(successful_accesses),
            "failed_accesses_24h": len(failed_accesses),
            "success_rate_24h": (len(successful_accesses) / max(len(recent_logs), 1)) * 100,
            "security_policy": self.security_policy.value
        }


# 全局权限管理器实例
permission_manager = PermissionManager()