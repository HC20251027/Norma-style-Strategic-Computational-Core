#!/usr/bin/env python3
"""
多智能体协作系统主入口
整合所有协作组件，提供完整的多智能体协作能力

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from pathlib import Path

from .core.coordination_manager import CoordinationManager
from .core.agent_collaboration_manager import AgentCollaborationManager, CollaborationType, AgentCapability
from .task_management.task_distributor import TaskDistributor, Task, TaskType, AssignmentStrategy
from .task_management.task_coordinator import TaskCoordinator, CoordinationWorkflow, CoordinationMode, TaskDependency, DependencyType
from .agent_pool.agent_pool_manager import AgentPoolManager, AgentInfo, PoolConfiguration, PoolStrategy
from .agent_pool.load_balancer import LoadBalancer, LoadBalanceRequest, LoadBalancingStrategy
from .communication.message_bus import MessageBus
from .communication.synchronizer import AgentSynchronizer, SyncType, SyncRequest
from .coordination.collaboration_patterns import CollaborationPatterns, CollaborationPattern
from .evaluation.performance_evaluator import PerformanceEvaluator
from .evaluation.efficiency_optimizer import EfficiencyOptimizer
from .utils.logger import MultiAgentLogger, get_logger


class MultiAgentCollaborationSystem:
    """多智能体协作系统主类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化多智能体协作系统
        
        Args:
            config_path: 配置文件路径
        """
        self.system_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        # 初始化日志
        self.logger = get_logger("multi_agent_system")
        
        # 核心组件
        self.message_bus: Optional[MessageBus] = None
        self.collaboration_manager: Optional[AgentCollaborationManager] = None
        self.coordination_manager: Optional[CoordinationManager] = None
        self.task_distributor: Optional[TaskDistributor] = None
        self.task_coordinator: Optional[TaskCoordinator] = None
        self.agent_pool: Optional[AgentPoolManager] = None
        self.load_balancer: Optional[LoadBalancer] = None
        self.synchronizer: Optional[AgentSynchronizer] = None
        self.collaboration_patterns: Optional[CollaborationPatterns] = None
        self.performance_evaluator: Optional[PerformanceEvaluator] = None
        self.efficiency_optimizer: Optional[EfficiencyOptimizer] = None
        
        # 系统状态
        self.is_initialized = False
        self.is_running = False
        
        # 配置
        self.config = self._load_config(config_path)
        
        # 统计信息
        self.metrics = {
            "agents_registered": 0,
            "collaborations_created": 0,
            "tasks_processed": 0,
            "optimizations_performed": 0,
            "system_uptime": 0.0,
            "performance_score": 0.0,
            "efficiency_score": 0.0
        }
        
        self.logger.info(f"多智能体协作系统 {self.system_id} 已初始化")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载系统配置"""
        default_config = {
            "system": {
                "name": "Norma Multi-Agent Collaboration System",
                "version": "1.0.0",
                "max_agents": 100,
                "max_concurrent_collaborations": 50,
                "default_timeout": 300
            },
            "components": {
                "message_bus": {
                    "buffer_size": 1000,
                    "enable_compression": True
                },
                "agent_pool": {
                    "strategy": "dynamic_scaling",
                    "min_agents": 2,
                    "max_agents": 50,
                    "health_check_interval": 30
                },
                "load_balancer": {
                    "strategy": "adaptive",
                    "enable_predictive": True
                },
                "performance_evaluation": {
                    "evaluation_interval": 300,
                    "enable_optimization": True
                }
            },
            "logging": {
                "level": "INFO",
                "enable_file_logging": True,
                "log_retention_days": 30
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
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
        """初始化系统"""
        try:
            self.logger.info("初始化多智能体协作系统...")
            
            # 1. 初始化消息总线
            self.message_bus = MessageBus(
                buffer_size=self.config["components"]["message_bus"]["buffer_size"]
            )
            await self.message_bus.initialize()
            self.logger.info("✓ 消息总线初始化完成")
            
            # 2. 初始化智能体池管理器
            self.agent_pool = AgentPoolManager(
                message_bus=self.message_bus,
                max_agents=self.config["system"]["max_agents"]
            )
            await self.agent_pool.initialize()
            self.logger.info("✓ 智能体池管理器初始化完成")
            
            # 3. 初始化负载均衡器
            self.load_balancer = LoadBalancer(message_bus=self.message_bus)
            await self.load_balancer.initialize()
            self.logger.info("✓ 负载均衡器初始化完成")
            
            # 4. 初始化任务分配器
            self.task_distributor = TaskDistributor(
                message_bus=self.message_bus,
                timeout=self.config["system"]["default_timeout"]
            )
            await self.task_distributor.initialize()
            self.logger.info("✓ 任务分配器初始化完成")
            
            # 5. 初始化任务协调器
            self.task_coordinator = TaskCoordinator(message_bus=self.message_bus)
            await self.task_coordinator.initialize()
            self.logger.info("✓ 任务协调器初始化完成")
            
            # 6. 初始化智能体同步器
            self.synchronizer = AgentSynchronizer(message_bus=self.message_bus)
            await self.synchronizer.initialize()
            self.logger.info("✓ 智能体同步器初始化完成")
            
            # 7. 初始化协作模式管理器
            self.collaboration_patterns = CollaborationPatterns(message_bus=self.message_bus)
            await self.collaboration_patterns.initialize()
            self.logger.info("✓ 协作模式管理器初始化完成")
            
            # 8. 初始化协作管理器
            self.collaboration_manager = AgentCollaborationManager()
            await self.collaboration_manager.initialize()
            self.logger.info("✓ 协作管理器初始化完成")
            
            # 9. 初始化协调管理器
            self.coordination_manager = CoordinationManager()
            await self.coordination_manager.initialize(self.collaboration_manager)
            self.logger.info("✓ 协调管理器初始化完成")
            
            # 10. 初始化性能评估器
            self.performance_evaluator = PerformanceEvaluator()
            await self.performance_evaluator.initialize()
            self.logger.info("✓ 性能评估器初始化完成")
            
            # 11. 初始化效率优化器
            self.efficiency_optimizer = EfficiencyOptimizer()
            await self.efficiency_optimizer.initialize()
            self.logger.info("✓ 效率优化器初始化完成")
            
            # 12. 建立组件间的连接
            await self._setup_component_connections()
            
            # 13. 启动后台监控任务
            asyncio.create_task(self._system_monitor())
            asyncio.create_task(self._performance_monitor())
            asyncio.create_task(self._health_checker())
            
            self.is_initialized = True
            self.logger.info("多智能体协作系统初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            return False
    
    async def _setup_component_connections(self):
        """建立组件间的连接"""
        # 连接智能体池和负载均衡器
        if self.agent_pool and self.load_balancer:
            # 注册智能体到负载均衡器
            pass  # 将在注册智能体时处理
        
        # 连接任务分配器和协调管理器
        if self.task_distributor and self.coordination_manager:
            # 建立任务分配协调
            pass  # 将在任务分配时处理
        
        # 连接性能评估器和效率优化器
        if self.performance_evaluator and self.efficiency_optimizer:
            # 建立性能优化闭环
            pass  # 将在性能评估时处理
    
    async def start(self) -> bool:
        """启动系统"""
        if not self.is_initialized:
            self.logger.error("系统尚未初始化，请先调用initialize()")
            return False
        
        try:
            self.logger.info("启动多智能体协作系统...")
            
            # 启动各个组件
            if self.collaboration_manager:
                await self.collaboration_manager.start()
            
            self.is_running = True
            
            # 发送系统启动事件
            await self._emit_system_event("system.started", {
                "system_id": self.system_id,
                "start_time": self.start_time.isoformat(),
                "components": self._get_active_components()
            })
            
            self.logger.info("多智能体协作系统启动完成")
            return True
            
        except Exception as e:
            self.logger.error(f"系统启动失败: {e}")
            return False
    
    async def stop(self) -> bool:
        """停止系统"""
        if not self.is_running:
            self.logger.warning("系统未在运行状态")
            return True
        
        try:
            self.logger.info("停止多智能体协作系统...")
            
            # 停止各个组件
            components = [
                ("collaboration_manager", self.collaboration_manager),
                ("coordination_manager", self.coordination_manager),
                ("efficiency_optimizer", self.efficiency_optimizer),
                ("performance_evaluator", self.performance_evaluator),
                ("collaboration_patterns", self.collaboration_patterns),
                ("synchronizer", self.synchronizer),
                ("task_coordinator", self.task_coordinator),
                ("task_distributor", self.task_distributor),
                ("load_balancer", self.load_balancer),
                ("agent_pool", self.agent_pool),
                ("message_bus", self.message_bus)
            ]
            
            for name, component in components:
                if component:
                    try:
                        await component.shutdown()
                        self.logger.info(f"✓ {name} 已停止")
                    except Exception as e:
                        self.logger.error(f"停止 {name} 时出错: {e}")
            
            self.is_running = False
            
            # 发送系统停止事件
            await self._emit_system_event("system.stopped", {
                "system_id": self.system_id,
                "stop_time": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
            })
            
            self.logger.info("多智能体协作系统已停止")
            return True
            
        except Exception as e:
            self.logger.error(f"系统停止失败: {e}")
            return False
    
    async def register_agent(
        self,
        agent_id: str,
        agent_info: Dict[str, Any],
        capabilities: List[AgentCapability]
    ) -> bool:
        """注册智能体"""
        try:
            # 注册到协作管理器
            if self.collaboration_manager:
                success = await self.collaboration_manager.register_agent(agent_id, agent_info, capabilities)
                if not success:
                    return False
            
            # 注册到智能体池
            if self.agent_pool:
                success = await self.agent_pool.register_agent(agent_id, agent_info, capabilities)
                if not success:
                    return False
            
            # 注册到负载均衡器
            if self.load_balancer:
                # 创建AgentInfo对象
                agent_info_obj = AgentInfo(
                    agent_id=agent_id,
                    name=agent_info.get("name", agent_id),
                    agent_type=agent_info.get("type", "general"),
                    capabilities=capabilities,
                    status=agent_info.get("status", "idle"),
                    registered_at=datetime.now(),
                    last_heartbeat=datetime.now(),
                    health_status=agent_info.get("health_status", "unknown")
                )
                await self.load_balancer.register_agent(agent_info_obj)
            
            # 注册到协调管理器
            if self.coordination_manager:
                self.coordination_manager.register_agent_capabilities(agent_id, capabilities)
            
            self.metrics["agents_registered"] += 1
            
            await self._emit_system_event("agent.registered", {
                "agent_id": agent_id,
                "capabilities": [cap.name for cap in capabilities]
            })
            
            self.logger.info(f"智能体 {agent_id} 注册成功")
            return True
            
        except Exception as e:
            self.logger.error(f"注册智能体失败 {agent_id}: {e}")
            return False
    
    async def create_collaboration(
        self,
        collaboration_type: CollaborationType,
        participant_agents: List[str],
        task_description: str,
        pattern: Optional[CollaborationPattern] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建协作"""
        try:
            if not self.collaboration_manager:
                raise RuntimeError("协作管理器未初始化")
            
            session_id = await self.collaboration_manager.create_collaboration(
                collaboration_type=collaboration_type,
                participant_agents=participant_agents,
                task_description=task_description,
                metadata=metadata
            )
            
            self.metrics["collaborations_created"] += 1
            
            await self._emit_system_event("collaboration.created", {
                "session_id": session_id,
                "type": collaboration_type.value,
                "participants": participant_agents
            })
            
            self.logger.info(f"协作会话 {session_id} 已创建")
            return session_id
            
        except Exception as e:
            self.logger.error(f"创建协作失败: {e}")
            raise
    
    async def execute_collaboration(
        self,
        session_id: str,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行协作"""
        try:
            if not self.collaboration_manager:
                raise RuntimeError("协作管理器未初始化")
            
            # 获取协作会话信息
            session_info = await self.collaboration_manager.get_collaboration_status(session_id)
            if "error" in session_info:
                raise ValueError(f"协作会话 {session_id} 不存在")
            
            # 执行协作
            result = await self.collaboration_manager.execute_collaboration(
                session_id=session_id,
                task_data=task_data
            )
            
            await self._emit_system_event("collaboration.executed", {
                "session_id": session_id,
                "result": result
            })
            
            self.logger.info(f"协作执行完成: {session_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"执行协作失败 {session_id}: {e}")
            raise
    
    async def submit_task(
        self,
        task_type: TaskType,
        description: str,
        priority: int = 5,
        required_capabilities: Optional[List[str]] = None,
        resource_requirements: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> str:
        """提交任务"""
        try:
            if not self.task_distributor:
                raise RuntimeError("任务分配器未初始化")
            
            task = Task(
                task_id=str(uuid.uuid4()),
                task_type=task_type,
                description=description,
                priority=priority,
                created_at=datetime.now(),
                required_capabilities=required_capabilities or [],
                resource_requirements=resource_requirements or {},
                estimated_duration=timeout or 60.0
            )
            
            task_id = await self.task_distributor.submit_task(task)
            
            self.metrics["tasks_processed"] += 1
            
            await self._emit_system_event("task.submitted", {
                "task_id": task_id,
                "task_type": task_type.value,
                "priority": priority
            })
            
            self.logger.info(f"任务 {task_id} 已提交")
            return task_id
            
        except Exception as e:
            self.logger.error(f"提交任务失败: {e}")
            raise
    
    async def create_workflow(
        self,
        name: str,
        tasks: List[Task],
        dependencies: Optional[List[TaskDependency]] = None,
        coordination_mode: Optional[CoordinationMode] = None
    ) -> str:
        """创建工作流"""
        try:
            if not self.task_coordinator:
                raise RuntimeError("任务协调器未初始化")
            
            workflow_id = await self.task_coordinator.create_workflow(
                name=name,
                tasks=tasks,
                dependencies=dependencies,
                coordination_mode=coordination_mode or CoordinationMode.PARALLEL
            )
            
            await self._emit_system_event("workflow.created", {
                "workflow_id": workflow_id,
                "name": name,
                "task_count": len(tasks)
            })
            
            self.logger.info(f"工作流 {workflow_id} 已创建")
            return workflow_id
            
        except Exception as e:
            self.logger.error(f"创建工作流失败: {e}")
            raise
    
    async def execute_workflow(self, workflow_id: str) -> str:
        """执行工作流"""
        try:
            if not self.task_coordinator:
                raise RuntimeError("任务协调器未初始化")
            
            execution_id = await self.task_coordinator.execute_workflow(workflow_id)
            
            await self._emit_system_event("workflow.started", {
                "workflow_id": workflow_id,
                "execution_id": execution_id
            })
            
            self.logger.info(f"工作流 {workflow_id} 开始执行")
            return execution_id
            
        except Exception as e:
            self.logger.error(f"执行工作流失败 {workflow_id}: {e}")
            raise
    
    async def request_sync(
        self,
        sync_type: SyncType,
        requester_id: str,
        target_agents: List[str],
        sync_data: Dict[str, Any],
        timeout: float = 30.0
    ) -> str:
        """请求同步"""
        try:
            if not self.synchronizer:
                raise RuntimeError("同步器未初始化")
            
            request_id = await self.synchronizer.request_sync(
                sync_type=sync_type,
                requester_id=requester_id,
                target_agents=target_agents,
                sync_data=sync_data,
                timeout=timeout
            )
            
            await self._emit_system_event("sync.requested", {
                "request_id": request_id,
                "sync_type": sync_type.value,
                "requester_id": requester_id
            })
            
            self.logger.info(f"同步请求 {request_id} 已创建")
            return request_id
            
        except Exception as e:
            self.logger.error(f"请求同步失败: {e}")
            raise
    
    async def optimize_performance(self, optimization_goals: Optional[List[str]] = None) -> Dict[str, Any]:
        """优化性能"""
        try:
            if not self.performance_evaluator or not self.efficiency_optimizer:
                raise RuntimeError("性能评估器或效率优化器未初始化")
            
            # 获取当前性能数据
            agents = {}  # 这里应该从实际系统获取
            collaboration_history = []  # 这里应该从实际系统获取
            active_collaborations = {}  # 这里应该从实际系统获取
            
            # 评估性能
            performance_data = await self.performance_evaluator.evaluate_system_performance(
                agents, collaboration_history, active_collaborations
            )
            
            # 执行优化
            optimization_result = await self.efficiency_optimizer.optimize_system(
                performance_data, optimization_goals
            )
            
            self.metrics["optimizations_performed"] += 1
            
            await self._emit_system_event("optimization.completed", optimization_result)
            
            self.logger.info(f"性能优化完成，改进: {optimization_result.get('total_improvement', 0):.2%}")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"性能优化失败: {e}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        status = {
            "system_id": self.system_id,
            "status": "running" if self.is_running else "stopped",
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_duration(uptime),
            "initialized": self.is_initialized,
            "metrics": self.metrics.copy(),
            "components": self._get_component_status(),
            "configuration": self.config
        }
        
        return status
    
    def _get_active_components(self) -> List[str]:
        """获取活跃组件列表"""
        active_components = []
        
        components = [
            ("message_bus", self.message_bus),
            ("collaboration_manager", self.collaboration_manager),
            ("coordination_manager", self.coordination_manager),
            ("task_distributor", self.task_distributor),
            ("task_coordinator", self.task_coordinator),
            ("agent_pool", self.agent_pool),
            ("load_balancer", self.load_balancer),
            ("synchronizer", self.synchronizer),
            ("collaboration_patterns", self.collaboration_patterns),
            ("performance_evaluator", self.performance_evaluator),
            ("efficiency_optimizer", self.efficiency_optimizer)
        ]
        
        for name, component in components:
            if component:
                active_components.append(name)
        
        return active_components
    
    def _get_component_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        status = {}
        
        # 消息总线状态
        if self.message_bus:
            status["message_bus"] = self.message_bus.get_bus_status()
        
        # 协作管理器状态
        if self.collaboration_manager:
            status["collaboration_manager"] = {
                "registered_agents": len(self.collaboration_manager.registered_agents),
                "active_collaborations": len(self.collaboration_manager.active_collaborations)
            }
        
        # 协调管理器状态
        if self.coordination_manager:
            status["coordination_manager"] = self.coordination_manager.get_coordination_status()
        
        # 任务分配器状态
        if self.task_distributor:
            status["task_distributor"] = self.task_distributor.get_distribution_status()
        
        # 任务协调器状态
        if self.task_coordinator:
            status["task_coordinator"] = self.task_coordinator.get_coordination_status()
        
        # 智能体池状态
        if self.agent_pool:
            status["agent_pool"] = self.agent_pool.get_pool_status()
        
        # 负载均衡器状态
        if self.load_balancer:
            status["load_balancer"] = self.load_balancer.get_load_balance_status()
        
        # 同步器状态
        if self.synchronizer:
            status["synchronizer"] = self.synchronizer.get_sync_status()
        
        # 协作模式状态
        if self.collaboration_patterns:
            status["collaboration_patterns"] = self.collaboration_patterns.get_collaboration_status()
        
        # 性能评估器状态
        if self.performance_evaluator:
            status["performance_evaluator"] = self.performance_evaluator.get_performance_summary()
        
        # 效率优化器状态
        if self.efficiency_optimizer:
            status["efficiency_optimizer"] = self.efficiency_optimizer.get_optimization_status()
        
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
    
    async def _system_monitor(self):
        """系统监控后台任务"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟监控一次
                
                # 更新系统指标
                self.metrics["system_uptime"] = (datetime.now() - self.start_time).total_seconds()
                
                # 发送系统健康状态
                await self._emit_system_event("system.monitor", {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": self.metrics,
                    "status": "healthy"
                })
                
            except Exception as e:
                self.logger.error(f"系统监控任务出错: {e}")
    
    async def _performance_monitor(self):
        """性能监控后台任务"""
        while True:
            try:
                await asyncio.sleep(300)  # 每5分钟监控一次
                
                # 执行性能评估
                if self.performance_evaluator:
                    # 这里可以添加自动性能评估逻辑
                    pass
                
            except Exception as e:
                self.logger.error(f"性能监控任务出错: {e}")
    
    async def _health_checker(self):
        """健康检查后台任务"""
        while True:
            try:
                await asyncio.sleep(120)  # 每2分钟检查一次
                
                # 检查各个组件的健康状态
                unhealthy_components = []
                
                components = [
                    ("collaboration_manager", self.collaboration_manager),
                    ("coordination_manager", self.coordination_manager),
                    ("task_distributor", self.task_distributor),
                    ("task_coordinator", self.task_coordinator),
                    ("agent_pool", self.agent_pool),
                    ("load_balancer", self.load_balancer),
                    ("synchronizer", self.synchronizer),
                    ("collaboration_patterns", self.collaboration_patterns),
                    ("performance_evaluator", self.performance_evaluator),
                    ("efficiency_optimizer", self.efficiency_optimizer)
                ]
                
                for name, component in components:
                    if component and hasattr(component, 'health_check'):
                        try:
                            # 这里可以调用具体的健康检查方法
                            pass
                        except Exception as e:
                            unhealthy_components.append(name)
                
                if unhealthy_components:
                    await self._emit_system_event("system.health_warning", {
                        "unhealthy_components": unhealthy_components,
                        "timestamp": datetime.now().isoformat()
                    })
                
            except Exception as e:
                self.logger.error(f"健康检查任务出错: {e}")
    
    async def _emit_system_event(self, event_type: str, data: Dict[str, Any]):
        """发送系统事件"""
        if self.message_bus:
            await self.message_bus.publish_event(event_type, data)
    
    def on_event(self, event_type: str, callback: Callable):
        """注册系统事件回调"""
        if self.message_bus:
            self.message_bus.on_event(event_type, callback)


# 全局系统实例
_global_system: Optional[MultiAgentCollaborationSystem] = None


async def get_multi_agent_system(config_path: Optional[str] = None) -> MultiAgentCollaborationSystem:
    """获取全局多智能体协作系统实例"""
    global _global_system
    if _global_system is None:
        _global_system = MultiAgentCollaborationSystem(config_path)
        await _global_system.initialize()
        await _global_system.start()
    return _global_system


async def shutdown_multi_agent_system():
    """关闭全局多智能体协作系统实例"""
    global _global_system
    if _global_system:
        await _global_system.stop()
        _global_system = None


# 便捷函数
async def quick_setup_demo() -> MultiAgentCollaborationSystem:
    """快速设置演示系统"""
    system = await get_multi_agent_system()
    
    # 注册一些示例智能体
    example_capabilities = [
        AgentCapability("data_analysis", "数据分析能力", 5, 30.0, {"cpu": 2, "memory": 4}, 0.9, 0.85),
        AgentCapability("text_generation", "文本生成能力", 3, 20.0, {"cpu": 1, "memory": 2}, 0.95, 0.90),
        AgentCapability("image_processing", "图像处理能力", 7, 60.0, {"cpu": 4, "memory": 8}, 0.85, 0.80)
    ]
    
    # 注册示例智能体
    for i in range(3):
        agent_id = f"demo_agent_{i+1}"
        agent_info = {
            "name": f"演示智能体{i+1}",
            "type": "demo",
            "status": "idle"
        }
        
        # 为每个智能体分配不同的能力
        capabilities = [example_capabilities[i % len(example_capabilities)]]
        await system.register_agent(agent_id, agent_info, capabilities)
    
    return system