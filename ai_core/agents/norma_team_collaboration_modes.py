#!/usr/bin/env python3
"""
诺玛Team协作模式实现
基于Agno框架的三种智能体协作模式

作者: 皇
创建时间: 2025-11-01
版本: 1.0.0
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# Agno框架导入
try:
    from agno import Agent, RunResponse
    from agno.models.openai import OpenAI
    from agno.team import Team
    from agno.models.deepseek import DeepSeek
    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False
    print("⚠️ Agno框架未安装，将使用模拟实现")

# 导入专业智能体团队
try:
    from norma_professional_agents_team import (
        NormaMasterAgent, TechExpertAgent, CreativeDesignAgent,
        DataAnalysisAgent, KnowledgeManagementAgent, CommunicationAgent,
        NormaProfessionalTeam, Task, TaskType, TaskPriority, AgentStatus
    )
    PROFESSIONAL_TEAM_AVAILABLE = True
except ImportError:
    PROFESSIONAL_TEAM_AVAILABLE = False
    print("⚠️ 专业智能体团队未找到，将使用基础实现")

# =============================================================================
# 协作模式枚举和数据结构
# =============================================================================

class CollaborationMode(Enum):
    """协作模式枚举"""
    SERIAL = "serial"           # 串行协作
    PARALLEL = "parallel"       # 并行协作
    HYBRID = "hybrid"           # 混合协作

class TaskComplexity(Enum):
    """任务复杂度"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    CRITICAL = "critical"

@dataclass
class CollaborationTask:
    """协作任务"""
    id: str
    title: str
    description: str
    complexity: TaskComplexity
    estimated_duration: float  # 秒
    required_agents: List[str]
    dependencies: List[str] = None
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class CollaborationResult:
    """协作结果"""
    task_id: str
    mode: CollaborationMode
    success: bool
    start_time: datetime
    end_time: datetime
    duration: float
    agent_results: Dict[str, Any]
    final_result: Any
    performance_metrics: Dict[str, float]
    error_message: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_duration: float
    agent_utilization: Dict[str, float]
    efficiency_score: float
    quality_score: float
    resource_usage: Dict[str, float]
    bottleneck_agents: List[str]

# =============================================================================
# 协作模式基类
# =============================================================================

class BaseCollaborationMode:
    """协作模式基类"""
    
    def __init__(self, mode_name: str, agents: Dict[str, Any] = None):
        self.mode_name = mode_name
        self.agents = agents or {}
        self.logger = self._setup_logger()
        self.performance_history = []
        self.active_tasks = {}
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"NormaCollaboration_{self.mode_name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def execute_task(self, task: CollaborationTask) -> CollaborationResult:
        """执行协作任务 - 子类必须实现"""
        raise NotImplementedError("子类必须实现execute_task方法")
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        if not self.performance_history:
            return PerformanceMetrics(
                total_duration=0.0,
                agent_utilization={},
                efficiency_score=0.0,
                quality_score=0.0,
                resource_usage={},
                bottleneck_agents=[]
            )
        
        # 计算平均性能指标
        durations = [r.duration for r in self.performance_history]
        quality_scores = [r.performance_metrics.get('quality_score', 0) for r in self.performance_history]
        efficiency_scores = [r.performance_metrics.get('efficiency_score', 0) for r in self.performance_history]
        
        # Agent利用率统计
        agent_utilization = {}
        for agent_id in self.agents.keys():
            utilization_scores = []
            for result in self.performance_history:
                if agent_id in result.agent_results:
                    utilization_scores.append(result.agent_results[agent_id].get('utilization', 0))
            agent_utilization[agent_id] = statistics.mean(utilization_scores) if utilization_scores else 0
        
        # 识别瓶颈Agent
        bottleneck_agents = [agent_id for agent_id, util in agent_utilization.items() if util > 0.9]
        
        return PerformanceMetrics(
            total_duration=statistics.mean(durations),
            agent_utilization=agent_utilization,
            efficiency_score=statistics.mean(efficiency_scores),
            quality_score=statistics.mean(quality_scores),
            resource_usage={'cpu': 0.5, 'memory': 0.6, 'network': 0.3},  # 示例数据
            bottleneck_agents=bottleneck_agents
        )

# =============================================================================
# 1. 串行协作模式
# =============================================================================

class SerialCollaborationMode(BaseCollaborationMode):
    """串行协作模式 - 任务按顺序在各Agent间流转处理"""
    
    def __init__(self, agents: Dict[str, Any] = None):
        super().__init__("Serial", agents)
        self.task_queue = []
        self.completed_tasks = []
        
    async def execute_task(self, task: CollaborationTask) -> CollaborationResult:
        """串行执行任务"""
        start_time = datetime.now()
        self.logger.info(f"开始串行协作执行任务: {task.title}")
        
        agent_results = {}
        current_input = task.description
        
        try:
            # 按依赖顺序执行
            execution_order = self._determine_execution_order(task)
            
            for agent_id in execution_order:
                if agent_id not in self.agents:
                    continue
                    
                agent = self.agents[agent_id]
                self.logger.info(f"执行Agent: {agent_id}")
                
                # 模拟Agent处理
                result = await self._execute_agent_task(agent, agent_id, current_input, task)
                agent_results[agent_id] = result
                
                # 更新输入为下一个Agent的输入
                current_input = result.get('output', current_input)
                
                # 记录性能指标
                self._record_agent_performance(agent_id, result)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 计算性能指标
            performance_metrics = self._calculate_performance_metrics(task, agent_results, duration)
            
            result = CollaborationResult(
                task_id=task.id,
                mode=CollaborationMode.SERIAL,
                success=True,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                agent_results=agent_results,
                final_result=current_input,
                performance_metrics=performance_metrics
            )
            
            self.performance_history.append(result)
            self.logger.info(f"串行协作完成，耗时: {duration:.2f}秒")
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.error(f"串行协作执行失败: {str(e)}")
            
            return CollaborationResult(
                task_id=task.id,
                mode=CollaborationMode.SERIAL,
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                agent_results=agent_results,
                final_result=None,
                performance_metrics={},
                error_message=str(e)
            )
    
    def _determine_execution_order(self, task: CollaborationTask) -> List[str]:
        """确定执行顺序"""
        required_agents = task.required_agents.copy()
        
        # 主控Agent优先
        if "norma_master" in required_agents:
            required_agents.remove("norma_master")
            required_agents.insert(0, "norma_master")
        
        # 根据依赖关系调整顺序
        ordered_agents = []
        remaining_agents = required_agents.copy()
        
        # 简单依赖解析
        while remaining_agents:
            for agent_id in remaining_agents.copy():
                # 检查依赖是否已满足
                dependencies_met = True
                for dep in task.dependencies:
                    if dep in remaining_agents:
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    ordered_agents.append(agent_id)
                    remaining_agents.remove(agent_id)
                    break
            
            # 防止无限循环
            if len(ordered_agents) == len(required_agents):
                break
        
        return ordered_agents
    
    async def _execute_agent_task(self, agent: Any, agent_id: str, input_data: str, task: CollaborationTask) -> Dict[str, Any]:
        """执行单个Agent任务"""
        # 模拟处理时间
        processing_time = self._estimate_processing_time(agent_id, task.complexity)
        await asyncio.sleep(processing_time)
        
        # 模拟Agent处理结果
        result = {
            "agent_id": agent_id,
            "input": input_data,
            "output": f"[{agent_id}]处理完成: {input_data[:100]}...",
            "processing_time": processing_time,
            "utilization": 0.8,
            "quality_score": 0.85,
            "success": True
        }
        
        return result
    
    def _estimate_processing_time(self, agent_id: str, complexity: TaskComplexity) -> float:
        """估算处理时间"""
        base_times = {
            "norma_master": 2.0,
            "tech_expert": 3.0,
            "creative_design": 4.0,
            "data_analyst": 3.5,
            "knowledge_manager": 2.5,
            "communication_agent": 1.5
        }
        
        complexity_multipliers = {
            TaskComplexity.SIMPLE: 0.5,
            TaskComplexity.MEDIUM: 1.0,
            TaskComplexity.COMPLEX: 2.0,
            TaskComplexity.CRITICAL: 3.0
        }
        
        base_time = base_times.get(agent_id, 3.0)
        multiplier = complexity_multipliers.get(complexity, 1.0)
        
        return base_time * multiplier
    
    def _record_agent_performance(self, agent_id: str, result: Dict[str, Any]):
        """记录Agent性能"""
        if agent_id not in self.active_tasks:
            self.active_tasks[agent_id] = []
        
        self.active_tasks[agent_id].append({
            "timestamp": datetime.now(),
            "processing_time": result.get("processing_time", 0),
            "quality_score": result.get("quality_score", 0),
            "utilization": result.get("utilization", 0)
        })
    
    def _calculate_performance_metrics(self, task: CollaborationTask, agent_results: Dict[str, Any], duration: float) -> Dict[str, float]:
        """计算性能指标"""
        # 效率分数 = 预期时间 / 实际时间
        estimated_time = sum(self._estimate_processing_time(agent_id, task.complexity) for agent_id in agent_results.keys())
        efficiency_score = min(1.0, estimated_time / duration) if duration > 0 else 0
        
        # 质量分数 = Agent质量分数的平均值
        quality_scores = [result.get("quality_score", 0) for result in agent_results.values()]
        quality_score = statistics.mean(quality_scores) if quality_scores else 0
        
        return {
            "efficiency_score": efficiency_score,
            "quality_score": quality_score,
            "agent_count": len(agent_results),
            "total_processing_time": duration,
            "estimated_time": estimated_time
        }

# =============================================================================
# 2. 并行协作模式
# =============================================================================

class ParallelCollaborationMode(BaseCollaborationMode):
    """并行协作模式 - 多个Agent同时处理不同子任务"""
    
    def __init__(self, agents: Dict[str, Any] = None):
        super().__init__("Parallel", agents)
        self.max_concurrent_agents = 3  # 最大并发Agent数量
        
    async def execute_task(self, task: CollaborationTask) -> CollaborationResult:
        """并行执行任务"""
        start_time = datetime.now()
        self.logger.info(f"开始并行协作执行任务: {task.title}")
        
        agent_results = {}
        
        try:
            # 分解任务为子任务
            subtasks = self._decompose_task(task)
            
            # 并行执行子任务
            semaphore = asyncio.Semaphore(self.max_concurrent_agents)
            
            async def execute_subtask(subtask):
                async with semaphore:
                    return await self._execute_subtask_parallel(subtask, task)
            
            # 启动并行任务
            subtask_results = await asyncio.gather(
                *[execute_subtask(subtask) for subtask in subtasks],
                return_exceptions=True
            )
            
            # 收集结果
            for i, result in enumerate(subtask_results):
                if isinstance(result, Exception):
                    self.logger.error(f"子任务 {i} 执行失败: {str(result)}")
                    continue
                
                agent_id = result.get("agent_id")
                if agent_id:
                    agent_results[agent_id] = result
            
            # 整合最终结果
            final_result = self._integrate_parallel_results(agent_results, task)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 计算性能指标
            performance_metrics = self._calculate_parallel_performance_metrics(task, agent_results, duration)
            
            result = CollaborationResult(
                task_id=task.id,
                mode=CollaborationMode.PARALLEL,
                success=True,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                agent_results=agent_results,
                final_result=final_result,
                performance_metrics=performance_metrics
            )
            
            self.performance_history.append(result)
            self.logger.info(f"并行协作完成，耗时: {duration:.2f}秒")
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.error(f"并行协作执行失败: {str(e)}")
            
            return CollaborationResult(
                task_id=task.id,
                mode=CollaborationMode.PARALLEL,
                success=False,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                agent_results=agent_results,
                final_result=None,
                performance_metrics={},
                error_message=str(e)
            )
    
    def _decompose_task(self, task: CollaborationTask) -> List[Dict[str, Any]]:
        """分解任务为子任务"""
        subtasks = []
        
        for agent_id in task.required_agents:
            if agent_id not in self.agents:
                continue
                
            subtask = {
                "id": f"{task.id}_{agent_id}",
                "agent_id": agent_id,
                "description": f"由{agent_id}处理: {task.description}",
                "complexity": task.complexity,
                "estimated_duration": self._estimate_processing_time(agent_id, task.complexity)
            }
            subtasks.append(subtask)
        
        return subtasks
    
    async def _execute_subtask_parallel(self, subtask: Dict[str, Any], parent_task: CollaborationTask) -> Dict[str, Any]:
        """并行执行子任务"""
        agent_id = subtask["agent_id"]
        agent = self.agents[agent_id]
        
        self.logger.info(f"并行执行子任务: {subtask['id']} by {agent_id}")
        
        # 模拟处理时间
        processing_time = subtask["estimated_duration"]
        await asyncio.sleep(processing_time)
        
        # 模拟Agent处理结果
        result = {
            "agent_id": agent_id,
            "subtask_id": subtask["id"],
            "input": subtask["description"],
            "output": f"[{agent_id}]并行处理完成: {subtask['description'][:80]}...",
            "processing_time": processing_time,
            "utilization": 0.9,
            "quality_score": 0.88,
            "success": True
        }
        
        return result
    
    def _integrate_parallel_results(self, agent_results: Dict[str, Any], task: CollaborationTask) -> str:
        """整合并行结果"""
        integration_summary = f"任务: {task.title}\n\n"
        integration_summary += "并行处理结果:\n"
        
        for agent_id, result in agent_results.items():
            integration_summary += f"- {agent_id}: {result.get('output', '')}\n"
        
        integration_summary += f"\n并行协作模式完成，共{len(agent_results)}个Agent参与"
        
        return integration_summary
    
    def _calculate_parallel_performance_metrics(self, task: CollaborationTask, agent_results: Dict[str, Any], duration: float) -> Dict[str, float]:
        """计算并行模式性能指标"""
        # 并行效率 = 最长Agent时间 / 总时间
        processing_times = [result.get("processing_time", 0) for result in agent_results.values()]
        max_processing_time = max(processing_times) if processing_times else 0
        parallel_efficiency = min(1.0, max_processing_time / duration) if duration > 0 else 0
        
        # 质量分数
        quality_scores = [result.get("quality_score", 0) for result in agent_results.values()]
        quality_score = statistics.mean(quality_scores) if quality_scores else 0
        
        # 资源利用率
        utilization_scores = [result.get("utilization", 0) for result in agent_results.values()]
        avg_utilization = statistics.mean(utilization_scores) if utilization_scores else 0
        
        return {
            "parallel_efficiency": parallel_efficiency,
            "quality_score": quality_score,
            "agent_count": len(agent_results),
            "avg_utilization": avg_utilization,
            "max_processing_time": max_processing_time,
            "total_duration": duration
        }

# =============================================================================
# 3. 混合协作模式
# =============================================================================

class HybridCollaborationMode(BaseCollaborationMode):
    """混合协作模式 - 智能判断并动态选择串行或并行"""
    
    def __init__(self, agents: Dict[str, Any] = None):
        super().__init__("Hybrid", agents)
        self.serial_mode = SerialCollaborationMode(agents)
        self.parallel_mode = ParallelCollaborationMode(agents)
        self.decision_history = []
        
    async def execute_task(self, task: CollaborationTask) -> CollaborationResult:
        """混合执行任务"""
        start_time = datetime.now()
        self.logger.info(f"开始混合协作执行任务: {task.title}")
        
        try:
            # 智能决策选择协作模式
            selected_mode = await self._intelligent_mode_selection(task)
            
            self.logger.info(f"智能选择协作模式: {selected_mode}")
            
            # 根据选择的模式执行任务
            if selected_mode == CollaborationMode.SERIAL:
                result = await self.serial_mode.execute_task(task)
            elif selected_mode == CollaborationMode.PARALLEL:
                result = await self.parallel_mode.execute_task(task)
            else:
                # 默认使用串行模式
                result = await self.serial_mode.execute_task(task)
            
            # 添加混合模式的特殊指标
            result.mode = CollaborationMode.HYBRID
            result.performance_metrics["mode_selection_accuracy"] = self._calculate_selection_accuracy(task, selected_mode)
            result.performance_metrics["adaptive_efficiency"] = self._calculate_adaptive_efficiency(result)
            
            self.performance_history.append(result)
            self.logger.info(f"混合协作完成，选择模式: {selected_mode}")
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.error(f"混合协作执行失败: {str(e)}")
            
            return CollaborationResult(
                task_id=task.id,
                mode=CollaborationMode.HYBRID,
                success=False,
                start_time=start_time,
                end_time=datetime.now(),
                duration=duration,
                agent_results={},
                final_result=None,
                performance_metrics={},
                error_message=str(e)
            )
    
    async def _intelligent_mode_selection(self, task: CollaborationTask) -> CollaborationMode:
        """智能模式选择"""
        # 决策因素评分
        factors = {
            "task_complexity": self._evaluate_complexity(task),
            "agent_dependencies": self._evaluate_dependencies(task),
            "resource_availability": self._evaluate_resources(),
            "time_constraints": self._evaluate_time_constraints(task),
            "quality_requirements": self._evaluate_quality_requirements(task)
        }
        
        # 计算串行和并行模式的得分
        serial_score = self._calculate_serial_score(factors, task)
        parallel_score = self._calculate_parallel_score(factors, task)
        
        # 记录决策过程
        decision = {
            "task_id": task.id,
            "factors": factors,
            "serial_score": serial_score,
            "parallel_score": parallel_score,
            "selected_mode": CollaborationMode.SERIAL if serial_score > parallel_score else CollaborationMode.PARALLEL,
            "timestamp": datetime.now()
        }
        
        self.decision_history.append(decision)
        
        return decision["selected_mode"]
    
    def _evaluate_complexity(self, task: CollaborationTask) -> float:
        """评估任务复杂度"""
        complexity_scores = {
            TaskComplexity.SIMPLE: 0.2,
            TaskComplexity.MEDIUM: 0.5,
            TaskComplexity.COMPLEX: 0.8,
            TaskComplexity.CRITICAL: 1.0
        }
        return complexity_scores.get(task.complexity, 0.5)
    
    def _evaluate_dependencies(self, task: CollaborationTask) -> float:
        """评估Agent依赖关系"""
        dependency_ratio = len(task.dependencies) / max(len(task.required_agents), 1)
        return min(1.0, dependency_ratio)
    
    def _evaluate_resources(self) -> float:
        """评估资源可用性"""
        # 模拟资源评估
        available_agents = len([agent for agent in self.agents.values() if hasattr(agent, 'status') and agent.status == AgentStatus.IDLE])
        total_agents = len(self.agents)
        return available_agents / max(total_agents, 1)
    
    def _evaluate_time_constraints(self, task: CollaborationTask) -> float:
        """评估时间约束"""
        # 任务紧急度评估
        urgency_scores = {
            TaskPriority.LOW: 0.2,
            TaskPriority.MEDIUM: 0.5,
            TaskPriority.HIGH: 0.8,
            TaskPriority.CRITICAL: 1.0
        }
        return urgency_scores.get(task.priority, 0.5)
    
    def _evaluate_quality_requirements(self, task: CollaborationTask) -> float:
        """评估质量要求"""
        # 基于复杂度和优先级评估质量要求
        complexity_weight = self._evaluate_complexity(task)
        priority_weight = self._evaluate_time_constraints(task)
        return (complexity_weight + priority_weight) / 2
    
    def _calculate_serial_score(self, factors: Dict[str, float], task: CollaborationTask) -> float:
        """计算串行模式得分"""
        # 串行模式适合高依赖、低资源的情况
        dependency_factor = factors["agent_dependencies"] * 0.3
        resource_factor = (1 - factors["resource_availability"]) * 0.2
        complexity_factor = factors["task_complexity"] * 0.2
        time_factor = (1 - factors["time_constraints"]) * 0.15
        quality_factor = factors["quality_requirements"] * 0.15
        
        return dependency_factor + resource_factor + complexity_factor + time_factor + quality_factor
    
    def _calculate_parallel_score(self, factors: Dict[str, float], task: CollaborationTask) -> float:
        """计算并行模式得分"""
        # 并行模式适合低依赖、高资源的情况
        dependency_factor = (1 - factors["agent_dependencies"]) * 0.3
        resource_factor = factors["resource_availability"] * 0.25
        complexity_factor = (1 - factors["task_complexity"]) * 0.2
        time_factor = factors["time_constraints"] * 0.15
        quality_factor = (1 - factors["quality_requirements"]) * 0.1
        
        return dependency_factor + resource_factor + complexity_factor + time_factor + quality_factor
    
    def _calculate_selection_accuracy(self, task: CollaborationTask, selected_mode: CollaborationMode) -> float:
        """计算模式选择准确性"""
        # 基于历史表现计算准确性
        if not self.decision_history:
            return 0.5
        
        # 简化的准确性计算
        recent_decisions = self.decision_history[-10:]  # 最近10次决策
        successful_decisions = sum(1 for d in recent_decisions if d.get("success", False))
        
        return successful_decisions / max(len(recent_decisions), 1)
    
    def _calculate_adaptive_efficiency(self, result: CollaborationResult) -> float:
        """计算自适应效率"""
        if not result.performance_metrics:
            return 0.5
        
        # 基于实际性能计算自适应效率
        efficiency = result.performance_metrics.get("efficiency_score", 0.5)
        quality = result.performance_metrics.get("quality_score", 0.5)
        
        return (efficiency + quality) / 2

# =============================================================================
# Team协作管理器
# =============================================================================

class NormaTeamCollaborationManager:
    """诺玛Team协作管理器"""
    
    def __init__(self):
        self.agents = {}
        self.collaboration_modes = {}
        self.active_sessions = {}
        self.performance_history = []
        self.logger = self._setup_logger()
        
        # 初始化专业智能体
        self._initialize_agents()
        
        # 初始化协作模式
        self._initialize_collaboration_modes()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("NormaTeamCollaborationManager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_agents(self):
        """初始化专业智能体"""
        if PROFESSIONAL_TEAM_AVAILABLE:
            try:
                # 创建专业智能体实例
                self.agents = {
                    "norma_master": NormaMasterAgent(),
                    "tech_expert": TechExpertAgent(),
                    "creative_design": CreativeDesignAgent(),
                    "data_analyst": DataAnalysisAgent(),
                    "knowledge_manager": KnowledgeManagementAgent(),
                    "communication_agent": CommunicationAgent()
                }
                self.logger.info("专业智能体团队初始化完成")
            except Exception as e:
                self.logger.error(f"专业智能体初始化失败: {str(e)}")
                self._create_mock_agents()
        else:
            self._create_mock_agents()
    
    def _create_mock_agents(self):
        """创建模拟智能体"""
        self.agents = {
            "norma_master": {"id": "norma_master", "status": "idle", "capabilities": ["coordination"]},
            "tech_expert": {"id": "tech_expert", "status": "idle", "capabilities": ["technical_analysis"]},
            "creative_design": {"id": "creative_design", "status": "idle", "capabilities": ["creative_design"]},
            "data_analyst": {"id": "data_analyst", "status": "idle", "capabilities": ["data_analysis"]},
            "knowledge_manager": {"id": "knowledge_manager", "status": "idle", "capabilities": ["knowledge_management"]},
            "communication_agent": {"id": "communication_agent", "status": "idle", "capabilities": ["communication"]}
        }
        self.logger.info("模拟智能体创建完成")
    
    def _initialize_collaboration_modes(self):
        """初始化协作模式"""
        self.collaboration_modes = {
            CollaborationMode.SERIAL: SerialCollaborationMode(self.agents),
            CollaborationMode.PARALLEL: ParallelCollaborationMode(self.agents),
            CollaborationMode.HYBRID: HybridCollaborationMode(self.agents)
        }
        self.logger.info("协作模式初始化完成")
    
    async def execute_collaboration_task(
        self, 
        task: CollaborationTask, 
        mode: CollaborationMode = CollaborationMode.HYBRID
    ) -> CollaborationResult:
        """执行协作任务"""
        session_id = str(uuid.uuid4())
        self.active_sessions[session_id] = {
            "task": task,
            "mode": mode,
            "start_time": datetime.now(),
            "status": "running"
        }
        
        try:
            self.logger.info(f"开始协作任务: {task.title} (模式: {mode.value})")
            
            # 执行协作任务
            collaboration_mode = self.collaboration_modes[mode]
            result = await collaboration_mode.execute_task(task)
            
            # 更新会话状态
            self.active_sessions[session_id]["status"] = "completed"
            self.active_sessions[session_id]["result"] = result
            
            # 记录到历史
            self.performance_history.append(result)
            
            self.logger.info(f"协作任务完成: {task.title}")
            
            return result
            
        except Exception as e:
            self.active_sessions[session_id]["status"] = "failed"
            self.active_sessions[session_id]["error"] = str(e)
            self.logger.error(f"协作任务失败: {str(e)}")
            raise
    
    async def compare_collaboration_modes(self, task: CollaborationTask) -> Dict[str, CollaborationResult]:
        """比较不同协作模式的性能"""
        results = {}
        
        for mode in CollaborationMode:
            try:
                result = await self.execute_collaboration_task(task, mode)
                results[mode.value] = result
            except Exception as e:
                self.logger.error(f"模式 {mode.value} 执行失败: {str(e)}")
                results[mode.value] = None
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.performance_history:
            return {"message": "暂无性能数据"}
        
        # 按模式分组统计
        mode_stats = {}
        for result in self.performance_history:
            mode = result.mode.value
            if mode not in mode_stats:
                mode_stats[mode] = {
                    "count": 0,
                    "total_duration": 0,
                    "success_rate": 0,
                    "avg_quality": 0,
                    "avg_efficiency": 0
                }
            
            stats = mode_stats[mode]
            stats["count"] += 1
            stats["total_duration"] += result.duration
            
            if result.success:
                stats["success_rate"] += 1
            
            quality = result.performance_metrics.get("quality_score", 0)
            efficiency = result.performance_metrics.get("efficiency_score", 0)
            stats["avg_quality"] += quality
            stats["avg_efficiency"] += efficiency
        
        # 计算平均值
        for mode, stats in mode_stats.items():
            if stats["count"] > 0:
                stats["success_rate"] = stats["success_rate"] / stats["count"]
                stats["avg_quality"] = stats["avg_quality"] / stats["count"]
                stats["avg_efficiency"] = stats["avg_efficiency"] / stats["count"]
                stats["avg_duration"] = stats["total_duration"] / stats["count"]
        
        return {
            "total_tasks": len(self.performance_history),
            "mode_statistics": mode_stats,
            "overall_success_rate": sum(1 for r in self.performance_history if r.success) / len(self.performance_history),
            "active_sessions": len([s for s in self.active_sessions.values() if s["status"] == "running"])
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """获取Agent状态"""
        status = {}
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'status'):
                status[agent_id] = {
                    "status": agent.status.value if hasattr(agent.status, 'value') else str(agent.status),
                    "capabilities": getattr(agent, 'capabilities', [])
                }
            else:
                status[agent_id] = {
                    "status": "idle",
                    "capabilities": agent.get("capabilities", [])
                }
        return status

# =============================================================================
# 演示和测试脚本
# =============================================================================

async def demo_collaboration_modes():
    """演示三种协作模式"""
    print("🎯 诺玛Team协作模式演示")
    print("=" * 60)
    
    # 初始化协作管理器
    manager = NormaTeamCollaborationManager()
    
    # 创建测试任务
    test_tasks = [
        CollaborationTask(
            id="task_1",
            title="系统性能优化咨询",
            description="分析当前诺玛Agent系统性能瓶颈，提供优化建议",
            complexity=TaskComplexity.COMPLEX,
            estimated_duration=300.0,
            required_agents=["norma_master", "tech_expert", "data_analyst"],
            dependencies=[],
            priority=TaskPriority.HIGH
        ),
        CollaborationTask(
            id="task_2", 
            title="品牌视觉设计优化",
            description="设计诺玛Agent的新版界面和视觉元素",
            complexity=TaskComplexity.MEDIUM,
            estimated_duration=240.0,
            required_agents=["creative_design", "norma_master", "communication_agent"],
            dependencies=[],
            priority=TaskPriority.MEDIUM
        ),
        CollaborationTask(
            id="task_3",
            title="知识库升级项目",
            description="升级诺玛Agent知识库，集成最新技术文档",
            complexity=TaskComplexity.CRITICAL,
            estimated_duration=480.0,
            required_agents=["knowledge_manager", "tech_expert", "norma_master", "data_analyst"],
            dependencies=["tech_expert"],
            priority=TaskPriority.CRITICAL
        )
    ]
    
    print(f"📋 创建了 {len(test_tasks)} 个测试任务")
    
    # 演示每种协作模式
    for i, task in enumerate(test_tasks, 1):
        print(f"\n🔄 演示任务 {i}: {task.title}")
        print("-" * 40)
        
        # 演示混合模式（智能选择）
        print("🤖 使用混合协作模式...")
        result = await manager.execute_collaboration_task(task, CollaborationMode.HYBRID)
        
        if result.success:
            print(f"✅ 任务完成")
            print(f"   耗时: {result.duration:.2f}秒")
            print(f"   效率分数: {result.performance_metrics.get('efficiency_score', 0):.2f}")
            print(f"   质量分数: {result.performance_metrics.get('quality_score', 0):.2f}")
        else:
            print(f"❌ 任务失败: {result.error_message}")
    
    # 比较不同模式性能
    print(f"\n📊 模式性能比较")
    print("-" * 40)
    
    comparison_task = test_tasks[0]  # 使用第一个任务进行比较
    mode_results = await manager.compare_collaboration_modes(comparison_task)
    
    for mode_name, result in mode_results.items():
        if result and result.success:
            print(f"{mode_name.upper()}: {result.duration:.2f}秒, 效率:{result.performance_metrics.get('efficiency_score', 0):.2f}")
        else:
            print(f"{mode_name.upper()}: 执行失败")
    
    # 显示性能摘要
    print(f"\n📈 性能摘要")
    print("-" * 40)
    summary = manager.get_performance_summary()
    print(f"总任务数: {summary['total_tasks']}")
    print(f"整体成功率: {summary['overall_success_rate']:.2%}")
    print(f"活跃会话: {summary['active_sessions']}")
    
    # 显示Agent状态
    print(f"\n🤖 Agent状态")
    print("-" * 40)
    agent_status = manager.get_agent_status()
    for agent_id, status in agent_status.items():
        print(f"{agent_id}: {status['status']} (能力: {len(status['capabilities'])})")
    
    print(f"\n✨ 协作模式演示完成！")

async def test_specific_modes():
    """测试特定协作模式"""
    print("\n🧪 特定协作模式测试")
    print("=" * 60)
    
    manager = NormaTeamCollaborationManager()
    
    # 测试串行模式
    serial_task = CollaborationTask(
        id="serial_test",
        title="串行协作测试",
        description="测试串行模式的任务流转",
        complexity=TaskComplexity.MEDIUM,
        estimated_duration=180.0,
        required_agents=["norma_master", "tech_expert", "creative_design"],
        dependencies=["norma_master"],
        priority=TaskPriority.MEDIUM
    )
    
    print("🔄 测试串行协作模式...")
    serial_result = await manager.execute_collaboration_task(serial_task, CollaborationMode.SERIAL)
    
    if serial_result.success:
        print(f"✅ 串行模式成功")
        print(f"   执行时间: {serial_result.duration:.2f}秒")
        print(f"   参与Agent: {list(serial_result.agent_results.keys())}")
    
    # 测试并行模式
    parallel_task = CollaborationTask(
        id="parallel_test",
        title="并行协作测试", 
        description="测试并行模式的并发处理",
        complexity=TaskComplexity.SIMPLE,
        estimated_duration=120.0,
        required_agents=["tech_expert", "creative_design", "data_analyst"],
        dependencies=[],
        priority=TaskPriority.LOW
    )
    
    print("\n🔄 测试并行协作模式...")
    parallel_result = await manager.execute_collaboration_task(parallel_task, CollaborationMode.PARALLEL)
    
    if parallel_result.success:
        print(f"✅ 并行模式成功")
        print(f"   执行时间: {parallel_result.duration:.2f}秒")
        print(f"   并发Agent: {list(parallel_result.agent_results.keys())}")
    
    # 测试混合模式
    hybrid_task = CollaborationTask(
        id="hybrid_test",
        title="混合协作测试",
        description="测试混合模式的智能选择",
        complexity=TaskComplexity.COMPLEX,
        estimated_duration=360.0,
        required_agents=["norma_master", "tech_expert", "creative_design", "data_analyst"],
        dependencies=["tech_expert"],
        priority=TaskPriority.HIGH
    )
    
    print("\n🔄 测试混合协作模式...")
    hybrid_result = await manager.execute_collaboration_task(hybrid_task, CollaborationMode.HYBRID)
    
    if hybrid_result.success:
        print(f"✅ 混合模式成功")
        print(f"   执行时间: {hybrid_result.duration:.2f}秒")
        print(f"   模式选择: {hybrid_result.mode.value}")
        print(f"   自适应效率: {hybrid_result.performance_metrics.get('adaptive_efficiency', 0):.2f}")

# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("🚀 诺玛Team协作模式系统启动")
    print("基于Agno框架的三种智能体协作模式")
    print("作者: 皇")
    print("版本: 1.0.0")
    print("=" * 60)
    
    # 运行演示
    asyncio.run(demo_collaboration_modes())
    
    # 运行特定测试
    asyncio.run(test_specific_modes())
    
    print("\n🎉 Team协作模式演示完成！")
