"""
任务规划器

整合所有任务规划组件的主要入口点
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime
import uuid

from .models import Task, TaskStatus, TaskPriority, TaskExecutionPlan, TaskDependency
from .task_decomposer import TaskDecomposer
from .dependency_analyzer import DependencyAnalyzer
from .scheduler import TaskScheduler, SchedulingStrategy
from .state_tracker import StateTracker
from .recovery_manager import RecoveryManager, RetryConfig
from ..llm.interfaces.llm_interface import LLMInterface


class TaskPlanner:
    """任务规划器主类"""
    
    def __init__(
        self,
        llm_interface: LLMInterface,
        max_parallel_tasks: int = 5,
        scheduling_strategy: SchedulingStrategy = SchedulingStrategy.HYBRID,
        enable_recovery: bool = True,
        enable_monitoring: bool = True
    ):
        # 核心组件
        self.llm = llm_interface
        self.task_decomposer = TaskDecomposer(llm_interface)
        self.dependency_analyzer = DependencyAnalyzer()
        self.scheduler = TaskScheduler(
            max_parallel_tasks=max_parallel_tasks,
            strategy=scheduling_strategy
        )
        self.state_tracker = StateTracker()
        self.recovery_manager = RecoveryManager(self.state_tracker) if enable_recovery else None
        
        # 配置
        self.enable_recovery = enable_recovery
        self.enable_monitoring = enable_monitoring
        
        # 任务管理
        self.active_plans: Dict[str, TaskExecutionPlan] = {}
        self.completed_plans: Dict[str, TaskExecutionPlan] = {}
        self.failed_plans: Dict[str, TaskExecutionPlan] = {}
        
        # 回调函数
        self.callbacks = {
            "on_plan_start": None,
            "on_plan_progress": None,
            "on_plan_complete": None,
            "on_plan_fail": None,
            "on_task_start": None,
            "on_task_complete": None,
            "on_task_fail": None,
            "on_recovery_start": None,
            "on_recovery_complete": None
        }
        
        # 设置回调
        self._setup_callbacks()
        
        # 启动监控
        if self.enable_monitoring:
            self.state_tracker.start_monitoring()
    
    def _setup_callbacks(self):
        """设置组件间的回调"""
        # 调度器回调
        self.scheduler.on_task_start = self._handle_task_start
        self.scheduler.on_task_complete = self._handle_task_complete
        self.scheduler.on_task_fail = self._handle_task_fail
        self.scheduler.on_plan_complete = self._handle_plan_complete
        
        # 状态跟踪器回调
        self.state_tracker.on_task_status_change = self._handle_task_status_change
        self.state_tracker.on_progress_update = self._handle_progress_update
        self.state_tracker.on_milestone_reached = self._handle_milestone_reached
        
        # 恢复管理器回调
        if self.recovery_manager:
            self.recovery_manager.on_recovery_start = self._handle_recovery_start
            self.recovery_manager.on_recovery_complete = self._handle_recovery_complete
            self.recovery_manager.on_recovery_failed = self._handle_recovery_failed
    
    async def create_plan(
        self,
        task_description: str,
        task_type: str = "generic",
        parameters: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_depth: int = 3,
        min_task_duration: int = 5,
        max_subtasks: int = 10
    ) -> TaskExecutionPlan:
        """
        创建任务执行计划
        
        Args:
            task_description: 任务描述
            task_type: 任务类型
            parameters: 任务参数
            priority: 任务优先级
            max_depth: 最大分解深度
            min_task_duration: 最小任务时长（分钟）
            max_subtasks: 最大子任务数量
            
        Returns:
            任务执行计划
        """
        # 创建主任务
        main_task = Task(
            name=f"主任务 - {task_description[:50]}",
            description=task_description,
            task_type=task_type,
            priority=priority,
            parameters=parameters or {},
            metadata={
                "created_by": "task_planner",
                "creation_time": datetime.now().isoformat()
            }
        )
        
        # 注册任务到状态跟踪器
        self.state_tracker.register_task(main_task)
        
        # 分解任务
        plan = await self.task_decomposer.decompose_task(
            main_task,
            max_depth=max_depth,
            min_task_duration=min_task_duration,
            max_subtasks=max_subtasks
        )
        
        # 注册执行计划
        self.state_tracker.register_execution_plan(plan)
        self.active_plans[plan.id] = plan
        
        # 分析依赖关系
        dependency_analysis = self.dependency_analyzer.analyze_dependencies(plan)
        
        # 添加依赖分析结果到计划元数据
        plan.metadata["dependency_analysis"] = dependency_analysis
        
        return plan
    
    async def execute_plan(
        self,
        plan: TaskExecutionPlan,
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, Any]:
        """
        执行任务计划
        
        Args:
            plan: 任务执行计划
            callbacks: 可选的回调函数
            
        Returns:
            执行结果
        """
        if callbacks:
            self.callbacks.update(callbacks)
        
        try:
            # 触发计划开始回调
            if self.callbacks["on_plan_start"]:
                await self._safe_callback(self.callbacks["on_plan_start"], plan)
            
            # 更新计划状态
            plan.status = TaskStatus.RUNNING
            
            # 执行计划
            result = await self.scheduler.execute_plan(plan, self.callbacks)
            
            # 处理执行结果
            if result.get("success", False):
                plan.status = TaskStatus.COMPLETED
                self.completed_plans[plan.id] = plan
                
                if self.callbacks["on_plan_complete"]:
                    await self._safe_callback(self.callbacks["on_plan_complete"], plan, result)
            else:
                plan.status = TaskStatus.FAILED
                self.failed_plans[plan.id] = plan
                
                if self.callbacks["on_plan_fail"]:
                    await self._safe_callback(self.callbacks["on_plan_fail"], plan, result)
            
            # 从活跃计划中移除
            if plan.id in self.active_plans:
                del self.active_plans[plan.id]
            
            return result
            
        except Exception as e:
            plan.status = TaskStatus.FAILED
            if self.callbacks["on_plan_fail"]:
                await self._safe_callback(self.callbacks["on_plan_fail"], plan, {"error": str(e)})
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_and_execute_plan(
        self,
        task_description: str,
        task_type: str = "generic",
        parameters: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        callbacks: Optional[Dict[str, Callable]] = None,
        **decompose_kwargs
    ) -> Dict[str, Any]:
        """
        创建并执行任务计划（一步完成）
        
        Args:
            task_description: 任务描述
            task_type: 任务类型
            parameters: 任务参数
            priority: 任务优先级
            callbacks: 回调函数
            **decompose_kwargs: 分解参数
            
        Returns:
            执行结果
        """
        # 创建计划
        plan = await self.create_plan(
            task_description=task_description,
            task_type=task_type,
            parameters=parameters,
            priority=priority,
            **decompose_kwargs
        )
        
        # 执行计划
        result = await self.execute_plan(plan, callbacks)
        
        return result
    
    def add_task_dependency(
        self,
        plan: TaskExecutionPlan,
        source_task_id: str,
        target_task_id: str,
        dependency_type: str = "finish_to_start",
        delay: int = 0,
        condition: Optional[str] = None
    ) -> bool:
        """添加任务依赖关系"""
        dependency = TaskDependency(
            source_task_id=source_task_id,
            target_task_id=target_task_id,
            dependency_type=dependency_type,
            delay=delay,
            condition=condition
        )
        
        plan.add_dependency(dependency)
        
        # 更新依赖分析
        plan.metadata["dependency_analysis"] = self.dependency_analyzer.analyze_dependencies(plan)
        
        return True
    
    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """获取计划状态"""
        plan = None
        if plan_id in self.active_plans:
            plan = self.active_plans[plan_id]
        elif plan_id in self.completed_plans:
            plan = self.completed_plans[plan_id]
        elif plan_id in self.failed_plans:
            plan = self.failed_plans[plan_id]
        
        if not plan:
            return None
        
        # 获取详细状态
        execution_status = self.scheduler.get_execution_status(plan)
        plan_status = self.state_tracker.get_execution_plan_status(plan_id)
        
        return {
            "plan": plan.to_dict() if hasattr(plan, 'to_dict') else {
                "id": plan.id,
                "name": plan.name,
                "status": plan.status.value,
                "progress": plan.progress
            },
            "execution_status": execution_status,
            "detailed_status": plan_status,
            "dependency_analysis": plan.metadata.get("dependency_analysis", {})
        }
    
    def get_all_plans_status(self) -> Dict[str, Any]:
        """获取所有计划状态"""
        return {
            "active_plans": {
                plan_id: self.get_plan_status(plan_id)
                for plan_id, plan in self.active_plans.items()
            },
            "completed_plans": {
                plan_id: self.get_plan_status(plan_id)
                for plan_id, plan in self.completed_plans.items()
            },
            "failed_plans": {
                plan_id: self.get_plan_status(plan_id)
                for plan_id, plan in self.failed_plans.items()
            },
            "summary": {
                "total_active": len(self.active_plans),
                "total_completed": len(self.completed_plans),
                "total_failed": len(self.failed_plans)
            }
        }
    
    def cancel_plan(self, plan_id: str) -> bool:
        """取消计划"""
        if plan_id not in self.active_plans:
            return False
        
        plan = self.active_plans[plan_id]
        plan.status = TaskStatus.CANCELLED
        
        # 取消所有运行中的任务
        for task in plan.tasks:
            if task.status == TaskStatus.RUNNING:
                self.scheduler.cancel_task(task.id)
        
        # 移到失败计划中
        self.failed_plans[plan_id] = plan
        del self.active_plans[plan_id]
        
        return True
    
    def pause_plan(self, plan_id: str) -> bool:
        """暂停计划"""
        if plan_id not in self.active_plans:
            return False
        
        plan = self.active_plans[plan_id]
        plan.status = TaskStatus.PENDING
        
        # 暂停调度器
        self.scheduler.pause_scheduler()
        
        return True
    
    def resume_plan(self, plan_id: str) -> bool:
        """恢复计划"""
        if plan_id not in self.active_plans:
            return False
        
        plan = self.active_plans[plan_id]
        plan.status = TaskStatus.RUNNING
        
        # 恢复调度器
        self.scheduler.resume_scheduler()
        
        return True
    
    def export_plan(self, plan_id: str, format_type: str = "json") -> Optional[str]:
        """导出计划"""
        plan_status = self.get_plan_status(plan_id)
        if not plan_status:
            return None
        
        if format_type == "json":
            return json.dumps(plan_status, ensure_ascii=False, indent=2, default=str)
        else:
            # 可以扩展其他格式
            return self.export_plan(plan_id, "json")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "task_planner": {
                "total_plans_created": len(self.completed_plans) + len(self.failed_plans) + len(self.active_plans),
                "success_rate": len(self.completed_plans) / max(1, len(self.completed_plans) + len(self.failed_plans)),
                "active_plans_count": len(self.active_plans)
            },
            "state_tracker": self.state_tracker.performance_metrics,
            "recovery_manager": self.recovery_manager.get_recovery_statistics() if self.recovery_manager else None,
            "scheduler": {
                "max_parallel_tasks": self.scheduler.max_parallel_tasks,
                "strategy": self.scheduler.strategy.value
            }
        }
    
    def cleanup(self):
        """清理资源"""
        # 停止监控
        if self.enable_monitoring:
            self.state_tracker.stop_monitoring()
        
        # 停止调度器
        self.scheduler.stop_scheduler()
        
        # 清理数据
        self.active_plans.clear()
    
    # 内部回调处理方法
    async def _handle_task_start(self, task: Task):
        """处理任务开始"""
        if self.callbacks["on_task_start"]:
            await self._safe_callback(self.callbacks["on_task_start"], task)
    
    async def _handle_task_complete(self, task: Task):
        """处理任务完成"""
        if self.callbacks["on_task_complete"]:
            await self._safe_callback(self.callbacks["on_task_complete"], task)
    
    async def _handle_task_fail(self, task: Task, error: str):
        """处理任务失败"""
        if self.callbacks["on_task_fail"]:
            await self._safe_callback(self.callbacks["on_task_fail"], task, error)
        
        # 启动恢复流程
        if self.enable_recovery and self.recovery_manager:
            try:
                await self.recovery_manager.handle_task_failure(
                    task, Exception(error)
                )
            except Exception as e:
                print(f"恢复流程启动失败: {e}")
    
    async def _handle_plan_complete(self, plan: TaskExecutionPlan):
        """处理计划完成"""
        if self.callbacks["on_plan_complete"]:
            await self._safe_callback(self.callbacks["on_plan_complete"], plan)
    
    async def _handle_task_status_change(self, task: Task, old_status: TaskStatus, new_status: TaskStatus):
        """处理任务状态变更"""
        # 可以在这里添加自定义逻辑
        pass
    
    async def _handle_progress_update(self, task: Task, old_progress: float, new_progress: float):
        """处理进度更新"""
        # 检查是否达到进度里程碑
        milestones = [0.25, 0.5, 0.75, 1.0]
        for milestone in milestones:
            if old_progress < milestone <= new_progress:
                if self.callbacks["on_plan_progress"]:
                    await self._safe_callback(
                        self.callbacks["on_plan_progress"], 
                        task, 
                        milestone, 
                        new_progress
                    )
                break
    
    async def _handle_milestone_reached(self, milestone: str, completion_rate: float):
        """处理里程碑到达"""
        if self.callbacks["on_plan_progress"]:
            await self._safe_callback(self.callbacks["on_plan_progress"], None, milestone, completion_rate)
    
    async def _handle_recovery_start(self, recovery_plan):
        """处理恢复开始"""
        if self.callbacks["on_recovery_start"]:
            await self._safe_callback(self.callbacks["on_recovery_start"], recovery_plan)
    
    async def _handle_recovery_complete(self, recovery_plan, message: str):
        """处理恢复完成"""
        if self.callbacks["on_recovery_complete"]:
            await self._safe_callback(self.callbacks["on_recovery_complete"], recovery_plan, message)
    
    async def _handle_recovery_failed(self, recovery_plan, message: str):
        """处理恢复失败"""
        if self.callbacks["on_recovery_failed"]:
            await self._safe_callback(self.callbacks["on_recovery_failed"], recovery_plan, message)
    
    def _safe_callback(self, callback: Callable, *args, **kwargs):
        """安全执行回调函数"""
        try:
            if asyncio.iscoroutinefunction(callback):
                asyncio.create_task(callback(*args, **kwargs))
            else:
                callback(*args, **kwargs)
        except Exception as e:
            print(f"回调函数执行失败: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()