"""
任务调度器

实现任务执行计划和调度系统，支持多种调度策略
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import threading
import uuid

from .models import Task, TaskStatus, TaskExecutionPlan, TaskPriority
from .dependency_analyzer import DependencyAnalyzer


class SchedulingStrategy(Enum):
    """调度策略"""
    FIFO = "fifo"                    # 先进先出
    PRIORITY = "priority"            # 优先级调度
    SHORTEST_JOB_FIRST = "sjf"       # 最短作业优先
    CRITICAL_PATH = "critical_path"  # 关键路径优先
    HYBRID = "hybrid"               # 混合策略


class TaskScheduler:
    """任务调度器"""
    
    def __init__(
        self,
        max_parallel_tasks: int = 5,
        strategy: SchedulingStrategy = SchedulingStrategy.HYBRID,
        task_executor: Optional[Callable] = None
    ):
        self.max_parallel_tasks = max_parallel_tasks
        self.strategy = strategy
        self.task_executor = task_executor or self._default_task_executor
        
        # 任务状态
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        self.task_results: Dict[str, Any] = {}
        
        # 调度状态
        self.is_running = False
        self.scheduler_thread = None
        self.lock = threading.Lock()
        
        # 依赖分析器
        self.dependency_analyzer = DependencyAnalyzer()
        
        # 回调函数
        self.on_task_start: Optional[Callable] = None
        self.on_task_complete: Optional[Callable] = None
        self.on_task_fail: Optional[Callable] = None
        self.on_plan_complete: Optional[Callable] = None
    
    async def execute_plan(
        self,
        plan: TaskExecutionPlan,
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, Any]:
        """
        执行任务计划
        
        Args:
            plan: 任务执行计划
            callbacks: 回调函数字典
            
        Returns:
            执行结果
        """
        if callbacks:
            self.on_task_start = callbacks.get('on_task_start')
            self.on_task_complete = callbacks.get('on_task_complete')
            self.on_task_fail = callbacks.get('on_task_fail')
            self.on_plan_complete = callbacks.get('on_plan_complete')
        
        # 验证计划
        is_valid, error_messages = self.dependency_analyzer.validate_execution_plan(plan)
        if not is_valid:
            return {
                "success": False,
                "error": "执行计划无效",
                "details": error_messages
            }
        
        # 分析依赖关系
        dependency_analysis = self.dependency_analyzer.analyze_dependencies(plan)
        
        # 开始执行
        self.is_running = True
        plan.status = TaskStatus.RUNNING
        
        try:
            # 根据调度策略执行任务
            if self.strategy == SchedulingStrategy.PARALLEL:
                result = await self._execute_parallel(plan, dependency_analysis)
            elif self.strategy == SchedulingStrategy.CRITICAL_PATH:
                result = await self._execute_critical_path(plan, dependency_analysis)
            else:
                result = await self._execute_hybrid(plan, dependency_analysis)
            
            # 更新计划状态
            if self._is_plan_completed(plan):
                plan.status = TaskStatus.COMPLETED
                if self.on_plan_complete:
                    await self._safe_callback(self.on_plan_complete, plan)
            else:
                plan.status = TaskStatus.FAILED
            
            return result
            
        except Exception as e:
            plan.status = TaskStatus.FAILED
            return {
                "success": False,
                "error": str(e),
                "completed_tasks": list(self.completed_tasks),
                "failed_tasks": list(self.failed_tasks)
            }
        finally:
            self.is_running = False
    
    async def _execute_hybrid(
        self,
        plan: TaskExecutionPlan,
        dependency_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """混合调度策略执行"""
        execution_phases = dependency_analysis["execution_phases"]
        total_phases = len(execution_phases)
        
        results = {
            "success": True,
            "completed_tasks": [],
            "failed_tasks": [],
            "phase_results": []
        }
        
        for phase_idx, phase in enumerate(execution_phases):
            if not self.is_running:
                break
            
            phase_result = await self._execute_phase(phase, plan)
            results["phase_results"].append(phase_result)
            
            # 更新整体进度
            plan.progress = (phase_idx + 1) / total_phases
            
            # 收集结果
            results["completed_tasks"].extend(phase_result["completed"])
            results["failed_tasks"].extend(phase_result["failed"])
        
        return results
    
    async def _execute_phase(
        self,
        phase: Dict[str, Any],
        plan: TaskExecutionPlan
    ) -> Dict[str, Any]:
        """执行一个阶段的任务"""
        phase_tasks = phase["tasks"]
        completed = []
        failed = []
        
        # 按优先级排序任务
        sorted_tasks = self._sort_tasks_by_priority(phase_tasks, plan)
        
        # 创建任务执行器
        with ThreadPoolExecutor(max_workers=self.max_parallel_tasks) as executor:
            # 提交所有任务
            future_to_task = {}
            for task_id in sorted_tasks:
                task = plan.get_task_by_id(task_id)
                if task and task.status == TaskStatus.READY:
                    future = executor.submit(
                        self._execute_single_task,
                        task,
                        plan
                    )
                    future_to_task[future] = task_id
            
            # 等待所有任务完成
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                    if result["success"]:
                        completed.append(task_id)
                        self.completed_tasks.add(task_id)
                    else:
                        failed.append(task_id)
                        self.failed_tasks.add(task_id)
                except Exception as e:
                    failed.append(task_id)
                    self.failed_tasks.add(task_id)
                    print(f"任务 {task_id} 执行异常: {e}")
        
        return {
            "phase_id": phase["phase_id"],
            "completed": completed,
            "failed": failed,
            "total_tasks": len(phase_tasks)
        }
    
    def _sort_tasks_by_priority(self, task_ids: List[str], plan: TaskExecutionPlan) -> List[str]:
        """按优先级排序任务"""
        task_priority_map = {}
        for task_id in task_ids:
            task = plan.get_task_by_id(task_id)
            if task:
                task_priority_map[task_id] = task.priority.value
        
        # 按优先级降序排序
        return sorted(task_ids, key=lambda x: task_priority_map.get(x, 2), reverse=True)
    
    def _execute_single_task(self, task: Task, plan: TaskExecutionPlan) -> Dict[str, Any]:
        """执行单个任务"""
        start_time = time.time()
        
        try:
            # 更新任务状态
            task.update_status(TaskStatus.RUNNING)
            
            # 调用回调
            if self.on_task_start:
                self._safe_callback(self.on_task_start, task)
            
            # 执行任务
            result = self.task_executor(task)
            
            # 处理同步/异步结果
            if asyncio.iscoroutine(result):
                # 如果是协程，等待其完成
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(result)
                finally:
                    loop.close()
            
            # 更新任务状态
            if result.get("success", False):
                task.update_status(TaskStatus.COMPLETED)
                task.update_progress(1.0)
                task.outputs.update(result.get("outputs", {}))
                self.task_results[task.id] = result
            else:
                task.update_status(TaskStatus.FAILED, result.get("error", "未知错误"))
                self.task_results[task.id] = result
            
            # 调用完成回调
            if result.get("success", False) and self.on_task_complete:
                self._safe_callback(self.on_task_complete, task)
            elif not result.get("success", False) and self.on_task_fail:
                self._safe_callback(self.on_task_fail, task, result.get("error"))
            
            return {
                "success": result.get("success", False),
                "task_id": task.id,
                "duration": time.time() - start_time,
                "result": result
            }
            
        except Exception as e:
            task.update_status(TaskStatus.FAILED, str(e))
            if self.on_task_fail:
                self._safe_callback(self.on_task_fail, task, str(e))
            
            return {
                "success": False,
                "task_id": task.id,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    async def _execute_parallel(
        self,
        plan: TaskExecutionPlan,
        dependency_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """并行执行策略"""
        parallel_groups = dependency_analysis["parallel_groups"]
        
        results = {
            "success": True,
            "completed_tasks": [],
            "failed_tasks": []
        }
        
        for group in parallel_groups:
            if not self.is_running:
                break
            
            group_result = await self._execute_parallel_group(group, plan)
            results["completed_tasks"].extend(group_result["completed"])
            results["failed_tasks"].extend(group_result["failed"])
        
        return results
    
    async def _execute_parallel_group(
        self,
        task_ids: List[str],
        plan: TaskExecutionPlan
    ) -> Dict[str, Any]:
        """执行并行任务组"""
        with ThreadPoolExecutor(max_workers=len(task_ids)) as executor:
            futures = {
                executor.submit(self._execute_single_task, plan.get_task_by_id(task_id), plan): task_id
                for task_id in task_ids
            }
            
            completed = []
            failed = []
            
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    if result["success"]:
                        completed.append(task_id)
                    else:
                        failed.append(task_id)
                except Exception as e:
                    failed.append(task_id)
                    print(f"并行任务 {task_id} 执行失败: {e}")
            
            return {"completed": completed, "failed": failed}
    
    async def _execute_critical_path(
        self,
        plan: TaskExecutionPlan,
        dependency_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """关键路径优先执行策略"""
        critical_path = dependency_analysis["critical_path"]
        
        results = {
            "success": True,
            "completed_tasks": [],
            "failed_tasks": []
        }
        
        # 首先执行关键路径上的任务
        for task_id in critical_path:
            if not self.is_running:
                break
            
            task = plan.get_task_by_id(task_id)
            if task:
                result = self._execute_single_task(task, plan)
                if result["success"]:
                    results["completed_tasks"].append(task_id)
                else:
                    results["failed_tasks"].append(task_id)
        
        # 然后执行其他任务
        remaining_tasks = [t.id for t in plan.tasks if t.id not in critical_path]
        if remaining_tasks:
            parallel_result = await self._execute_parallel_group(remaining_tasks, plan)
            results["completed_tasks"].extend(parallel_result["completed"])
            results["failed_tasks"].extend(parallel_result["failed"])
        
        return results
    
    def _default_task_executor(self, task: Task) -> Dict[str, Any]:
        """默认任务执行器"""
        # 模拟任务执行
        import random
        
        # 模拟执行时间
        execution_time = random.uniform(1, 5)
        time.sleep(execution_time)
        
        # 模拟执行结果
        if random.random() > 0.1:  # 90% 成功率
            return {
                "success": True,
                "outputs": {
                    "result": f"任务 {task.name} 执行完成",
                    "execution_time": execution_time
                }
            }
        else:
            return {
                "success": False,
                "error": f"任务 {task.name} 执行失败"
            }
    
    def _safe_callback(self, callback: Callable, *args, **kwargs):
        """安全执行回调函数"""
        try:
            if asyncio.iscoroutinefunction(callback):
                # 如果是异步函数，创建新的事件循环执行
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(callback(*args, **kwargs))
                finally:
                    loop.close()
            else:
                # 同步函数直接执行
                callback(*args, **kwargs)
        except Exception as e:
            print(f"回调函数执行失败: {e}")
    
    def _is_plan_completed(self, plan: TaskExecutionPlan) -> bool:
        """检查计划是否完成"""
        return all(
            task.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]
            for task in plan.tasks
        )
    
    def get_execution_status(self, plan: TaskExecutionPlan) -> Dict[str, Any]:
        """获取执行状态"""
        status_counts = {
            TaskStatus.PENDING: 0,
            TaskStatus.READY: 0,
            TaskStatus.RUNNING: 0,
            TaskStatus.COMPLETED: 0,
            TaskStatus.FAILED: 0,
            TaskStatus.CANCELLED: 0,
            TaskStatus.SKIPPED: 0
        }
        
        for task in plan.tasks:
            status_counts[task.status] += 1
        
        return {
            "total_tasks": len(plan.tasks),
            "status_distribution": {k.value: v for k, v in status_counts.items()},
            "completed_count": status_counts[TaskStatus.COMPLETED],
            "failed_count": status_counts[TaskStatus.FAILED],
            "running_count": status_counts[TaskStatus.RUNNING],
            "overall_progress": sum(task.progress for task in plan.tasks) / len(plan.tasks) if plan.tasks else 0
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self.lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.update_status(TaskStatus.CANCELLED)
                del self.running_tasks[task_id]
                return True
            return False
    
    def pause_scheduler(self):
        """暂停调度器"""
        self.is_running = False
    
    def resume_scheduler(self):
        """恢复调度器"""
        self.is_running = True
    
    def stop_scheduler(self):
        """停止调度器"""
        self.is_running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)