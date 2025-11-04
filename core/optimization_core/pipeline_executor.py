"""
流水线执行器

支持任务流水线执行、阶段管理和数据流控制
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """流水线阶段"""
    INPUT = "input"
    PROCESSING = "processing"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    OUTPUT = "output"
    ERROR_HANDLING = "error_handling"


class PipelineStatus(Enum):
    """流水线状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataFlowMode(Enum):
    """数据流模式"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BROADCAST = "broadcast"
    FAN_IN = "fan_in"
    FAN_OUT = "fan_out"


@dataclass
class PipelineStageConfig:
    """流水线阶段配置"""
    stage_id: str
    name: str
    stage_type: PipelineStage
    handler: Callable
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    parallel_workers: int = 1
    condition: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineTask:
    """流水线任务"""
    task_id: str
    pipeline_id: str
    data: Any
    current_stage: str
    stage_results: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineExecution:
    """流水线执行"""
    pipeline_id: str
    status: PipelineStatus
    stages: List[PipelineStageConfig]
    data_flow_mode: DataFlowMode
    current_stage_index: int = 0
    completed_stages: List[str] = field(default_factory=list)
    failed_stages: List[str] = field(default_factory=list)
    execution_stats: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class PipelineExecutor:
    """流水线执行器"""
    
    def __init__(self, max_concurrent_pipelines: int = 5):
        self.max_concurrent_pipelines = max_concurrent_pipelines
        
        # 流水线管理
        self.pipelines: Dict[str, PipelineExecution] = {}
        self.pipeline_stages: Dict[str, Dict[str, PipelineStageConfig]] = {}
        
        # 任务管理
        self.pending_tasks: deque = deque()
        self.running_tasks: Dict[str, PipelineTask] = {}
        self.completed_tasks: Dict[str, PipelineTask] = {}
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(max_concurrent_pipelines)
        self.stage_semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # 统计信息
        self.stats = {
            'total_pipelines': 0,
            'active_pipelines': 0,
            'completed_pipelines': 0,
            'failed_pipelines': 0,
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0,
            'throughput': 0.0
        }
        
        # 监控和回调
        self.pipeline_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.stage_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """启动流水线执行器"""
        self._running = True
        
        # 启动工作线程
        for i in range(self.max_concurrent_pipelines):
            task = asyncio.create_task(self._pipeline_worker(f"pipeline-worker-{i}"))
            self._worker_tasks.append(task)
        
        # 启动监控任务
        monitor_task = asyncio.create_task(self._monitor_pipelines())
        self._worker_tasks.append(monitor_task)
        
        logger.info(f"流水线执行器已启动，最大并发流水线数: {self.max_concurrent_pipelines}")
    
    async def stop(self):
        """停止流水线执行器"""
        self._running = False
        
        # 取消所有任务
        for task in self._worker_tasks:
            task.cancel()
        
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        logger.info("流水线执行器已停止")
    
    def register_pipeline(self, pipeline_id: str, stages: List[PipelineStageConfig],
                         data_flow_mode: DataFlowMode = DataFlowMode.SEQUENTIAL) -> bool:
        """注册流水线"""
        try:
            # 创建流水线执行实例
            pipeline = PipelineExecution(
                pipeline_id=pipeline_id,
                status=PipelineStatus.IDLE,
                stages=stages,
                data_flow_mode=data_flow_mode
            )
            
            self.pipelines[pipeline_id] = pipeline
            
            # 创建阶段映射
            stage_map = {stage.stage_id: stage for stage in stages}
            self.pipeline_stages[pipeline_id] = stage_map
            
            # 为每个阶段创建信号量
            for stage in stages:
                self.stage_semaphores[f"{pipeline_id}:{stage.stage_id}"] = asyncio.Semaphore(
                    stage.parallel_workers
                )
            
            self.stats['total_pipelines'] += 1
            
            logger.info(f"流水线 {pipeline_id} 注册成功，阶段数: {len(stages)}")
            return True
            
        except Exception as e:
            logger.error(f"注册流水线失败: {e}")
            return False
    
    async def execute_pipeline(self, pipeline_id: str, data: Any, 
                             task_id: Optional[str] = None) -> str:
        """执行流水线"""
        try:
            if pipeline_id not in self.pipelines:
                raise ValueError(f"流水线 {pipeline_id} 不存在")
            
            # 生成任务ID
            if task_id is None:
                task_id = f"{pipeline_id}_{int(time.time() * 1000)}"
            
            # 创建流水线任务
            pipeline_task = PipelineTask(
                task_id=task_id,
                pipeline_id=pipeline_id,
                data=data,
                current_stage=pipeline_id  # 初始阶段为流水线ID
            )
            
            # 添加到待执行队列
            self.pending_tasks.append(pipeline_task)
            self.stats['total_tasks'] += 1
            
            logger.info(f"流水线任务 {task_id} 已提交到流水线 {pipeline_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"执行流水线失败: {e}")
            raise
    
    async def _pipeline_worker(self, worker_id: str):
        """流水线工作线程"""
        logger.debug(f"流水线工作线程 {worker_id} 已启动")
        
        while self._running:
            try:
                if self.pending_tasks:
                    # 获取下一个任务
                    task = self.pending_tasks.popleft()
                    
                    # 执行流水线
                    await self._execute_pipeline_task(task)
                else:
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"流水线工作线程 {worker_id} 异常: {e}")
    
    async def _execute_pipeline_task(self, task: PipelineTask):
        """执行流水线任务"""
        pipeline = self.pipelines[task.pipeline_id]
        
        async with self.semaphore:
            try:
                # 更新状态
                pipeline.status = PipelineStatus.RUNNING
                pipeline.started_at = time.time()
                task.started_at = time.time()
                
                self.stats['active_pipelines'] += 1
                self.running_tasks[task.task_id] = task
                
                # 执行流水线阶段
                result = await self._execute_pipeline_stages(task, pipeline)
                
                if result:
                    # 流水线执行成功
                    pipeline.status = PipelineStatus.COMPLETED
                    pipeline.completed_at = time.time()
                    task.completed_at = time.time()
                    
                    self.stats['completed_pipelines'] += 1
                    self.stats['completed_tasks'] += 1
                else:
                    # 流水线执行失败
                    pipeline.status = PipelineStatus.FAILED
                    pipeline.completed_at = time.time()
                    
                    self.stats['failed_pipelines'] += 1
                    self.stats['failed_tasks'] += 1
                
                # 移动到完成队列
                self.completed_tasks[task.task_id] = task
                
                # 触发回调
                await self._trigger_pipeline_callbacks(task.pipeline_id, task, result)
                
            except Exception as e:
                logger.error(f"流水线任务执行异常: {e}")
                pipeline.status = PipelineStatus.FAILED
                task.error_info = str(e)
                
                self.stats['failed_pipelines'] += 1
                self.stats['failed_tasks'] += 1
            
            finally:
                # 清理
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                self.stats['active_pipelines'] -= 1
    
    async def _execute_pipeline_stages(self, task: PipelineTask, 
                                     pipeline: PipelineExecution) -> bool:
        """执行流水线阶段"""
        try:
            stages = pipeline.stages
            
            for i, stage in enumerate(stages):
                # 检查是否应该跳过此阶段
                if stage.condition and not await self._evaluate_condition(stage.condition, task):
                    logger.debug(f"跳过阶段 {stage.stage_id}: 条件不满足")
                    continue
                
                pipeline.current_stage_index = i
                task.current_stage = stage.stage_id
                
                # 执行阶段
                stage_result = await self._execute_stage(task, stage, pipeline)
                
                if stage_result is None:
                    # 阶段执行失败
                    pipeline.failed_stages.append(stage.stage_id)
                    logger.error(f"流水线阶段 {stage.stage_id} 执行失败")
                    return False
                
                # 记录阶段结果
                task.stage_results[stage.stage_id] = stage_result
                pipeline.completed_stages.append(stage.stage_id)
                
                # 触发阶段回调
                await self._trigger_stage_callbacks(stage.stage_id, task, stage_result)
                
                logger.debug(f"流水线阶段 {stage.stage_id} 执行完成")
            
            return True
            
        except Exception as e:
            logger.error(f"执行流水线阶段异常: {e}")
            return False
    
    async def _execute_stage(self, task: PipelineTask, stage: PipelineStageConfig,
                           pipeline: PipelineExecution) -> Optional[Any]:
        """执行单个阶段"""
        try:
            stage_key = f"{pipeline.pipeline_id}:{stage.stage_id}"
            semaphore = self.stage_semaphores[stage_key]
            
            async with semaphore:
                # 准备输入数据
                input_data = await self._prepare_stage_input(task, stage, pipeline)
                
                # 执行阶段处理
                if asyncio.iscoroutinefunction(stage.handler):
                    if stage.timeout:
                        result = await asyncio.wait_for(
                            stage.handler(input_data, task.metadata),
                            timeout=stage.timeout
                        )
                    else:
                        result = await stage.handler(input_data, task.metadata)
                else:
                    if stage.timeout:
                        result = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, stage.handler, input_data, task.metadata
                            ),
                            timeout=stage.timeout
                        )
                    else:
                        result = stage.handler(input_data, task.metadata)
                
                # 验证输出
                if stage.output_schema and not await self._validate_output(result, stage.output_schema):
                    raise ValueError(f"阶段 {stage.stage_id} 输出验证失败")
                
                return result
                
        except asyncio.TimeoutError:
            logger.warning(f"流水线阶段 {stage.stage_id} 执行超时")
            return None
        except Exception as e:
            logger.error(f"流水线阶段 {stage.stage_id} 执行异常: {e}")
            return None
    
    async def _prepare_stage_input(self, task: PipelineTask, stage: PipelineStageConfig,
                                 pipeline: PipelineExecution) -> Any:
        """准备阶段输入数据"""
        if stage.stage_type == PipelineStage.INPUT:
            return task.data
        elif stage.stage_type == PipelineStage.PROCESSING:
            # 使用前一个阶段的结果
            previous_stage = pipeline.stages[pipeline.current_stage_index - 1] if pipeline.current_stage_index > 0 else None
            if previous_stage:
                return task.stage_results.get(previous_stage.stage_id, task.data)
            return task.data
        elif stage.stage_type == PipelineStage.TRANSFORMATION:
            # 组合多个阶段的结果
            return {
                'original_data': task.data,
                'stage_results': task.stage_results,
                'current_stage': stage.stage_id
            }
        else:
            # 默认返回任务数据
            return task.data
    
    async def _validate_output(self, result: Any, schema: Dict[str, Any]) -> bool:
        """验证输出数据"""
        try:
            # 简化的验证逻辑
            if 'type' in schema:
                expected_type = schema['type']
                if expected_type == 'dict' and not isinstance(result, dict):
                    return False
                elif expected_type == 'list' and not isinstance(result, list):
                    return False
                elif expected_type == 'str' and not isinstance(result, str):
                    return False
                elif expected_type == 'int' and not isinstance(result, int):
                    return False
                elif expected_type == 'float' and not isinstance(result, (int, float)):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"输出验证异常: {e}")
            return False
    
    async def _evaluate_condition(self, condition: Callable, task: PipelineTask) -> bool:
        """评估阶段执行条件"""
        try:
            if asyncio.iscoroutinefunction(condition):
                return await condition(task)
            else:
                return condition(task)
        except Exception as e:
            logger.error(f"评估条件异常: {e}")
            return False
    
    async def _trigger_pipeline_callbacks(self, pipeline_id: str, task: PipelineTask, result: bool):
        """触发流水线回调"""
        try:
            callbacks = self.pipeline_callbacks.get(pipeline_id, [])
            for callback in callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task, result)
                else:
                    callback(task, result)
        except Exception as e:
            logger.error(f"触发流水线回调失败: {e}")
    
    async def _trigger_stage_callbacks(self, stage_id: str, task: PipelineTask, result: Any):
        """触发阶段回调"""
        try:
            callbacks = self.stage_callbacks.get(stage_id, [])
            for callback in callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(task, result)
                else:
                    callback(task, result)
        except Exception as e:
            logger.error(f"触发阶段回调失败: {e}")
    
    async def _monitor_pipelines(self):
        """监控流水线"""
        while self._running:
            try:
                await asyncio.sleep(30)  # 每30秒监控一次
                
                # 检查超时的流水线
                current_time = time.time()
                for pipeline_id, pipeline in self.pipelines.items():
                    if (pipeline.status == PipelineStatus.RUNNING and 
                        pipeline.started_at and
                        current_time - pipeline.started_at > 300):  # 5分钟超时
                        
                        logger.warning(f"流水线 {pipeline_id} 执行超时")
                        pipeline.status = PipelineStatus.FAILED
                
                # 更新统计
                await self._update_stats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"流水线监控异常: {e}")
    
    async def _update_stats(self):
        """更新统计信息"""
        try:
            # 计算平均执行时间
            completed_pipelines = [
                p for p in self.pipelines.values()
                if p.status == PipelineStatus.COMPLETED and p.completed_at and p.started_at
            ]
            
            if completed_pipelines:
                total_time = sum(
                    p.completed_at - p.started_at for p in completed_pipelines
                )
                self.stats['average_execution_time'] = total_time / len(completed_pipelines)
            
            # 计算吞吐量
            recent_pipelines = [
                p for p in completed_pipelines
                if time.time() - p.completed_at < 60
            ]
            self.stats['throughput'] = len(recent_pipelines) / 60.0
            
        except Exception as e:
            logger.error(f"更新统计失败: {e}")
    
    async def pause_pipeline(self, pipeline_id: str) -> bool:
        """暂停流水线"""
        try:
            if pipeline_id in self.pipelines:
                pipeline = self.pipelines[pipeline_id]
                if pipeline.status == PipelineStatus.RUNNING:
                    pipeline.status = PipelineStatus.PAUSED
                    logger.info(f"流水线 {pipeline_id} 已暂停")
                    return True
            return False
        except Exception as e:
            logger.error(f"暂停流水线失败: {e}")
            return False
    
    async def resume_pipeline(self, pipeline_id: str) -> bool:
        """恢复流水线"""
        try:
            if pipeline_id in self.pipelines:
                pipeline = self.pipelines[pipeline_id]
                if pipeline.status == PipelineStatus.PAUSED:
                    pipeline.status = PipelineStatus.RUNNING
                    logger.info(f"流水线 {pipeline_id} 已恢复")
                    return True
            return False
        except Exception as e:
            logger.error(f"恢复流水线失败: {e}")
            return False
    
    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """取消流水线"""
        try:
            if pipeline_id in self.pipelines:
                pipeline = self.pipelines[pipeline_id]
                pipeline.status = PipelineStatus.CANCELLED
                
                # 取消相关的运行任务
                tasks_to_cancel = [
                    task_id for task_id, task in self.running_tasks.items()
                    if task.pipeline_id == pipeline_id
                ]
                
                for task_id in tasks_to_cancel:
                    if task_id in self.running_tasks:
                        del self.running_tasks[task_id]
                
                logger.info(f"流水线 {pipeline_id} 已取消")
                return True
            return False
        except Exception as e:
            logger.error(f"取消流水线失败: {e}")
            return False
    
    def add_pipeline_callback(self, pipeline_id: str, callback: Callable):
        """添加流水线回调"""
        self.pipeline_callbacks[pipeline_id].append(callback)
    
    def add_stage_callback(self, stage_id: str, callback: Callable):
        """添加阶段回调"""
        self.stage_callbacks[stage_id].append(callback)
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[PipelineStatus]:
        """获取流水线状态"""
        if pipeline_id in self.pipelines:
            return self.pipelines[pipeline_id].status
        return None
    
    def get_pipeline_stats(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """获取流水线统计"""
        if pipeline_id in self.pipelines:
            pipeline = self.pipelines[pipeline_id]
            return {
                'pipeline_id': pipeline_id,
                'status': pipeline.status.value,
                'current_stage_index': pipeline.current_stage_index,
                'completed_stages': pipeline.completed_stages,
                'failed_stages': pipeline.failed_stages,
                'total_stages': len(pipeline.stages),
                'execution_time': (
                    pipeline.completed_at - pipeline.started_at 
                    if pipeline.completed_at and pipeline.started_at else None
                )
            }
        return None
    
    def get_executor_stats(self) -> Dict[str, Any]:
        """获取执行器统计"""
        return {
            'active_pipelines': len([p for p in self.pipelines.values() if p.status == PipelineStatus.RUNNING]),
            'completed_pipelines': len([p for p in self.pipelines.values() if p.status == PipelineStatus.COMPLETED]),
            'failed_pipelines': len([p for p in self.pipelines.values() if p.status == PipelineStatus.FAILED]),
            'pending_tasks': len(self.pending_tasks),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'max_concurrent_pipelines': self.max_concurrent_pipelines,
            'stats': self.stats.copy()
        }