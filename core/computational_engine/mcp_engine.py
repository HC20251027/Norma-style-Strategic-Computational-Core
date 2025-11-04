"""
优化MCP执行引擎

支持并发执行、流水线执行、错误恢复和性能优化
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import subprocess

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ExecutionMode(Enum):
    """执行模式"""
    SYNC = "sync"
    ASYNC = "async"
    STREAM = "stream"
    BATCH = "batch"
    PIPELINE = "pipeline"


@dataclass
class MCPRequest:
    """MCP请求"""
    id: str
    method: str
    params: Dict[str, Any]
    timeout: float = 30.0
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class MCPResponse:
    """MCP响应"""
    id: str
    status: ExecutionStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class ExecutionContext:
    """执行上下文"""
    request_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    callbacks: List[Callable] = field(default_factory=list)


class MCPEngine:
    """优化MCP执行引擎"""
    
    def __init__(self, max_concurrent: int = 10, max_pipeline_depth: int = 5):
        self.max_concurrent = max_concurrent
        self.max_pipeline_depth = max_pipeline_depth
        
        # 执行管理
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.completed_executions: Dict[str, MCPResponse] = {}
        self.execution_history: List[MCPResponse] = []
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_count = 0
        self.queue = asyncio.Queue()
        
        # 性能统计
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_execution_time': 0.0,
            'throughput': 0.0,
            'error_rate': 0.0
        }
        
        # 错误处理
        self.retry_config = {
            'max_retries': 3,
            'retry_delay': 1.0,
            'backoff_factor': 2.0
        }
        
        # 缓存
        self.response_cache: Dict[str, MCPResponse] = {}
        self.cache_ttl = 300  # 5分钟
        
        # 监控
        self.execution_monitor: Optional[Callable] = None
        self.performance_monitor: Optional[Callable] = None
        
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """启动MCP引擎"""
        self._running = True
        
        # 启动工作线程
        for i in range(self.max_concurrent):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self._worker_tasks.append(task)
        
        # 启动清理任务
        cleanup_task = asyncio.create_task(self._cleanup_worker())
        self._worker_tasks.append(cleanup_task)
        
        logger.info(f"MCP引擎已启动，最大并发数: {self.max_concurrent}")
    
    async def stop(self):
        """停止MCP引擎"""
        self._running = False
        
        # 取消所有工作线程
        for task in self._worker_tasks:
            task.cancel()
        
        # 等待所有任务完成
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        logger.info("MCP引擎已停止")
    
    async def execute(self, request: MCPRequest, context: Optional[ExecutionContext] = None,
                     mode: ExecutionMode = ExecutionMode.ASYNC) -> MCPResponse:
        """执行MCP请求"""
        try:
            start_time = time.time()
            
            # 创建执行上下文
            if context is None:
                context = ExecutionContext(request_id=request.id)
            
            # 检查缓存
            if request.id in self.response_cache:
                cached_response = self.response_cache[request.id]
                if time.time() - cached_response.created_at < self.cache_ttl:
                    logger.debug(f"使用缓存响应: {request.id}")
                    return cached_response
            
            # 根据执行模式处理
            if mode == ExecutionMode.SYNC:
                return await self._execute_sync(request, context)
            elif mode == ExecutionMode.ASYNC:
                return await self._execute_async(request, context)
            elif mode == ExecutionMode.STREAM:
                return await self._execute_stream(request, context)
            elif mode == ExecutionMode.BATCH:
                return await self._execute_batch(request, context)
            elif mode == ExecutionMode.PIPELINE:
                return await self._execute_pipeline(request, context)
            else:
                raise ValueError(f"不支持的执行模式: {mode}")
                
        except Exception as e:
            logger.error(f"MCP执行异常: {e}")
            return MCPResponse(
                id=request.id,
                status=ExecutionStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _execute_sync(self, request: MCPRequest, context: ExecutionContext) -> MCPResponse:
        """同步执行"""
        async with self.semaphore:
            return await self._process_request(request, context)
    
    async def _execute_async(self, request: MCPRequest, context: ExecutionContext) -> MCPResponse:
        """异步执行"""
        # 添加到队列
        await self.queue.put((request, context))
        
        # 等待完成
        while request.id not in self.completed_executions:
            await asyncio.sleep(0.01)
        
        return self.completed_executions[request.id]
    
    async def _execute_stream(self, request: MCPRequest, context: ExecutionContext) -> MCPResponse:
        """流式执行"""
        # 流式执行返回部分结果
        response = MCPResponse(id=request.id, status=ExecutionStatus.RUNNING)
        
        try:
            # 模拟流式处理
            for i in range(5):
                if not self._running:
                    response.status = ExecutionStatus.CANCELLED
                    break
                
                # 发送中间结果
                partial_result = {
                    'progress': (i + 1) * 20,
                    'message': f'处理中... {i + 1}/5'
                }
                
                if hasattr(context, 'callbacks'):
                    for callback in context.callbacks:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(request.id, partial_result)
                        else:
                            callback(request.id, partial_result)
                
                await asyncio.sleep(0.5)
            
            # 完成执行
            if response.status == ExecutionStatus.RUNNING:
                response = await self._process_request(request, context)
                response.status = ExecutionStatus.COMPLETED
            
        except Exception as e:
            response.status = ExecutionStatus.FAILED
            response.error = str(e)
        
        return response
    
    async def _execute_batch(self, request: MCPRequest, context: ExecutionContext) -> MCPResponse:
        """批量执行"""
        # 批量执行多个相似请求
        batch_requests = request.metadata.get('batch_requests', [request])
        
        results = []
        for batch_request in batch_requests:
            try:
                result = await self._process_request(batch_request, context)
                results.append(result)
            except Exception as e:
                logger.error(f"批量请求处理失败: {e}")
                results.append(MCPResponse(
                    id=batch_request.id,
                    status=ExecutionStatus.FAILED,
                    error=str(e)
                ))
        
        return MCPResponse(
            id=request.id,
            status=ExecutionStatus.COMPLETED,
            result=results,
            execution_time=time.time() - request.created_at
        )
    
    async def _execute_pipeline(self, request: MCPRequest, context: ExecutionContext) -> MCPResponse:
        """流水线执行"""
        pipeline_steps = request.metadata.get('pipeline_steps', [])
        
        if not pipeline_steps:
            return await self._process_request(request, context)
        
        current_result = None
        total_time = 0.0
        
        for i, step in enumerate(pipeline_steps):
            if not self._running:
                break
            
            step_start = time.time()
            
            try:
                # 创建步骤请求
                step_request = MCPRequest(
                    id=f"{request.id}_step_{i}",
                    method=step['method'],
                    params={**(step.get('params', {})), 'previous_result': current_result}
                )
                
                # 执行步骤
                step_response = await self._process_request(step_request, context)
                
                if step_response.status == ExecutionStatus.COMPLETED:
                    current_result = step_response.result
                    total_time += time.time() - step_start
                else:
                    return MCPResponse(
                        id=request.id,
                        status=ExecutionStatus.FAILED,
                        error=f"流水线步骤 {i} 失败: {step_response.error}"
                    )
                
            except Exception as e:
                return MCPResponse(
                    id=request.id,
                    status=ExecutionStatus.FAILED,
                    error=f"流水线步骤 {i} 异常: {str(e)}"
                )
        
        return MCPResponse(
            id=request.id,
            status=ExecutionStatus.COMPLETED,
            result=current_result,
            execution_time=total_time
        )
    
    async def _process_request(self, request: MCPRequest, context: ExecutionContext) -> MCPResponse:
        """处理单个请求"""
        start_time = time.time()
        self.active_count += 1
        self.stats['total_requests'] += 1
        
        # 记录活跃执行
        self.active_executions[request.id] = context
        
        try:
            # 重试机制
            for attempt in range(self.retry_config['max_retries'] + 1):
                try:
                    # 执行MCP调用
                    result = await self._call_mcp_method(request, context)
                    
                    # 成功响应
                    response = MCPResponse(
                        id=request.id,
                        status=ExecutionStatus.COMPLETED,
                        result=result,
                        execution_time=time.time() - start_time,
                        completed_at=time.time()
                    )
                    
                    self.stats['successful_requests'] += 1
                    break
                    
                except Exception as e:
                    if attempt == self.retry_config['max_retries']:
                        # 最后一次尝试失败
                        response = MCPResponse(
                            id=request.id,
                            status=ExecutionStatus.FAILED,
                            error=str(e),
                            execution_time=time.time() - start_time,
                            completed_at=time.time()
                        )
                        self.stats['failed_requests'] += 1
                    else:
                        # 重试
                        delay = self.retry_config['retry_delay'] * (
                            self.retry_config['backoff_factor'] ** attempt
                        )
                        await asyncio.sleep(delay)
                        logger.warning(f"请求 {request.id} 第 {attempt + 1} 次重试")
            
            # 缓存响应
            self.response_cache[request.id] = response
            
            # 记录完成
            self.completed_executions[request.id] = response
            self.execution_history.append(response)
            
            # 触发回调
            if context.callbacks:
                for callback in context.callbacks:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(request.id, response)
                    else:
                        callback(request.id, response)
            
            return response
            
        finally:
            self.active_count -= 1
            # 从活跃执行中移除
            if request.id in self.active_executions:
                del self.active_executions[request.id]
    
    async def _call_mcp_method(self, request: MCPRequest, context: ExecutionContext) -> Any:
        """调用MCP方法"""
        try:
            # 这里应该实现实际的MCP方法调用
            # 目前使用模拟实现
            
            method = request.method
            params = request.params
            
            logger.debug(f"调用MCP方法: {method}, 参数: {params}")
            
            # 模拟处理时间
            await asyncio.sleep(0.1)
            
            # 根据方法类型返回不同结果
            if method == "tools/list":
                return {"tools": []}
            elif method == "tools/call":
                return {"result": f"执行工具: {params.get('name', 'unknown')}"}
            elif method == "resources/list":
                return {"resources": []}
            elif method == "prompts/list":
                return {"prompts": []}
            else:
                return {"method": method, "params": params, "status": "executed"}
                
        except Exception as e:
            logger.error(f"MCP方法调用失败: {e}")
            raise
    
    async def _worker(self, worker_id: str):
        """工作线程"""
        logger.debug(f"工作线程 {worker_id} 已启动")
        
        while self._running:
            try:
                # 从队列获取任务
                request, context = await asyncio.wait_for(
                    self.queue.get(), timeout=1.0
                )
                
                # 处理任务
                async with self.semaphore:
                    await self._process_request(request, context)
                
                # 标记任务完成
                self.queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"工作线程 {worker_id} 异常: {e}")
    
    async def _cleanup_worker(self):
        """清理工作线程"""
        while self._running:
            try:
                await asyncio.sleep(60)  # 每分钟清理一次
                
                current_time = time.time()
                
                # 清理过期的缓存响应
                expired_keys = [
                    key for key, response in self.response_cache.items()
                    if current_time - response.created_at > self.cache_ttl
                ]
                
                for key in expired_keys:
                    del self.response_cache[key]
                
                # 清理历史记录（保留最近1000条）
                if len(self.execution_history) > 1000:
                    self.execution_history = self.execution_history[-1000:]
                
                # 更新统计
                await self._update_stats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理工作线程异常: {e}")
    
    async def _update_stats(self):
        """更新统计信息"""
        try:
            total = self.stats['total_requests']
            if total > 0:
                self.stats['error_rate'] = (
                    self.stats['failed_requests'] / total * 100
                )
            
            # 计算吞吐量（每分钟请求数）
            recent_requests = [
                response for response in self.execution_history
                if time.time() - response.created_at < 60
            ]
            self.stats['throughput'] = len(recent_requests) / 60.0
            
        except Exception as e:
            logger.error(f"更新统计信息失败: {e}")
    
    async def cancel_request(self, request_id: str) -> bool:
        """取消请求"""
        try:
            if request_id in self.active_executions:
                # 标记为已取消
                response = MCPResponse(
                    id=request_id,
                    status=ExecutionStatus.CANCELLED,
                    completed_at=time.time()
                )
                
                self.completed_executions[request_id] = response
                del self.active_executions[request_id]
                
                logger.info(f"请求 {request_id} 已取消")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"取消请求失败: {e}")
            return False
    
    def add_execution_callback(self, callback: Callable):
        """添加执行回调"""
        # 这里可以添加全局回调
        pass
    
    def set_execution_monitor(self, monitor: Callable):
        """设置执行监控"""
        self.execution_monitor = monitor
    
    def set_performance_monitor(self, monitor: Callable):
        """设置性能监控"""
        self.performance_monitor = monitor
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        return {
            'active_executions': len(self.active_executions),
            'completed_executions': len(self.completed_executions),
            'active_count': self.active_count,
            'max_concurrent': self.max_concurrent,
            'queue_size': self.queue.qsize(),
            'cache_size': len(self.response_cache),
            'stats': self.stats.copy(),
            'recent_executions': [
                {
                    'id': resp.id,
                    'status': resp.status.value,
                    'execution_time': resp.execution_time
                }
                for resp in self.execution_history[-10:]
            ]
        }