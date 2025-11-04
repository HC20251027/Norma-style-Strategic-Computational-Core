"""
LLM编排器
统一管理多模型、多模态的智能编排系统
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .router import ModelRouter, RoutingStrategy
from .cache import ResponseCache
from .load_balancer import LoadBalancer, LoadBalanceAlgorithm
from ..config import ModelType, config
from ..utils import measure_time, retry_async

logger = logging.getLogger(__name__)

class RequestPriority(Enum):
    """请求优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class LLMRequest:
    """LLM请求"""
    id: str
    prompt: str
    model_type: ModelType
    media_data: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    preferred_models: Optional[List[str]] = None
    priority: RequestPriority = RequestPriority.NORMAL
    timeout: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.metadata is None:
            self.metadata = {}

@dataclass
class LLMResponse:
    """LLM响应"""
    id: str
    content: Any
    model_used: str
    tokens_used: Optional[int] = None
    response_time: float = 0.0
    cached: bool = False
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class LLMOrchestrator:
    """LLM智能编排器"""
    
    def __init__(self):
        self.router = ModelRouter()
        self.cache = ResponseCache()
        self.load_balancer = LoadBalancer()
        
        # 请求队列和并发控制
        self.request_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.max_concurrent_requests = 50
        self.active_requests: Dict[str, asyncio.Task] = {}
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cached_requests': 0,
            'avg_response_time': 0.0,
            'start_time': time.time()
        }
        
        # 启动工作进程
        self.worker_tasks: List[asyncio.Task] = []
        self._start_workers()
        
        logger.info("LLM编排器初始化完成")
    
    def _start_workers(self):
        """启动工作进程"""
        for i in range(self.max_concurrent_requests):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(task)
    
    async def _worker(self, worker_id: str):
        """工作进程"""
        logger.info(f"工作进程 {worker_id} 已启动")
        
        while True:
            try:
                # 从队列获取请求
                request = await self.request_queue.get()
                
                # 处理请求
                await self._process_request(request, worker_id)
                
                # 标记任务完成
                self.request_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info(f"工作进程 {worker_id} 已取消")
                break
            except Exception as e:
                logger.error(f"工作进程 {worker_id} 错误: {e}")
                await asyncio.sleep(1)
    
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """
        处理LLM请求
        
        Args:
            request: LLM请求对象
            
        Returns:
            LLM响应对象
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # 生成请求ID（如果未提供）
            if not request.id:
                request.id = f"req_{int(time.time() * 1000)}"
            
            # 检查缓存
            cached_response = await self._check_cache(request)
            if cached_response:
                self.stats['cached_requests'] += 1
                return cached_response
            
            # 添加到处理队列
            queue_task = asyncio.create_task(self._enqueue_request(request))
            
            # 等待处理完成
            response = await queue_task
            
            # 更新统计信息
            response_time = time.time() - start_time
            self._update_stats(response, response_time)
            
            # 缓存响应
            if not response.cached and not response.error:
                await self._cache_response(request, response)
            
            return response
            
        except Exception as e:
            logger.error(f"请求处理失败: {e}")
            self.stats['failed_requests'] += 1
            
            return LLMResponse(
                id=request.id,
                content=None,
                model_used="",
                error=str(e),
                response_time=time.time() - start_time
            )
    
    async def _enqueue_request(self, request: LLMRequest) -> LLMResponse:
        """将请求加入队列"""
        try:
            # 等待队列有空间
            await asyncio.wait_for(
                self.request_queue.put(request),
                timeout=request.timeout or 30.0
            )
            
            # 等待处理完成
            response = await asyncio.create_task(self._wait_for_response(request.id))
            return response
            
        except asyncio.TimeoutError:
            raise Exception("请求队列超时")
    
    async def _wait_for_response(self, request_id: str) -> LLMResponse:
        """等待响应"""
        # 这里应该实现更复杂的等待逻辑
        # 简化实现：直接处理请求
        return await self._process_request_sync(request_id)
    
    async def _process_request(self, request: LLMRequest, worker_id: str):
        """处理请求"""
        try:
            response = await self._process_request_sync(request.id)
            # 这里应该将响应存储到某个地方供等待的请求获取
            # 简化实现暂时跳过
        except Exception as e:
            logger.error(f"工作进程 {worker_id} 处理请求失败: {e}")
    
    async def _process_request_sync(self, request_id: str) -> LLMResponse:
        """同步处理请求"""
        # 这里应该根据request_id获取完整的请求信息
        # 简化实现：创建一个模拟响应
        await asyncio.sleep(0.1)  # 模拟处理时间
        
        return LLMResponse(
            id=request_id,
            content="模拟响应内容",
            model_used="gpt-4",
            response_time=0.1
        )
    
    async def _check_cache(self, request: LLMRequest) -> Optional[LLMResponse]:
        """检查缓存"""
        try:
            cache_key = self.cache.get_cache_key_for_request(
                model_name="default",
                prompt=request.prompt,
                parameters=request.parameters,
                media_data=request.media_data
            )
            
            cached_data = await self.cache.get(cache_key)
            if cached_data:
                return LLMResponse(
                    id=request.id,
                    content=cached_data.get('content'),
                    model_used=cached_data.get('model_used', 'cached'),
                    cached=True,
                    metadata=cached_data.get('metadata', {})
                )
        except Exception as e:
            logger.warning(f"缓存检查失败: {e}")
        
        return None
    
    async def _cache_response(self, request: LLMRequest, response: LLMResponse):
        """缓存响应"""
        try:
            cache_key = self.cache.get_cache_key_for_request(
                model_name=response.model_used,
                prompt=request.prompt,
                parameters=request.parameters,
                media_data=request.media_data
            )
            
            cache_data = {
                'content': response.content,
                'model_used': response.model_used,
                'tokens_used': response.tokens_used,
                'metadata': response.metadata
            }
            
            await self.cache.set(cache_key, cache_data, ttl=3600)
        except Exception as e:
            logger.warning(f"缓存响应失败: {e}")
    
    def _update_stats(self, response: LLMResponse, response_time: float):
        """更新统计信息"""
        if response.error:
            self.stats['failed_requests'] += 1
        else:
            self.stats['successful_requests'] += 1
        
        # 更新平均响应时间
        total_requests = self.stats['successful_requests'] + self.stats['failed_requests']
        if total_requests > 1:
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (total_requests - 1) + response_time) / total_requests
            )
        else:
            self.stats['avg_response_time'] = response_time
    
    @measure_time
    async def process_text_request(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """处理文本请求"""
        request = LLMRequest(
            id=f"text_{int(time.time() * 1000)}",
            prompt=prompt,
            model_type=ModelType.TEXT,
            preferred_models=[model_name] if model_name else None,
            parameters=parameters or {},
            metadata=kwargs
        )
        
        return await self.process_request(request)
    
    @measure_time
    async def process_multimodal_request(
        self,
        prompt: str,
        media_data: Dict[str, Any],
        model_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """处理多模态请求"""
        request = LLMRequest(
            id=f"multi_{int(time.time() * 1000)}",
            prompt=prompt,
            model_type=ModelType.MULTIMODAL,
            media_data=media_data,
            preferred_models=[model_name] if model_name else None,
            parameters=parameters or {},
            metadata=kwargs
        )
        
        return await self.process_request(request)
    
    async def stream_response(
        self,
        request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """流式响应"""
        # 简化实现：生成模拟流式数据
        response_text = "这是流式响应的模拟内容"
        
        for i in range(0, len(response_text), 5):
            chunk = response_text[i:i+5]
            yield chunk
            await asyncio.sleep(0.1)
    
    async def batch_process_requests(
        self,
        requests: List[LLMRequest],
        max_concurrent: int = 10
    ) -> List[LLMResponse]:
        """批量处理请求"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(request: LLMRequest) -> LLMResponse:
            async with semaphore:
                return await self.process_request(request)
        
        tasks = [process_with_semaphore(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                processed_responses.append(LLMResponse(
                    id=requests[i].id,
                    content=None,
                    model_used="",
                    error=str(response)
                ))
            else:
                processed_responses.append(response)
        
        return processed_responses
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        uptime = time.time() - self.stats['start_time']
        
        return {
            'uptime': uptime,
            'stats': self.stats.copy(),
            'router_status': self.router.get_model_status(),
            'cache_stats': self.cache.get_stats(),
            'load_balancer_stats': self.load_balancer.get_load_balance_stats(),
            'queue_size': self.request_queue.qsize(),
            'active_requests': len(self.active_requests),
            'worker_count': len(self.worker_tasks)
        }
    
    def set_routing_strategy(self, strategy: RoutingStrategy):
        """设置路由策略"""
        self.router.set_routing_strategy(strategy)
    
    def set_load_balance_algorithm(self, algorithm: LoadBalanceAlgorithm):
        """设置负载均衡算法"""
        self.load_balancer.set_algorithm(algorithm)
    
    def enable_cache(self):
        """启用缓存"""
        self.cache.enable()
    
    def disable_cache(self):
        """禁用缓存"""
        self.cache.disable()
    
    async def clear_cache(self):
        """清空缓存"""
        await self.cache.clear()
    
    def add_model_node(
        self,
        node_id: str,
        host: str,
        port: int,
        weight: int = 1
    ):
        """添加模型节点"""
        from .load_balancer import ServerNode
        node = ServerNode(
            id=node_id,
            host=host,
            port=port,
            weight=weight
        )
        self.load_balancer.add_node(node)
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        router_health = await self.router.health_check()
        system_stats = self.get_system_status()
        
        return {
            'orchestrator_healthy': True,
            'router_health': router_health,
            'system_stats': system_stats,
            'timestamp': time.time()
        }
    
    async def shutdown(self):
        """关闭编排器"""
        logger.info("开始关闭LLM编排器")
        
        # 取消所有工作进程
        for task in self.worker_tasks:
            if not task.done():
                task.cancel()
        
        # 等待工作进程完成
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # 关闭负载均衡器
        await self.load_balancer.shutdown()
        
        logger.info("LLM编排器已关闭")