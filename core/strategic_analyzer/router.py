"""
模型路由系统
根据请求类型、负载和优先级智能路由到合适的模型
"""

import asyncio
import random
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from ..config import ModelConfig, ModelType, Provider, config
from ..utils import RateLimiter, CircuitBreaker, measure_time

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """路由策略枚举"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    FASTEST_RESPONSE = "fastest_response"
    PRIORITY_BASED = "priority_based"
    LOAD_BALANCED = "load_balanced"
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"

@dataclass
class ModelMetrics:
    """模型性能指标"""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_request_time: float = 0.0
    current_connections: int = 0
    rate_limiter: Optional[RateLimiter] = None
    circuit_breaker: Optional[CircuitBreaker] = None
    error_rate: float = 0.0
    cost_per_token: float = 0.0
    quality_score: float = 0.0

class ModelRouter:
    """智能模型路由器"""
    
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.LOAD_BALANCED):
        self.strategy = strategy
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.round_robin_index = 0
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """初始化模型指标"""
        for model_name in config.get_available_models():
            model_config = config.get_model(model_name)
            if model_config:
                self.model_metrics[model_name] = ModelMetrics(
                    model_name=model_name,
                    rate_limiter=RateLimiter(
                        max_requests=model_config.rate_limit,
                        time_window=60
                    ),
                    circuit_breaker=CircuitBreaker(
                        failure_threshold=model_config.rate_limit // 10,
                        recovery_timeout=60
                    )
                )
    
    @measure_time
    async def route_request(
        self,
        request_type: ModelType,
        content: Dict[str, Any],
        preferred_models: Optional[List[str]] = None,
        **kwargs
    ) -> Optional[str]:
        """
        智能路由请求到合适的模型
        
        Args:
            request_type: 请求类型（文本、图像、音频、视频、多模态）
            content: 请求内容
            preferred_models: 首选模型列表
            **kwargs: 其他路由参数
            
        Returns:
            选择的模型名称，如果没有可用模型则返回None
        """
        available_models = self._get_available_models(request_type, preferred_models)
        
        if not available_models:
            logger.warning(f"没有找到适用于{request_type.value}的可用模型")
            return None
        
        # 根据策略选择模型
        selected_model = await self._select_model(available_models, content, **kwargs)
        
        if selected_model:
            # 更新指标
            await self._update_metrics(selected_model, True)
            logger.info(f"请求路由到模型: {selected_model}")
        else:
            logger.warning("所有模型都不可用")
        
        return selected_model
    
    def _get_available_models(
        self,
        request_type: ModelType,
        preferred_models: Optional[List[str]] = None
    ) -> List[str]:
        """获取可用模型列表"""
        available = []
        
        # 如果指定了首选模型，优先考虑
        if preferred_models:
            for model_name in preferred_models:
                if (model_name in self.model_metrics and 
                    self._is_model_available(model_name, request_type)):
                    available.append(model_name)
        
        # 获取所有支持该类型的模型
        for model_name in config.get_available_models():
            if (model_name not in available and 
                self._is_model_available(model_name, request_type)):
                available.append(model_name)
        
        return available
    
    def _is_model_available(self, model_name: str, request_type: ModelType) -> bool:
        """检查模型是否可用"""
        model_config = config.get_model(model_name)
        model_metrics = self.model_metrics.get(model_name)
        
        if not model_config or not model_metrics:
            return False
        
        # 检查类型支持
        if request_type == ModelType.MULTIMODAL and not model_config.supports_multimodal:
            return False
        
        # 检查熔断器状态
        if model_metrics.circuit_breaker and model_metrics.circuit_breaker.state == 'OPEN':
            return False
        
        # 检查限流器
        if model_metrics.rate_limiter:
            # 这里应该检查限流器状态，但RateLimiter的acquire是异步的
            # 在路由阶段我们只做基本检查
            pass
        
        return True
    
    async def _select_model(
        self,
        available_models: List[str],
        content: Dict[str, Any],
        **kwargs
    ) -> Optional[str]:
        """根据策略选择模型"""
        if not available_models:
            return None
        
        if len(available_models) == 1:
            return available_models[0]
        
        # 过滤出真正可用的模型（通过限流器检查）
        healthy_models = []
        for model_name in available_models:
            model_metrics = self.model_metrics[model_name]
            if model_metrics.rate_limiter:
                if await model_metrics.rate_limiter.acquire():
                    healthy_models.append(model_name)
            else:
                healthy_models.append(model_name)
        
        if not healthy_models:
            return None
        
        # 根据策略选择
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_models)
        elif self.strategy == RoutingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_models)
        elif self.strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_models)
        elif self.strategy == RoutingStrategy.FASTEST_RESPONSE:
            return self._fastest_response_select(healthy_models)
        elif self.strategy == RoutingStrategy.PRIORITY_BASED:
            return self._priority_based_select(healthy_models)
        elif self.strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_select(healthy_models)
        elif self.strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            return self._quality_optimized_select(healthy_models)
        else:  # LOAD_BALANCED
            return self._load_balanced_select(healthy_models, content)
    
    def _round_robin_select(self, models: List[str]) -> str:
        """轮询选择"""
        selected = models[self.round_robin_index % len(models)]
        self.round_robin_index += 1
        return selected
    
    def _weighted_round_robin_select(self, models: List[str]) -> str:
        """加权轮询选择"""
        weights = []
        for model_name in models:
            model_config = config.get_model(model_name)
            weight = model_config.priority if model_config else 1
            weights.extend([model_name] * weight)
        
        return random.choice(weights) if weights else models[0]
    
    def _least_connections_select(self, models: List[str]) -> str:
        """最少连接选择"""
        return min(models, key=lambda m: self.model_metrics[m].current_connections)
    
    def _fastest_response_select(self, models: List[str]) -> str:
        """最快响应选择"""
        return min(models, key=lambda m: self.model_metrics[m].avg_response_time)
    
    def _priority_based_select(self, models: List[str]) -> str:
        """优先级选择"""
        return max(models, key=lambda m: config.get_model(m).priority if config.get_model(m) else 0)
    
    def _cost_optimized_select(self, models: List[str]) -> str:
        """成本优化选择"""
        return min(models, key=lambda m: self.model_metrics[m].cost_per_token)
    
    def _quality_optimized_select(self, models: List[str]) -> str:
        """质量优化选择"""
        return max(models, key=lambda m: self.model_metrics[m].quality_score)
    
    def _load_balanced_select(self, models: List[str], content: Dict[str, Any]) -> str:
        """负载均衡选择（综合多种因素）"""
        scores = {}
        
        for model_name in models:
            model_config = config.get_model(model_name)
            model_metrics = self.model_metrics[model_name]
            
            if not model_config:
                continue
            
            # 计算综合得分
            score = 0
            
            # 1. 响应时间得分（越快越好）
            if model_metrics.avg_response_time > 0:
                score += 100 / (1 + model_metrics.avg_response_time)
            
            # 2. 错误率得分（越低越好）
            error_rate = model_metrics.error_rate
            score += 100 * (1 - error_rate)
            
            # 3. 当前连接数得分（越少越好）
            score += 50 / (1 + model_metrics.current_connections)
            
            # 4. 优先级得分
            score += model_config.priority * 10
            
            # 5. 成功率得分
            if model_metrics.total_requests > 0:
                success_rate = model_metrics.successful_requests / model_metrics.total_requests
                score += success_rate * 50
            
            scores[model_name] = score
        
        # 选择得分最高的模型
        if scores:
            return max(scores.keys(), key=lambda k: scores[k])
        
        return models[0]
    
    async def _update_metrics(self, model_name: str, success: bool, response_time: float = 0.0):
        """更新模型指标"""
        if model_name not in self.model_metrics:
            return
        
        metrics = self.model_metrics[model_name]
        metrics.total_requests += 1
        metrics.last_request_time = time.time()
        
        if success:
            metrics.successful_requests += 1
            # 更新平均响应时间
            if response_time > 0:
                if metrics.avg_response_time == 0:
                    metrics.avg_response_time = response_time
                else:
                    metrics.avg_response_time = (metrics.avg_response_time + response_time) / 2
        else:
            metrics.failed_requests += 1
        
        # 更新错误率
        if metrics.total_requests > 0:
            metrics.error_rate = metrics.failed_requests / metrics.total_requests
        
        # 更新熔断器
        if metrics.circuit_breaker:
            if success:
                metrics.circuit_breaker._on_success()
            else:
                metrics.circuit_breaker._on_failure()
    
    def get_model_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模型状态"""
        status = {}
        for model_name, metrics in self.model_metrics.items():
            model_config = config.get_model(model_name)
            status[model_name] = {
                'available': self._is_model_available(model_name, ModelType.TEXT),  # 默认检查文本类型
                'total_requests': metrics.total_requests,
                'successful_requests': metrics.successful_requests,
                'failed_requests': metrics.failed_requests,
                'avg_response_time': metrics.avg_response_time,
                'current_connections': metrics.current_connections,
                'error_rate': metrics.error_rate,
                'circuit_breaker_state': metrics.circuit_breaker.state if metrics.circuit_breaker else 'UNKNOWN',
                'priority': model_config.priority if model_config else 0,
                'provider': model_config.provider.value if model_config else 'UNKNOWN',
                'model_type': model_config.model_type.value if model_config else 'UNKNOWN'
            }
        
        return status
    
    def get_best_model_for_type(self, model_type: ModelType) -> Optional[str]:
        """获取某个类型的最佳模型"""
        available_models = config.get_models_by_type(model_type)
        if not available_models:
            return None
        
        # 优先选择支持多模态的模型
        multimodal_models = [m for m in available_models if m.supports_multimodal]
        if multimodal_models:
            return max(multimodal_models, key=lambda m: m.priority).name
        
        # 否则选择优先级最高的
        return max(available_models, key=lambda m: m.priority).name
    
    def set_routing_strategy(self, strategy: RoutingStrategy):
        """设置路由策略"""
        self.strategy = strategy
        logger.info(f"路由策略已更新为: {strategy.value}")
    
    async def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        health_status = {}
        for model_name in self.model_metrics.keys():
            try:
                # 简单的健康检查逻辑
                metrics = self.model_metrics[model_name]
                is_healthy = (
                    metrics.error_rate < 0.5 and  # 错误率小于50%
                    (metrics.circuit_breaker is None or metrics.circuit_breaker.state != 'OPEN')
                )
                health_status[model_name] = is_healthy
            except Exception as e:
                logger.error(f"健康检查失败 {model_name}: {e}")
                health_status[model_name] = False
        
        return health_status