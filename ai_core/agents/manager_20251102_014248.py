"""
LLM管理系统
提供完整的LLM服务管理和监控功能
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from .llm_interface import LLMInterface, LLMInterfaceConfig
from ..core.router import RoutingStrategy
from ..core.load_balancer import LoadBalanceAlgorithm
from ..config import ModelType, config, ModelConfig, Provider
from ..utils import measure_time, retry_async

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """服务状态"""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class ServiceMetrics:
    """服务指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    active_connections: int = 0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    uptime: float = 0.0
    last_updated: float = 0.0

@dataclass
class ModelPerformance:
    """模型性能"""
    model_name: str
    total_requests: int
    success_rate: float
    avg_response_time: float
    error_rate: float
    cost_per_request: float
    quality_score: float
    last_used: float

class LLMManager:
    """LLM管理系统"""
    
    def __init__(self, interface_config: LLMInterfaceConfig = None):
        self.interface_config = interface_config or LLMInterfaceConfig()
        self.interface = LLMInterface(self.interface_config)
        
        # 服务状态
        self.status = ServiceStatus.STOPPED
        self.start_time: Optional[float] = None
        
        # 指标收集
        self.metrics = ServiceMetrics()
        self.model_performance: Dict[str, ModelPerformance] = {}
        
        # 监控任务
        self.monitoring_tasks: List[asyncio.Task] = []
        self.alert_callbacks: List[Callable] = []
        
        # 配置管理
        self.config_history: List[Dict[str, Any]] = []
        
        logger.info("LLM管理系统初始化完成")
    
    async def start(self):
        """启动服务"""
        try:
            logger.info("启动LLM服务...")
            
            self.status = ServiceStatus.RUNNING
            self.start_time = time.time()
            
            # 启动监控任务
            self._start_monitoring_tasks()
            
            # 初始化模型性能数据
            await self._initialize_model_performance()
            
            logger.info("LLM服务启动完成")
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            logger.error(f"LLM服务启动失败: {e}")
            raise
    
    async def stop(self):
        """停止服务"""
        try:
            logger.info("停止LLM服务...")
            
            self.status = ServiceStatus.STOPPED
            
            # 停止监控任务
            for task in self.monitoring_tasks:
                if not task.done():
                    task.cancel()
            
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
            # 关闭接口
            await self.interface.shutdown()
            
            logger.info("LLM服务已停止")
            
        except Exception as e:
            logger.error(f"LLM服务停止失败: {e}")
    
    def _start_monitoring_tasks(self):
        """启动监控任务"""
        # 性能监控
        perf_task = asyncio.create_task(self._performance_monitor())
        self.monitoring_tasks.append(perf_task)
        
        # 健康检查
        health_task = asyncio.create_task(self._health_monitor())
        self.monitoring_tasks.append(health_task)
        
        # 指标收集
        metrics_task = asyncio.create_task(self._metrics_collector())
        self.monitoring_tasks.append(metrics_task)
        
        logger.info(f"启动{len(self.monitoring_tasks)}个监控任务")
    
    async def _performance_monitor(self):
        """性能监控"""
        while self.status == ServiceStatus.RUNNING:
            try:
                # 获取系统状态
                system_status = self.interface.get_system_status()
                
                # 更新模型性能指标
                await self._update_model_performance(system_status)
                
                # 检查性能告警
                await self._check_performance_alerts()
                
                await asyncio.sleep(30)  # 30秒监控一次
                
            except Exception as e:
                logger.error(f"性能监控错误: {e}")
                await asyncio.sleep(10)
    
    async def _health_monitor(self):
        """健康监控"""
        while self.status == ServiceStatus.RUNNING:
            try:
                health_status = await self.interface.health_check()
                
                if not health_status.get('interface_healthy', False):
                    await self._handle_unhealthy_status(health_status)
                
                await asyncio.sleep(60)  # 60秒检查一次
                
            except Exception as e:
                logger.error(f"健康监控错误: {e}")
                await asyncio.sleep(30)
    
    async def _metrics_collector(self):
        """指标收集器"""
        while self.status == ServiceStatus.RUNNING:
            try:
                # 更新服务指标
                await self._update_service_metrics()
                
                # 清理过期数据
                await self._cleanup_old_data()
                
                await asyncio.sleep(60)  # 60秒收集一次
                
            except Exception as e:
                logger.error(f"指标收集错误: {e}")
                await asyncio.sleep(30)
    
    async def _initialize_model_performance(self):
        """初始化模型性能数据"""
        available_models = self.interface.get_model_list()
        
        for model_name in available_models:
            self.model_performance[model_name] = ModelPerformance(
                model_name=model_name,
                total_requests=0,
                success_rate=1.0,
                avg_response_time=0.0,
                error_rate=0.0,
                cost_per_request=0.0,
                quality_score=0.8,
                last_used=0.0
            )
    
    async def _update_model_performance(self, system_status: Dict[str, Any]):
        """更新模型性能"""
        try:
            router_status = system_status.get('orchestrator_status', {}).get('router_status', {})
            
            for model_name, model_stats in router_status.items():
                if model_name in self.model_performance:
                    perf = self.model_performance[model_name]
                    
                    perf.total_requests = model_stats.get('total_requests', 0)
                    perf.success_rate = (
                        model_stats.get('successful_requests', 0) / 
                        max(perf.total_requests, 1)
                    )
                    perf.avg_response_time = model_stats.get('avg_response_time', 0.0)
                    perf.error_rate = model_stats.get('error_rate', 0.0)
                    perf.last_used = time.time()
        
        except Exception as e:
            logger.warning(f"模型性能更新失败: {e}")
    
    async def _update_service_metrics(self):
        """更新服务指标"""
        try:
            system_status = self.interface.get_system_status()
            
            # 更新基础指标
            self.metrics.total_requests = system_status.get('stats', {}).get('total_requests', 0)
            self.metrics.successful_requests = system_status.get('stats', {}).get('successful_requests', 0)
            self.metrics.failed_requests = system_status.get('stats', {}).get('failed_requests', 0)
            self.metrics.avg_response_time = system_status.get('stats', {}).get('avg_response_time', 0.0)
            
            # 更新连接数
            stream_stats = system_status.get('stream_manager_stats', {})
            self.metrics.active_connections = stream_stats.get('active_connections', 0)
            
            # 更新运行时间
            if self.start_time:
                self.metrics.uptime = time.time() - self.start_time
            
            self.metrics.last_updated = time.time()
            
        except Exception as e:
            logger.warning(f"服务指标更新失败: {e}")
    
    async def _check_performance_alerts(self):
        """检查性能告警"""
        try:
            # 检查响应时间
            if self.metrics.avg_response_time > 10.0:  # 超过10秒
                await self._trigger_alert('high_response_time', {
                    'response_time': self.metrics.avg_response_time,
                    'threshold': 10.0
                })
            
            # 检查错误率
            total_requests = self.metrics.total_requests
            if total_requests > 100:  # 至少100个请求后才检查错误率
                error_rate = self.metrics.failed_requests / total_requests
                if error_rate > 0.1:  # 错误率超过10%
                    await self._trigger_alert('high_error_rate', {
                        'error_rate': error_rate,
                        'threshold': 0.1
                    })
            
            # 检查连接数
            if self.metrics.active_connections > 1000:  # 连接数过多
                await self._trigger_alert('high_connections', {
                    'active_connections': self.metrics.active_connections,
                    'threshold': 1000
                })
        
        except Exception as e:
            logger.warning(f"性能告警检查失败: {e}")
    
    async def _handle_unhealthy_status(self, health_status: Dict[str, Any]):
        """处理不健康状态"""
        logger.warning(f"检测到不健康状态: {health_status}")
        
        # 触发告警
        await self._trigger_alert('unhealthy_status', health_status)
        
        # 尝试自动恢复
        if self.interface_config.fallback_enabled:
            await self._attempt_auto_recovery()
    
    async def _attempt_auto_recovery(self):
        """尝试自动恢复"""
        try:
            logger.info("尝试自动恢复...")
            
            # 重启接口
            await self.interface.shutdown()
            self.interface = LLMInterface(self.interface_config)
            
            # 重新初始化模型性能数据
            await self._initialize_model_performance()
            
            logger.info("自动恢复完成")
            
        except Exception as e:
            logger.error(f"自动恢复失败: {e}")
    
    async def _cleanup_old_data(self):
        """清理旧数据"""
        try:
            current_time = time.time()
            
            # 清理超过24小时的配置历史
            self.config_history = [
                config for config in self.config_history
                if current_time - config.get('timestamp', 0) < 86400
            ]
            
            # 清理长时间未使用的模型性能数据
            unused_models = []
            for model_name, perf in self.model_performance.items():
                if current_time - perf.last_used > 86400:  # 24小时未使用
                    unused_models.append(model_name)
            
            for model_name in unused_models:
                del self.model_performance[model_name]
            
            if unused_models:
                logger.info(f"清理了{len(unused_models)}个未使用的模型数据")
        
        except Exception as e:
            logger.warning(f"数据清理失败: {e}")
    
    async def _trigger_alert(self, alert_type: str, data: Dict[str, Any]):
        """触发告警"""
        alert_data = {
            'type': alert_type,
            'timestamp': time.time(),
            'data': data,
            'service_status': self.status.value
        }
        
        logger.warning(f"触发告警 [{alert_type}]: {data}")
        
        # 调用告警回调
        for callback in self.alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                logger.error(f"告警回调失败: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """添加告警回调"""
        self.alert_callbacks.append(callback)
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """生成文本（带指标记录）"""
        start_time = time.time()
        
        try:
            response = await self.interface.generate_text(
                prompt=prompt,
                model=model,
                parameters=parameters,
                **kwargs
            )
            
            # 记录成功指标
            await self._record_success_metrics(response.model_used, time.time() - start_time)
            
            return {
                'success': True,
                'response': response,
                'metrics': {
                    'response_time': time.time() - start_time,
                    'model_used': response.model_used,
                    'tokens_used': response.tokens_used
                }
            }
            
        except Exception as e:
            # 记录失败指标
            await self._record_failure_metrics(model or 'unknown', time.time() - start_time)
            
            return {
                'success': False,
                'error': str(e),
                'metrics': {
                    'response_time': time.time() - start_time,
                    'model_used': model or 'unknown'
                }
            }
    
    async def _record_success_metrics(self, model_name: str, response_time: float):
        """记录成功指标"""
        if model_name in self.model_performance:
            perf = self.model_performance[model_name]
            perf.total_requests += 1
            perf.last_used = time.time()
            
            # 更新平均响应时间
            if perf.avg_response_time == 0:
                perf.avg_response_time = response_time
            else:
                perf.avg_response_time = (perf.avg_response_time + response_time) / 2
    
    async def _record_failure_metrics(self, model_name: str, response_time: float):
        """记录失败指标"""
        if model_name in self.model_performance:
            perf = self.model_performance[model_name]
            perf.total_requests += 1
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'status': self.status.value,
            'start_time': self.start_time,
            'uptime': time.time() - self.start_time if self.start_time else 0,
            'metrics': asdict(self.metrics),
            'model_performance': {
                name: asdict(perf) for name, perf in self.model_performance.items()
            },
            'configuration': asdict(self.interface_config),
            'available_models': self.interface.get_model_list(),
            'timestamp': time.time()
        }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """获取详细指标"""
        return {
            'service_metrics': asdict(self.metrics),
            'model_performance': {
                name: asdict(perf) for name, perf in self.model_performance.items()
            },
            'system_status': self.interface.get_system_status(),
            'monitoring_tasks': len(self.monitoring_tasks),
            'alert_callbacks': len(self.alert_callbacks),
            'config_history_count': len(self.config_history)
        }
    
    def update_configuration(self, **kwargs):
        """更新配置"""
        old_config = asdict(self.interface_config)
        
        for key, value in kwargs.items():
            if hasattr(self.interface_config, key):
                setattr(self.interface_config, key, value)
        
        # 记录配置变更
        config_change = {
            'timestamp': time.time(),
            'old_config': old_config,
            'new_config': asdict(self.interface_config),
            'changes': kwargs
        }
        self.config_history.append(config_change)
        
        logger.info(f"配置已更新: {kwargs}")
    
    def set_routing_strategy(self, strategy: RoutingStrategy):
        """设置路由策略"""
        self.interface.orchestrator.set_routing_strategy(strategy)
        logger.info(f"路由策略已设置为: {strategy.value}")
    
    def set_load_balance_algorithm(self, algorithm: LoadBalanceAlgorithm):
        """设置负载均衡算法"""
        self.interface.orchestrator.set_load_balance_algorithm(algorithm)
        logger.info(f"负载均衡算法已设置为: {algorithm.value}")
    
    def enable_caching(self):
        """启用缓存"""
        self.interface.orchestrator.enable_cache()
        logger.info("缓存已启用")
    
    def disable_caching(self):
        """禁用缓存"""
        self.interface.orchestrator.disable_cache()
        logger.info("缓存已禁用")
    
    async def clear_cache(self):
        """清空缓存"""
        await self.interface.orchestrator.clear_cache()
        logger.info("缓存已清空")
    
    def add_model(self, model_config: ModelConfig):
        """添加模型"""
        config.add_model(model_config)
        logger.info(f"已添加模型: {model_config.name}")
    
    def remove_model(self, model_name: str):
        """移除模型"""
        config.remove_model(model_name)
        if model_name in self.model_performance:
            del self.model_performance[model_name]
        logger.info(f"已移除模型: {model_name}")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            interface_health = await self.interface.health_check()
            
            return {
                'manager_healthy': self.status == ServiceStatus.RUNNING,
                'interface_health': interface_health,
                'uptime': time.time() - self.start_time if self.start_time else 0,
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'manager_healthy': False,
                'error': str(e),
                'timestamp': time.time()
            }
    
    async def shutdown(self):
        """关闭管理系统"""
        logger.info("开始关闭LLM管理系统")
        
        await self.stop()
        
        logger.info("LLM管理系统已关闭")
    
    def export_metrics(self, format: str = 'json') -> str:
        """导出指标"""
        metrics_data = self.get_detailed_metrics()
        
        if format == 'json':
            return json.dumps(metrics_data, indent=2, ensure_ascii=False)
        elif format == 'csv':
            # 简化的CSV导出
            lines = ['Model,Total Requests,Success Rate,Avg Response Time,Error Rate']
            for name, perf in self.model_performance.items():
                lines.append(f"{name},{perf.total_requests},{perf.success_rate:.3f},{perf.avg_response_time:.3f},{perf.error_rate:.3f}")
            return '\n'.join(lines)
        else:
            raise ValueError(f"不支持的导出格式: {format}")