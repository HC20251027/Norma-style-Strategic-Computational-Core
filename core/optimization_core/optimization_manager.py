"""
统一优化管理器

整合所有优化组件，提供统一的调度、执行、资源管理和监控功能
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from concurrent.futures import ThreadPoolExecutor

# 导入优化组件
from .scheduler.agent_scheduler import AgentScheduler, Task as SchedulerTask, TaskType, TaskComplexity, AgentCapability
from .scheduler.task_queue import TaskQueue, QueueItem, QueueType
from .scheduler.priority_manager import PriorityManager, PriorityTask, PriorityLevel

from .executor.mcp_engine import MCPEngine, MCPRequest, ExecutionContext, ExecutionMode
from .executor.concurrent_executor import ConcurrentExecutor, Task as ConcurrentTask
from .executor.pipeline_executor import PipelineExecutor, PipelineStageConfig, PipelineStage

from .resources.resource_manager import ResourceManager, ResourceRequest, ResourceType, ResourceAllocation
from .resources.load_balancer import LoadBalancer, BackendServer, LoadBalanceRequest, LoadBalanceStrategy
from .resources.resource_monitor import ResourceMonitor, Metric, AlertRule, AlertSeverity

from .monitoring.execution_monitor import ExecutionMonitor, ExecutionRecord, ExecutionType, ExecutionState
from .monitoring.performance_stats import PerformanceStats, PerformanceMetric, MetricCategory
from .monitoring.metrics_collector import MetricsCollector, CollectedMetric, MetricFormat

logger = logging.getLogger(__name__)


class OptimizationMode(Enum):
    """优化模式"""
    PERFORMANCE = "performance"  # 性能优先
    RESOURCE_EFFICIENCY = "resource_efficiency"  # 资源效率优先
    RELIABILITY = "reliability"  # 可靠性优先
    BALANCED = "balanced"  # 平衡模式
    CUSTOM = "custom"  # 自定义


@dataclass
class OptimizationConfig:
    """优化配置"""
    mode: OptimizationMode = OptimizationMode.BALANCED
    max_concurrent_tasks: int = 10
    resource_allocation_strategy: str = "fair"
    load_balancing_strategy: str = "round_robin"
    monitoring_enabled: bool = True
    auto_scaling_enabled: bool = False
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class OptimizationManager:
    """统一优化管理器"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        
        # 组件初始化
        self.scheduler = AgentScheduler(max_workers=self.config.max_concurrent_tasks)
        self.task_queue = TaskQueue(max_size=10000)
        self.priority_manager = PriorityManager(max_tasks=5000)
        
        self.mcp_engine = MCPEngine(max_concurrent=self.config.max_concurrent_tasks)
        self.concurrent_executor = ConcurrentExecutor(max_workers=self.config.max_concurrent_tasks)
        self.pipeline_executor = PipelineExecutor(max_concurrent_pipelines=5)
        
        self.resource_manager = ResourceManager()
        self.load_balancer = LoadBalancer(name="optimization_lb")
        self.resource_monitor = ResourceMonitor(collection_interval=30.0)
        
        self.execution_monitor = ExecutionMonitor(max_history=10000)
        self.performance_stats = PerformanceStats(retention_hours=168, aggregation_interval=60)
        self.metrics_collector = MetricsCollector(max_collectors=50)
        
        # 状态管理
        self.is_running = False
        self.start_time: Optional[float] = None
        
        # 统计信息
        self.stats = {
            'total_tasks_processed': 0,
            'total_executions': 0,
            'average_response_time': 0.0,
            'resource_utilization': 0.0,
            'success_rate': 0.0,
            'optimization_score': 0.0
        }
        
        # 回调函数
        self.optimization_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks)
        
        # 内部状态
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """启动优化管理器"""
        if self.is_running:
            logger.warning("优化管理器已在运行")
            return
        
        try:
            self.start_time = time.time()
            self.is_running = True
            
            # 启动所有组件
            await self.scheduler.start()
            await self.task_queue.start()
            await self.priority_manager.start()
            
            await self.mcp_engine.start()
            await self.concurrent_executor.start()
            await self.pipeline_executor.start()
            
            await self.resource_manager.start_monitoring()
            await self.load_balancer.start()
            await self.resource_monitor.start()
            
            await self.execution_monitor.start()
            await self.performance_stats.start()
            await self.metrics_collector.start()
            
            # 设置默认配置
            await self._setup_default_configuration()
            
            # 启动优化循环
            asyncio.create_task(self._optimization_loop())
            
            logger.info("优化管理器已启动")
            
        except Exception as e:
            logger.error(f"启动优化管理器失败: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """停止优化管理器"""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            self._shutdown_event.set()
            
            # 停止所有组件
            await self.scheduler.stop()
            await self.task_queue.stop()
            await self.priority_manager.stop()
            
            await self.mcp_engine.stop()
            await self.concurrent_executor.stop()
            await self.pipeline_executor.stop()
            
            await self.resource_manager.stop_monitoring()
            await self.load_balancer.stop()
            await self.resource_monitor.stop()
            
            await self.execution_monitor.stop()
            await self.performance_stats.stop()
            await self.metrics_collector.stop()
            
            # 关闭线程池
            self.executor.shutdown(wait=True)
            
            logger.info("优化管理器已停止")
            
        except Exception as e:
            logger.error(f"停止优化管理器失败: {e}")
    
    async def _setup_default_configuration(self):
        """设置默认配置"""
        try:
            # 设置调度策略
            await self.scheduler.set_scheduling_strategy('capability_matching')
            
            # 设置优先级管理
            self.priority_manager.set_scheduling_mode('preemptive')
            
            # 设置负载均衡策略
            self.load_balancer.set_load_balance_strategy(
                LoadBalanceStrategy.ROUND_ROBIN
            )
            
            # 设置告警阈值
            default_thresholds = {
                'cpu_usage': 80.0,
                'memory_usage': 85.0,
                'disk_usage': 90.0,
                'error_rate': 5.0,
                'response_time': 1000.0
            }
            
            for metric, threshold in default_thresholds.items():
                rule = AlertRule(
                    rule_id=f"threshold_{metric}",
                    name=f"{metric}阈值告警",
                    metric_name=f"system.{metric}",
                    condition=">",
                    threshold=threshold,
                    severity=AlertSeverity.WARNING
                )
                self.resource_monitor.add_alert_rule(rule)
            
            # 注册默认收集器
            await self._register_default_collectors()
            
            logger.info("默认配置设置完成")
            
        except Exception as e:
            logger.error(f"设置默认配置失败: {e}")
    
    async def _register_default_collectors(self):
        """注册默认收集器"""
        try:
            # 系统指标收集器
            from .monitoring.metrics_collector import CollectorConfig, CollectorType
            system_config = CollectorConfig(
                collector_id="system_metrics",
                name="系统指标收集器",
                collector_type=CollectorType.SYSTEM,
                interval=30.0
            )
            self.metrics_collector.register_collector(system_config)
            
            # 应用指标收集器
            app_config = CollectorConfig(
                collector_id="app_metrics",
                name="应用指标收集器",
                collector_type=CollectorType.APPLICATION,
                interval=60.0
            )
            self.metrics_collector.register_collector(app_config)
            
        except Exception as e:
            logger.error(f"注册默认收集器失败: {e}")
    
    async def submit_task(self, task_data: Dict[str, Any]) -> str:
        """提交任务"""
        try:
            # 创建任务
            task = SchedulerTask(
                id=f"task_{int(time.time() * 1000)}",
                type=TaskType(task_data.get('type', 'computation')),
                complexity=TaskComplexity(task_data.get('complexity', 'moderate')),
                priority=task_data.get('priority', 2),
                payload=task_data.get('payload', {}),
                estimated_duration=task_data.get('estimated_duration', 0.0),
                resource_requirements=task_data.get('resource_requirements', {}),
                deadline=task_data.get('deadline')
            )
            
            # 提交到调度器
            task_id = await self.scheduler.submit_task(task)
            
            # 记录执行监控
            self.execution_monitor.start_execution(
                execution_id=task_id,
                execution_type=ExecutionType.TASK,
                task_type=task.type.value,
                metadata=task_data
            )
            
            self.stats['total_tasks_processed'] += 1
            
            logger.info(f"任务已提交: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"提交任务失败: {e}")
            raise
    
    async def execute_mcp_request(self, request_data: Dict[str, Any]) -> str:
        """执行MCP请求"""
        try:
            # 创建MCP请求
            request = MCPRequest(
                id=f"mcp_{int(time.time() * 1000)}",
                method=request_data.get('method', ''),
                params=request_data.get('params', {}),
                timeout=request_data.get('timeout', 30.0),
                priority=request_data.get('priority', 0)
            )
            
            # 创建执行上下文
            context = ExecutionContext(
                request_id=request.id,
                session_id=request_data.get('session_id'),
                user_id=request_data.get('user_id'),
                environment=request_data.get('environment', {}),
                resources=request_data.get('resources', {})
            )
            
            # 执行请求
            execution_mode = ExecutionMode(request_data.get('mode', 'async'))
            response = await self.mcp_engine.execute(request, context, execution_mode)
            
            # 记录执行监控
            self.execution_monitor.complete_execution(
                execution_id=request.id,
                status=ExecutionState.COMPLETED if response.status.value == 'completed' else ExecutionState.FAILED,
                result=response.result,
                error=response.error
            )
            
            self.stats['total_executions'] += 1
            
            logger.info(f"MCP请求已执行: {request.id}")
            return request.id
            
        except Exception as e:
            logger.error(f"执行MCP请求失败: {e}")
            raise
    
    async def allocate_resources(self, request_data: Dict[str, Any]) -> Optional[str]:
        """分配资源"""
        try:
            # 创建资源请求
            request = ResourceRequest(
                request_id=f"res_{int(time.time() * 1000)}",
                consumer_id=request_data.get('consumer_id', 'default'),
                resource_type=ResourceType(request_data.get('resource_type', 'cpu')),
                amount=request_data.get('amount', 1.0),
                priority=request_data.get('priority', 0),
                duration=request_data.get('duration'),
                metadata=request_data.get('metadata', {})
            )
            
            # 分配资源
            allocation = await self.resource_manager.request_resource(request)
            
            if allocation:
                logger.info(f"资源分配成功: {allocation.allocation_id}")
                return allocation.allocation_id
            else:
                logger.warning(f"资源分配失败: {request.request_id}")
                return None
                
        except Exception as e:
            logger.error(f"分配资源失败: {e}")
            return None
    
    async def load_balance(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """负载均衡"""
        try:
            # 创建负载均衡请求
            request = LoadBalanceRequest(
                request_id=f"lb_{int(time.time() * 1000)}",
                client_ip=request_data.get('client_ip', '127.0.0.1'),
                request_data=request_data.get('request_data', {}),
                priority=request_data.get('priority', 0),
                metadata=request_data.get('metadata', {})
            )
            
            # 执行负载均衡
            result = await self.load_balancer.select_server(request)
            
            if result.selected_server:
                logger.info(f"负载均衡成功: {result.selected_server.server_id}")
                return {
                    'server_id': result.selected_server.server_id,
                    'host': result.selected_server.host,
                    'port': result.selected_server.port,
                    'selection_time': result.selection_time
                }
            else:
                logger.warning("负载均衡失败: 没有可用的服务器")
                return None
                
        except Exception as e:
            logger.error(f"负载均衡失败: {e}")
            return None
    
    async def _optimization_loop(self):
        """优化循环"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # 每分钟优化一次
                
                # 收集当前状态
                current_stats = await self._collect_current_stats()
                
                # 执行优化策略
                await self._apply_optimization_strategies(current_stats)
                
                # 更新性能指标
                await self._update_performance_metrics(current_stats)
                
                # 触发优化回调
                await self._trigger_optimization_callbacks(current_stats)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"优化循环异常: {e}")
    
    async def _collect_current_stats(self) -> Dict[str, Any]:
        """收集当前统计信息"""
        try:
            stats = {
                'timestamp': time.time(),
                'scheduler': self.scheduler.get_scheduling_stats(),
                'task_queue': await self.task_queue.get_queue_stats(),
                'priority': self.priority_manager.get_priority_stats(),
                'mcp_engine': self.mcp_engine.get_execution_stats(),
                'concurrent_executor': self.concurrent_executor.get_executor_stats(),
                'pipeline_executor': self.pipeline_executor.get_executor_stats(),
                'resource_manager': self.resource_manager.get_resource_manager_stats(),
                'load_balancer': self.load_balancer.get_load_balancer_stats(),
                'resource_monitor': self.resource_monitor.get_monitor_stats(),
                'execution_monitor': self.execution_monitor.get_monitor_stats(),
                'metrics_collector': self.metrics_collector.get_collection_stats()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"收集当前统计失败: {e}")
            return {}
    
    async def _apply_optimization_strategies(self, stats: Dict[str, Any]):
        """应用优化策略"""
        try:
            mode = self.config.mode
            
            if mode == OptimizationMode.PERFORMANCE:
                await self._optimize_for_performance(stats)
            elif mode == OptimizationMode.RESOURCE_EFFICIENCY:
                await self._optimize_for_resource_efficiency(stats)
            elif mode == OptimizationMode.RELIABILITY:
                await self._optimize_for_reliability(stats)
            elif mode == OptimizationMode.BALANCED:
                await self._optimize_balanced(stats)
            
        except Exception as e:
            logger.error(f"应用优化策略失败: {e}")
    
    async def _optimize_for_performance(self, stats: Dict[str, Any]):
        """性能优先优化"""
        try:
            # 增加并发度
            if stats['scheduler']['pending_tasks'] > 50:
                await self.scheduler.set_scheduling_strategy('performance_optimization')
            
            # 优化负载均衡
            if stats['load_balancer']['average_response_time'] > 1000:
                self.load_balancer.set_load_balance_strategy(LoadBalanceStrategy.LEAST_RESPONSE_TIME)
            
        except Exception as e:
            logger.error(f"性能优化失败: {e}")
    
    async def _optimize_for_resource_efficiency(self, stats: Dict[str, Any]):
        """资源效率优先优化"""
        try:
            # 优化资源分配
            if stats['resource_manager']['resource_efficiency'] < 70:
                # 可以考虑释放未使用的资源或调整分配策略
                pass
            
            # 优化负载均衡
            if stats['load_balancer']['average_response_time'] > 500:
                self.load_balancer.set_load_balance_strategy(LoadBalanceStrategy.LEAST_CONNECTIONS)
            
        except Exception as e:
            logger.error(f"资源效率优化失败: {e}")
    
    async def _optimize_for_reliability(self, stats: Dict[str, Any]):
        """可靠性优先优化"""
        try:
            # 检查错误率
            if stats['execution_monitor']['overall_metrics']['error_rate'] > 5:
                # 降低并发度以提高稳定性
                pass
            
            # 优化调度策略
            await self.scheduler.set_scheduling_strategy('capability_matching')
            
        except Exception as e:
            logger.error(f"可靠性优化失败: {e}")
    
    async def _optimize_balanced(self, stats: Dict[str, Any]):
        """平衡优化"""
        try:
            # 综合考虑多个指标
            if (stats['scheduler']['pending_tasks'] > 30 and 
                stats['resource_manager']['resource_efficiency'] < 80):
                await self.scheduler.set_scheduling_strategy('load_balancing')
            
        except Exception as e:
            logger.error(f"平衡优化失败: {e}")
    
    async def _update_performance_metrics(self, stats: Dict[str, Any]):
        """更新性能指标"""
        try:
            # 记录综合性能指标
            self.performance_stats.add_metric(
                name="optimization.total_tasks",
                value=stats['scheduler']['total_scheduled'],
                category=MetricCategory.APPLICATION
            )
            
            self.performance_stats.add_metric(
                name="optimization.resource_efficiency",
                value=stats['resource_manager']['resource_efficiency'],
                category=MetricCategory.SYSTEM
            )
            
            self.performance_stats.add_metric(
                name="optimization.success_rate",
                value=stats['execution_monitor']['overall_metrics']['success_rate'],
                category=MetricCategory.APPLICATION
            )
            
        except Exception as e:
            logger.error(f"更新性能指标失败: {e}")
    
    async def _trigger_optimization_callbacks(self, stats: Dict[str, Any]):
        """触发优化回调"""
        try:
            for callback in self.optimization_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(stats)
                else:
                    callback(stats)
        except Exception as e:
            logger.error(f"触发优化回调失败: {e}")
    
    def add_optimization_callback(self, callback: Callable):
        """添加优化回调"""
        self.optimization_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """添加错误回调"""
        self.error_callbacks.append(callback)
    
    def set_optimization_mode(self, mode: OptimizationMode):
        """设置优化模式"""
        self.config.mode = mode
        logger.info(f"优化模式已设置为: {mode.value}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """获取优化状态"""
        try:
            uptime = time.time() - self.start_time if self.start_time else 0
            
            return {
                'is_running': self.is_running,
                'uptime': uptime,
                'optimization_mode': self.config.mode.value,
                'stats': self.stats.copy(),
                'components': {
                    'scheduler': self.scheduler.get_scheduling_stats(),
                    'task_queue': self.task_queue.get_metrics(),
                    'mcp_engine': self.mcp_engine.get_execution_stats(),
                    'resource_manager': self.resource_manager.get_resource_manager_stats(),
                    'load_balancer': self.load_balancer.get_load_balancer_stats(),
                    'execution_monitor': self.execution_monitor.get_monitor_stats(),
                    'metrics_collector': self.metrics_collector.get_collection_stats()
                }
            }
        except Exception as e:
            logger.error(f"获取优化状态失败: {e}")
            return {}
    
    def get_optimization_report(self, hours: int = 24) -> Dict[str, Any]:
        """获取优化报告"""
        try:
            # 生成性能报告
            performance_report = self.performance_stats.generate_performance_report(hours)
            
            # 收集各种统计信息
            status = self.get_optimization_status()
            
            return {
                'report_id': f"opt_report_{int(time.time())}",
                'generated_at': time.time(),
                'time_range_hours': hours,
                'optimization_mode': self.config.mode.value,
                'uptime_hours': status.get('uptime', 0) / 3600,
                'performance_report': performance_report.to_dict(),
                'current_status': status,
                'recommendations': performance_report.recommendations
            }
            
        except Exception as e:
            logger.error(f"生成优化报告失败: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            health_status = {
                'overall_health': 'healthy',
                'components': {},
                'issues': []
            }
            
            # 检查各组件健康状态
            components = {
                'scheduler': self.scheduler,
                'task_queue': self.task_queue,
                'mcp_engine': self.mcp_engine,
                'resource_manager': self.resource_manager,
                'load_balancer': self.load_balancer,
                'execution_monitor': self.execution_monitor,
                'metrics_collector': self.metrics_collector
            }
            
            for name, component in components.items():
                try:
                    if hasattr(component, 'get_monitor_stats'):
                        stats = component.get_monitor_stats()
                        health_status['components'][name] = 'healthy'
                    else:
                        health_status['components'][name] = 'unknown'
                except Exception as e:
                    health_status['components'][name] = 'unhealthy'
                    health_status['issues'].append(f"{name}: {str(e)}")
            
            # 检查是否有严重问题
            if health_status['issues']:
                health_status['overall_health'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                'overall_health': 'unhealthy',
                'components': {},
                'issues': [str(e)]
            }