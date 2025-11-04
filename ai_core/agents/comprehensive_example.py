"""
多任务并行处理系统综合示例
展示如何使用所有功能模块构建完整的并行处理系统
"""

import asyncio
import logging
import time
import random
from typing import List, Dict, Any
import json

# 导入并行处理系统模块
from task_scheduler import TaskScheduler, Task, TaskPriority, create_task
from resource_manager import ResourceManager, ResourceRequest, ResourceType
from execution_pool import ExecutionPool, LoadBalancingStrategy
from result_merger import ResultMerger, ResultAggregator, TaskResult, MergeRule, MergeStrategy, AggregationType
from performance_monitor import PerformanceMonitor, MetricsCollector
from fault_tolerance import TaskRecovery, FaultTolerance, FailureType

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParallelProcessingSystem:
    """多任务并行处理系统"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # 初始化各个组件
        self.task_scheduler = TaskScheduler(
            max_workers=self.config.get('max_workers', 10),
            enable_monitoring=True
        )
        
        self.resource_manager = ResourceManager(
            enable_monitoring=True
        )
        
        self.execution_pool = ExecutionPool(
            name="MainPool",
            max_workers=self.config.get('max_workers', 10),
            min_workers=self.config.get('min_workers', 2),
            load_balancing_strategy=LoadBalancingStrategy.LEAST_CONNECTIONS,
            enable_monitoring=True
        )
        
        self.result_merger = ResultAggregator(
            max_workers=5
        )
        
        self.performance_monitor = PerformanceMonitor(
            collection_interval=5.0,
            alert_threshold_cpu=80.0,
            alert_threshold_memory=85.0
        )
        
        self.task_recovery = TaskRecovery(
            max_retry_attempts=3,
            enable_circuit_breaker=True
        )
        
        self.fault_tolerance = FaultTolerance(self.task_recovery)
        
        # 系统状态
        self.running = False
        self.system_stats = {
            'start_time': None,
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_execution_time': 0.0
        }
        
        logger.info("多任务并行处理系统初始化完成")
    
    async def start(self):
        """启动系统"""
        if self.running:
            logger.warning("系统已在运行")
            return
        
        self.running = True
        self.system_stats['start_time'] = time.time()
        
        # 启动各个组件
        await self.task_scheduler.start()
        await self.resource_manager.start_monitoring()
        await self.execution_pool.start()
        await self.performance_monitor.start_monitoring()
        
        # 注册健康检查
        await self._register_health_checks()
        
        logger.info("多任务并行处理系统已启动")
    
    async def stop(self):
        """停止系统"""
        if not self.running:
            return
        
        self.running = False
        
        # 停止各个组件
        await self.task_scheduler.stop()
        await self.resource_manager.stop_monitoring()
        await self.execution_pool.stop()
        await self.performance_monitor.stop_monitoring()
        
        # 生成最终报告
        await self._generate_final_report()
        
        logger.info("多任务并行处理系统已停止")
    
    async def submit_task(
        self,
        task_func,
        args: tuple = (),
        kwargs: dict = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float = None,
        resource_requirements: dict = None,
        task_name: str = None
    ) -> str:
        """提交任务"""
        kwargs = kwargs or {}
        resource_requirements = resource_requirements or {}
        task_name = task_name or f"Task_{int(time.time())}"
        
        # 创建任务
        task = create_task(
            name=task_name,
            func=task_func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout
        )
        
        # 请求资源
        if resource_requirements:
            resource_request = ResourceRequest(
                task_id=task.id,
                resource_requirements=resource_requirements,
                priority=priority.value
            )
            
            success = await self.resource_manager.request_resources(resource_request)
            if not success:
                logger.warning(f"资源分配失败: {task_name}")
                return None
        
        # 提交到调度器
        task_id = await self.task_scheduler.submit_task(task)
        
        self.system_stats['total_tasks'] += 1
        
        logger.info(f"任务已提交: {task_name} (ID: {task_id})")
        return task_id
    
    async def submit_batch(
        self,
        tasks: List[dict]
    ) -> List[str]:
        """批量提交任务"""
        task_ids = []
        
        for task_config in tasks:
            task_id = await self.submit_task(**task_config)
            if task_id:
                task_ids.append(task_id)
        
        logger.info(f"批量提交完成，共 {len(task_ids)} 个任务")
        return task_ids
    
    async def wait_for_completion(self, timeout: float = None) -> bool:
        """等待所有任务完成"""
        start_time = time.time()
        
        while True:
            # 检查任务状态
            running_tasks = self.task_scheduler.get_running_tasks()
            pending_tasks = self.task_scheduler.get_pending_tasks()
            
            if not running_tasks and not pending_tasks:
                return True
            
            # 检查超时
            if timeout and (time.time() - start_time) >= timeout:
                logger.warning("等待任务完成超时")
                return False
            
            await asyncio.sleep(1)
    
    def get_task_result(self, task_id: str) -> Any:
        """获取任务结果"""
        return self.task_scheduler.get_task_result(task_id)
    
    def get_task_status(self, task_id: str):
        """获取任务状态"""
        return self.task_scheduler.get_task_status(task_id)
    
    async def merge_results(
        self,
        task_ids: List[str],
        merge_rule: str = "default"
    ) -> Any:
        """合并任务结果"""
        # 收集结果
        results = []
        for task_id in task_ids:
            result = self.get_task_result(task_id)
            if result is not None:
                task_result = TaskResult(
                    task_id=task_id,
                    result=result,
                    status="success",
                    execution_time=1.0  # 实际应该从任务调度器获取
                )
                results.append(task_result)
        
        # 添加到结果合并器
        self.result_merger.add_results_batch(results)
        
        # 执行合并
        merged_result = await self.result_merger.merge_results(
            task_ids,
            rule_name=merge_rule
        )
        
        return merged_result
    
    def get_system_status(self) -> dict:
        """获取系统状态"""
        return {
            'running': self.running,
            'system_stats': self.system_stats,
            'task_scheduler_stats': self.task_scheduler.get_statistics(),
            'resource_manager_stats': self.resource_manager.get_all_statistics(),
            'execution_pool_stats': self.execution_pool.get_statistics(),
            'performance_stats': self.performance_monitor.get_performance_statistics(),
            'recovery_stats': self.task_recovery.get_recovery_statistics()
        }
    
    async def _register_health_checks(self):
        """注册健康检查"""
        async def system_health_check():
            """系统整体健康检查"""
            return self.running and len(self.task_scheduler.get_running_tasks()) < 100
        
        async def resource_health_check():
            """资源健康检查"""
            stats = self.resource_manager.get_all_statistics()
            return stats['manager_stats']['pending_requests'] < 10
        
        async def performance_health_check():
            """性能健康检查"""
            cpu_usage = self.performance_monitor.metrics_collector.get_current_value('system.cpu.usage')
            return cpu_usage and cpu_usage < 90
        
        self.fault_tolerance.register_health_check("system", system_health_check)
        self.fault_tolerance.register_health_check("resources", resource_health_check)
        self.fault_tolerance.register_health_check("performance", performance_health_check)
    
    async def _generate_final_report(self):
        """生成最终报告"""
        report = {
            'report_time': time.time(),
            'system_uptime': time.time() - self.system_stats['start_time'],
            'final_status': self.get_system_status(),
            'performance_report': self.performance_monitor.get_performance_statistics(),
            'optimization_recommendations': self.performance_monitor.get_optimization_recommendations(),
            'recovery_log': self.task_recovery.get_recovery_statistics()
        }
        
        # 保存报告
        with open('system_final_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info("最终报告已生成: system_final_report.json")


# 示例任务函数
def cpu_intensive_task(n: int) -> int:
    """CPU密集型任务"""
    result = 0
    for i in range(n):
        result += i * i
    return result


def io_bound_task(duration: float) -> str:
    """I/O密集型任务"""
    import time
    time.sleep(duration)
    return f"I/O任务完成，耗时 {duration}s"


def memory_intensive_task(size: int) -> list:
    """内存密集型任务"""
    return [random.random() for _ in range(size)]


def network_task(url: str) -> dict:
    """网络任务"""
    import requests
    try:
        response = requests.get(url, timeout=10)
        return {
            'url': url,
            'status_code': response.status_code,
            'content_length': len(response.content)
        }
    except Exception as e:
        raise Exception(f"网络请求失败: {e}")


def data_processing_task(data: list) -> dict:
    """数据处理任务"""
    if not data:
        return {'error': 'No data provided'}
    
    return {
        'count': len(data),
        'sum': sum(data),
        'average': sum(data) / len(data),
        'max': max(data),
        'min': min(data)
    }


# 综合示例
async def comprehensive_example():
    """综合示例"""
    logger.info("开始多任务并行处理系统综合示例")
    
    # 创建系统
    system = ParallelProcessingSystem({
        'max_workers': 8,
        'min_workers': 2
    })
    
    try:
        # 启动系统
        await system.start()
        
        # 准备任务配置
        tasks = [
            # CPU密集型任务
            {
                'task_func': cpu_intensive_task,
                'args': (100000,),
                'priority': TaskPriority.HIGH,
                'timeout': 30.0,
                'resource_requirements': {ResourceType.CPU: 1.0},
                'task_name': 'CPU密集型任务1'
            },
            {
                'task_func': cpu_intensive_task,
                'args': (50000,),
                'priority': TaskPriority.NORMAL,
                'timeout': 20.0,
                'resource_requirements': {ResourceType.CPU: 1.0},
                'task_name': 'CPU密集型任务2'
            },
            
            # I/O密集型任务
            {
                'task_func': io_bound_task,
                'args': (2.0,),
                'priority': TaskPriority.NORMAL,
                'timeout': 10.0,
                'task_name': 'I/O任务1'
            },
            {
                'task_func': io_bound_task,
                'args': (1.5,),
                'priority': TaskPriority.LOW,
                'timeout': 10.0,
                'task_name': 'I/O任务2'
            },
            
            # 内存密集型任务
            {
                'task_func': memory_intensive_task,
                'args': (10000,),
                'priority': TaskPriority.NORMAL,
                'timeout': 15.0,
                'resource_requirements': {ResourceType.MEMORY: 0.5},
                'task_name': '内存密集型任务'
            },
            
            # 数据处理任务
            {
                'task_func': data_processing_task,
                'args': ([random.randint(1, 100) for _ in range(100)],),
                'priority': TaskPriority.HIGH,
                'timeout': 5.0,
                'task_name': '数据处理任务1'
            },
            {
                'task_func': data_processing_task,
                'args': ([random.randint(1, 100) for _ in range(200)],),
                'priority': TaskPriority.HIGH,
                'timeout': 5.0,
                'task_name': '数据处理任务2'
            }
        ]
        
        # 批量提交任务
        task_ids = await system.submit_batch(tasks)
        logger.info(f"已提交 {len(task_ids)} 个任务")
        
        # 等待任务完成
        await system.wait_for_completion(timeout=60)
        
        # 获取结果
        logger.info("任务执行结果:")
        for i, task_id in enumerate(task_ids):
            result = system.get_task_result(task_id)
            status = system.get_task_status(task_id)
            logger.info(f"任务 {i+1}: 状态={status.value}, 结果={result}")
        
        # 合并结果
        data_task_ids = [task_ids[5], task_ids[6]]  # 数据处理任务
        if len(data_task_ids) == 2:
            merged_result = await system.merge_results(data_task_ids, "stats_average")
            logger.info(f"合并结果: {merged_result}")
        
        # 获取系统状态
        system_status = system.get_system_status()
        logger.info("系统状态:")
        logger.info(f"- 运行状态: {system_status['running']}")
        logger.info(f"- 总任务数: {system_status['system_stats']['total_tasks']}")
        logger.info(f"- 平均执行时间: {system_status['system_stats']['avg_execution_time']:.2f}s")
        
        # 获取性能建议
        recommendations = system.performance_monitor.get_optimization_recommendations()
        if recommendations:
            logger.info("性能优化建议:")
            for rec in recommendations:
                logger.info(f"- {rec['title']}: {rec['description']}")
        
    except Exception as e:
        logger.error(f"示例执行错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 停止系统
        await system.stop()
    
    logger.info("多任务并行处理系统综合示例完成")


# 性能测试示例
async def performance_test():
    """性能测试示例"""
    logger.info("开始性能测试")
    
    system = ParallelProcessingSystem({'max_workers': 4})
    
    try:
        await system.start()
        
        # 提交大量小任务
        task_configs = []
        for i in range(50):
            task_configs.append({
                'task_func': lambda x: x * x,
                'args': (i,),
                'priority': TaskPriority.NORMAL,
                'timeout': 5.0,
                'task_name': f'性能测试任务_{i}'
            })
        
        start_time = time.time()
        task_ids = await system.submit_batch(task_configs)
        
        await system.wait_for_completion(timeout=30)
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = len(task_ids) / total_time
        
        logger.info(f"性能测试结果:")
        logger.info(f"- 总任务数: {len(task_ids)}")
        logger.info(f"- 总耗时: {total_time:.2f}s")
        logger.info(f"- 吞吐量: {throughput:.2f} 任务/秒")
        
        # 获取系统资源使用情况
        resource_stats = system.resource_manager.get_all_statistics()
        logger.info(f"- CPU使用率: {resource_stats['resource_pools']['cpu']['usage_percentage']:.1f}%")
        logger.info(f"- 内存使用率: {resource_stats['resource_pools']['memory']['usage_percentage']:.1f}%")
        
    finally:
        await system.stop()


if __name__ == "__main__":
    # 运行综合示例
    asyncio.run(comprehensive_example())
    
    # 运行性能测试
    asyncio.run(performance_test())