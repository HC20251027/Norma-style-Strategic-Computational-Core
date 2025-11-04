"""
优化调度执行机制使用示例

展示如何使用优化调度执行机制的各个功能
"""

import asyncio
import logging
from typing import Dict, Any

# 导入优化组件
from optimization import (
    OptimizationManager, 
    OptimizationConfig, 
    OptimizationMode,
    AgentScheduler,
    MCPEngine,
    ResourceManager,
    LoadBalancer
)
from optimization.scheduler.agent_scheduler import AgentCapability, Task, TaskType, TaskComplexity
from optimization.executor.mcp_engine import MCPRequest, ExecutionContext, ExecutionMode
from optimization.resources.resource_manager import ResourceRequest, ResourceType
from optimization.resources.load_balancer import BackendServer, LoadBalanceRequest, LoadBalanceStrategy

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_basic_usage():
    """基本使用示例"""
    logger.info("=== 基本使用示例 ===")
    
    # 1. 创建优化配置
    config = OptimizationConfig(
        mode=OptimizationMode.BALANCED,
        max_concurrent_tasks=5,
        monitoring_enabled=True
    )
    
    # 2. 创建优化管理器
    manager = OptimizationManager(config)
    
    # 3. 启动管理器
    await manager.start()
    logger.info("优化管理器已启动")
    
    # 4. 提交计算任务
    task_data = {
        'type': 'computation',
        'complexity': 'moderate',
        'priority': 3,
        'payload': {
            'operation': 'matrix_multiplication',
            'size': 100,
            'data': 'sample_matrix_data'
        },
        'estimated_duration': 5.0,
        'resource_requirements': {ResourceType.CPU: 2.0}
    }
    
    task_id = await manager.submit_task(task_data)
    logger.info(f"计算任务已提交: {task_id}")
    
    # 5. 执行MCP请求
    mcp_request_data = {
        'method': 'tools/call',
        'params': {
            'name': 'calculator',
            'arguments': {
                'operation': 'add',
                'a': 10,
                'b': 20
            }
        },
        'mode': 'async',
        'timeout': 30.0
    }
    
    mcp_id = await manager.execute_mcp_request(mcp_request_data)
    logger.info(f"MCP请求已执行: {mcp_id}")
    
    # 6. 分配资源
    resource_request_data = {
        'consumer_id': 'calculation_worker',
        'resource_type': 'cpu',
        'amount': 2.0,
        'priority': 5,
        'duration': 3600.0,  # 1小时
        'metadata': {'purpose': 'matrix_calculation'}
    }
    
    allocation_id = await manager.allocate_resources(resource_request_data)
    if allocation_id:
        logger.info(f"资源分配成功: {allocation_id}")
    else:
        logger.warning("资源分配失败")
    
    # 7. 负载均衡
    lb_request_data = {
        'client_ip': '192.168.1.100',
        'request_data': {
            'path': '/api/v1/calculate',
            'method': 'POST',
            'body': {'operation': 'multiply', 'values': [1, 2, 3]}
        },
        'priority': 2
    }
    
    lb_result = await manager.load_balance(lb_request_data)
    if lb_result:
        logger.info(f"负载均衡成功: 服务器 {lb_result['server_id']} ({lb_result['host']}:{lb_result['port']})")
    else:
        logger.warning("负载均衡失败")
    
    # 8. 等待处理完成
    await asyncio.sleep(3)
    
    # 9. 获取状态信息
    status = manager.get_optimization_status()
    logger.info(f"系统状态: {status['stats']}")
    
    # 10. 生成优化报告
    report = manager.get_optimization_report(hours=1)
    logger.info(f"优化报告: {len(report.get('recommendations', []))} 条建议")
    
    # 11. 健康检查
    health = await manager.health_check()
    logger.info(f"健康检查: {health['overall_health']}")
    
    # 12. 停止管理器
    await manager.stop()
    logger.info("优化管理器已停止")


async def example_advanced_scheduling():
    """高级调度示例"""
    logger.info("=== 高级调度示例 ===")
    
    # 创建Agent调度器
    scheduler = AgentScheduler(max_workers=8)
    await scheduler.start()
    
    # 注册多个Agent
    agents = [
        AgentCapability(
            agent_id="cpu_worker_1",
            supported_task_types=[TaskType.COMPUTATION, TaskType.ANALYSIS],
            max_concurrent_tasks=3,
            performance_score=1.2,
            specialization_score={
                TaskType.COMPUTATION: 0.9,
                TaskType.ANALYSIS: 0.8
            }
        ),
        AgentCapability(
            agent_id="io_worker_1",
            supported_task_types=[TaskType.IO_INTENSIVE, TaskType.NETWORK_INTENSIVE],
            max_concurrent_tasks=5,
            performance_score=1.0,
            specialization_score={
                TaskType.IO_INTENSIVE: 0.95,
                TaskType.NETWORK_INTENSIVE: 0.9
            }
        ),
        AgentCapability(
            agent_id="memory_worker_1",
            supported_task_types=[TaskType.MEMORY_INTENSIVE, TaskType.MULTIMODAL],
            max_concurrent_tasks=2,
            performance_score=0.8,
            specialization_score={
                TaskType.MEMORY_INTENSIVE: 0.9,
                TaskType.MULTIMODAL: 0.95
            }
        )
    ]
    
    for agent in agents:
        await scheduler.register_agent(agent)
        logger.info(f"注册Agent: {agent.agent_id}")
    
    # 创建不同类型的任务
    tasks = [
        Task(
            id="task_computation_1",
            type=TaskType.COMPUTATION,
            complexity=TaskComplexity.COMPLEX,
            priority=4,
            payload={'operation': 'prime_factorization', 'number': 123456789},
            estimated_duration=10.0,
            resource_requirements={ResourceType.CPU: 3.0}
        ),
        Task(
            id="task_io_1",
            type=TaskType.IO_INTENSIVE,
            complexity=TaskComplexity.MODERATE,
            priority=2,
            payload={'operation': 'file_processing', 'files': ['data1.txt', 'data2.txt']},
            estimated_duration=15.0,
            resource_requirements={ResourceType.DISK: 1.0}
        ),
        Task(
            id="task_network_1",
            type=TaskType.NETWORK_INTENSIVE,
            complexity=TaskComplexity.MODERATE,
            priority=3,
            payload={'operation': 'api_batch_request', 'urls': ['http://api1.com', 'http://api2.com']},
            estimated_duration=8.0,
            resource_requirements={ResourceType.NETWORK: 2.0}
        ),
        Task(
            id="task_memory_1",
            type=TaskType.MEMORY_INTENSIVE,
            complexity=TaskComplexity.CRITICAL,
            priority=5,
            payload={'operation': 'large_dataset_processing', 'size_gb': 5},
            estimated_duration=20.0,
            resource_requirements={ResourceType.MEMORY: 4.0}
        )
    ]
    
    # 提交任务
    for task in tasks:
        task_id = await scheduler.submit_task(task)
        logger.info(f"任务已提交: {task_id} (类型: {task.type.value})")
    
    # 批量调度任务
    decisions = await scheduler.schedule_batch(max_concurrent=3)
    logger.info(f"批量调度完成: {len(decisions)} 个任务")
    
    for decision in decisions:
        logger.info(f"任务 {decision.task_id} -> Agent {decision.assigned_agent_id} "
                   f"(置信度: {decision.confidence_score:.2f})")
    
    # 等待任务执行
    await asyncio.sleep(2)
    
    # 模拟任务完成
    for decision in decisions:
        success = True  # 模拟成功
        execution_time = 5.0 + (hash(decision.task_id) % 10)  # 模拟执行时间
        await scheduler.complete_task(decision.task_id, success, execution_time)
        logger.info(f"任务 {decision.task_id} 已完成")
    
    # 获取调度统计
    stats = scheduler.get_scheduling_stats()
    logger.info(f"调度统计: {stats['scheduling_stats']}")
    
    await scheduler.stop()


async def example_mcp_pipeline():
    """MCP流水线执行示例"""
    logger.info("=== MCP流水线执行示例 ===")
    
    # 创建MCP引擎
    engine = MCPEngine(max_concurrent=5)
    await engine.start()
    
    # 创建流水线请求
    pipeline_request = MCPRequest(
        id="pipeline_1",
        method="pipeline_execute",
        params={
            'pipeline_steps': [
                {
                    'method': 'data_validation',
                    'params': {'schema': 'user_data_schema'}
                },
                {
                    'method': 'data_transformation',
                    'params': {'transformations': ['normalize', 'encode']}
                },
                {
                    'method': 'model_inference',
                    'params': {'model': 'prediction_model', 'version': 'v2'}
                },
                {
                    'method': 'result_postprocessing',
                    'params': {'format': 'json', 'include_metadata': True}
                }
            ]
        },
        timeout=120.0,
        metadata={'pipeline_type': 'ml_inference'}
    )
    
    # 创建上下文
    context = ExecutionContext(
        request_id="pipeline_1",
        session_id="session_ml_001",
        user_id="user_123",
        environment={'pipeline_mode': 'production'},
        resources={'gpu_available': True, 'memory_gb': 16}
    )
    
    # 执行流水线
    response = await engine.execute(pipeline_request, context, ExecutionMode.PIPELINE)
    
    logger.info(f"流水线执行状态: {response.status.value}")
    if response.result:
        logger.info(f"流水线结果: {response.result}")
    
    # 创建批量请求
    batch_request = MCPRequest(
        id="batch_1",
        method="batch_process",
        params={
            'batch_requests': [
                MCPRequest(
                    id="batch_item_1",
                    method="data_analysis",
                    params={'dataset': 'dataset_a'}
                ),
                MCPRequest(
                    id="batch_item_2", 
                    method="data_analysis",
                    params={'dataset': 'dataset_b'}
                ),
                MCPRequest(
                    id="batch_item_3",
                    method="data_analysis", 
                    params={'dataset': 'dataset_c'}
                )
            ]
        },
        timeout=60.0
    )
    
    # 执行批量请求
    batch_response = await engine.execute(batch_request, context, ExecutionMode.BATCH)
    logger.info(f"批量执行状态: {batch_response.status.value}")
    
    # 获取引擎统计
    stats = engine.get_execution_stats()
    logger.info(f"MCP引擎统计: {stats}")
    
    await engine.stop()


async def example_resource_management():
    """资源管理示例"""
    logger.info("=== 资源管理示例 ===")
    
    # 创建资源管理器
    resource_manager = ResourceManager()
    await resource_manager.start_monitoring()
    
    # 添加自定义资源池
    from optimization.resources.resource_manager import Resource, AllocationStrategy
    
    gpu_resources = [
        Resource(
            resource_id="gpu_1",
            resource_type=ResourceType.GPU,
            name="NVIDIA RTX 4090",
            capacity=24.0,  # 24GB显存
            metadata={'model': 'RTX 4090', 'memory': '24GB'}
        ),
        Resource(
            resource_id="gpu_2",
            resource_type=ResourceType.GPU,
            name="NVIDIA RTX 3080",
            capacity=10.0,  # 10GB显存
            metadata={'model': 'RTX 3080', 'memory': '10GB'}
        )
    ]
    
    resource_manager.create_resource_pool(
        pool_id="gpu_pool",
        resource_type=ResourceType.GPU,
        resources=gpu_resources,
        allocation_strategy=AllocationStrategy.PRIORITY,
        auto_scaling=True
    )
    
    # 创建多个资源请求
    requests = [
        ResourceRequest(
            request_id="ml_training_1",
            consumer_id="ml_worker_1",
            resource_type=ResourceType.GPU,
            amount=24.0,  # 需要完整GPU
            priority=8,
            duration=7200.0,  # 2小时
            metadata={'task': 'model_training', 'framework': 'pytorch'}
        ),
        ResourceRequest(
            request_id="inference_1",
            consumer_id="inference_service",
            resource_type=ResourceType.GPU,
            amount=5.0,  # 部分GPU资源
            priority=5,
            duration=3600.0,  # 1小时
            metadata={'task': 'real_time_inference', 'batch_size': 32}
        ),
        ResourceRequest(
            request_id="data_processing_1",
            consumer_id="data_worker",
            resource_type=ResourceType.MEMORY,
            amount=8.0,  # 8GB内存
            priority=3,
            duration=1800.0,  # 30分钟
            metadata={'task': 'data_preprocessing'}
        )
    ]
    
    # 分配资源
    allocations = []
    for request in requests:
        allocation = await resource_manager.request_resource(request)
        if allocation:
            allocations.append(allocation)
            logger.info(f"资源分配成功: {allocation.allocation_id} "
                       f"({request.resource_type.value}: {request.amount})")
        else:
            logger.warning(f"资源分配失败: {request.request_id}")
    
    # 获取资源状态
    for allocation in allocations:
        status = resource_manager.get_resource_status(allocation.resource_id)
        logger.info(f"资源状态: {status}")
    
    # 获取消费者资源
    consumer_resources = resource_manager.get_consumer_allocations("ml_worker_1")
    logger.info(f"ML工作器资源: {len(consumer_resources)} 个分配")
    
    # 释放资源
    for allocation in allocations:
        success = await resource_manager.release_resource(allocation.allocation_id)
        if success:
            logger.info(f"资源已释放: {allocation.allocation_id}")
    
    # 获取资源管理器统计
    stats = resource_manager.get_resource_manager_stats()
    logger.info(f"资源管理器统计: {stats}")
    
    await resource_manager.stop_monitoring()


async def example_load_balancing():
    """负载均衡示例"""
    logger.info("=== 负载均衡示例 ===")
    
    # 创建负载均衡器
    load_balancer = LoadBalancer(name="api_load_balancer")
    await load_balancer.start()
    
    # 添加后端服务器
    servers = [
        BackendServer(
            server_id="api_server_1",
            host="192.168.1.10",
            port=8080,
            weight=2.0,
            max_connections=100,
            metadata={'region': 'us-east-1', 'version': 'v1.2.0'}
        ),
        BackendServer(
            server_id="api_server_2",
            host="192.168.1.11", 
            port=8080,
            weight=1.5,
            max_connections=80,
            metadata={'region': 'us-east-1', 'version': 'v1.2.0'}
        ),
        BackendServer(
            server_id="api_server_3",
            host="192.168.1.12",
            port=8080,
            weight=1.0,
            max_connections=120,
            metadata={'region': 'us-west-2', 'version': 'v1.1.5'}
        )
    ]
    
    for server in servers:
        load_balancer.add_server(server, group="api_servers")
        logger.info(f"添加服务器: {server.server_id} ({server.host}:{server.port})")
    
    # 测试不同的负载均衡策略
    strategies = [
        LoadBalanceStrategy.ROUND_ROBIN,
        LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN,
        LoadBalanceStrategy.LEAST_CONNECTIONS,
        LoadBalanceStrategy.LEAST_RESPONSE_TIME,
        LoadBalanceStrategy.RESOURCE_BASED
    ]
    
    for strategy in strategies:
        load_balancer.set_load_balance_strategy(strategy)
        logger.info(f"测试策略: {strategy.value}")
        
        # 模拟多个请求
        for i in range(5):
            request = LoadBalanceRequest(
                request_id=f"req_{strategy.value}_{i}",
                client_ip=f"192.168.1.{100 + i}",
                request_data={
                    'path': '/api/v1/users',
                    'method': 'GET',
                    'user_id': f'user_{i}'
                },
                priority=1
            )
            
            result = await load_balancer.select_server(request)
            if result.selected_server:
                logger.info(f"  请求 {i} -> {result.selected_server.server_id} "
                           f"(响应时间: {result.selection_time:.3f}s)")
                
                # 模拟释放连接
                await load_balancer.release_server_connection(result.selected_server.server_id)
    
    # 获取负载均衡器统计
    stats = load_balancer.get_load_balancer_stats()
    logger.info(f"负载均衡器统计: {stats}")
    
    await load_balancer.stop()


async def main():
    """主函数"""
    logger.info("开始优化调度执行机制示例演示")
    
    try:
        # 运行各种示例
        await example_basic_usage()
        await asyncio.sleep(1)
        
        await example_advanced_scheduling()
        await asyncio.sleep(1)
        
        await example_mcp_pipeline()
        await asyncio.sleep(1)
        
        await example_resource_management()
        await asyncio.sleep(1)
        
        await example_load_balancing()
        
        logger.info("所有示例演示完成")
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())