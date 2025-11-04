"""
Toolcall抽象层使用示例

演示如何使用Toolcall抽象层实现智能工具调用
"""

import asyncio
import logging
from typing import Dict, Any

from . import (
    ToolcallManager,
    ToolcallRegistry,
    ToolcallRequest,
    OptimizationConfig,
    process_toolcall_request,
    get_toolcall_status,
    get_toolcall_statistics
)

from .llm_interface import MockLLMInterface
from ..mcp.core.tool_registry import global_registry as mcp_tool_registry


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_usage_example():
    """基本使用示例"""
    print("=== Toolcall抽象层基本使用示例 ===")
    
    # 1. 创建LLM接口（这里使用模拟接口）
    llm_interface = MockLLMInterface()
    
    # 2. 创建优化配置
    config = OptimizationConfig(
        enable_caching=True,
        cache_ttl=3600,
        enable_parallel_execution=True,
        max_parallel_tools=3,
        enable_retry=True,
        max_retries=3
    )
    
    # 3. 注册ToolcallManager
    success = register_toolcall_manager(
        manager_id="default",
        llm_interface=llm_interface,
        tool_registry=mcp_tool_registry,
        config=config
    )
    
    if not success:
        print("注册ToolcallManager失败")
        return
    
    # 4. 处理工具调用请求
    response = await process_toolcall_request(
        manager_id="default",
        user_query="监控系统的CPU使用率",
        session_id="demo_session_001",
        user_id="demo_user",
        context={"environment": "production"}
    )
    
    if response:
        print(f"请求处理结果:")
        print(f"  状态: {response.status}")
        print(f"  成功: {response.success}")
        print(f"  耗时: {response.execution_time:.2f}s")
        if response.data:
            print(f"  数据: {response.data}")
        if response.error:
            print(f"  错误: {response.error}")
    else:
        print("请求处理失败")


async def advanced_usage_example():
    """高级使用示例"""
    print("\n=== Toolcall抽象层高级使用示例 ===")
    
    # 1. 获取注册中心
    registry = ToolcallRegistry()
    
    # 2. 获取管理器
    manager = registry.get_manager("default")
    if not manager:
        print("管理器不存在")
        return
    
    # 3. 创建复杂请求
    request = ToolcallRequest(
        user_query="分析网络流量并生成安全报告",
        session_id="advanced_session_001",
        user_id="security_analyst",
        context={
            "environment": "production",
            "analysis_type": "security",
            "time_range": "last_24_hours"
        },
        priority=2,  # 高优先级
        timeout=60,
        max_tools=5
    )
    
    # 4. 处理请求
    response = await manager.process_request(request)
    
    if response:
        print(f"高级请求处理结果:")
        print(f"  状态: {response.status}")
        print(f"  成功: {response.success}")
        print(f"  工具结果数量: {len(response.tool_results)}")
        
        # 显示工具执行详情
        for result in response.tool_results:
            print(f"    工具 {result['tool_id']}: {'成功' if result['success'] else '失败'}")
            if not result['success']:
                print(f"      错误: {result.get('error', '未知错误')}")
    else:
        print("高级请求处理失败")


async def batch_processing_example():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    registry = ToolcallRegistry()
    manager = registry.get_manager("default")
    if not manager:
        print("管理器不存在")
        return
    
    # 创建批量请求
    requests = [
        ToolcallRequest(
            user_query="检查系统状态",
            session_id="batch_session_001",
            user_id="system_admin"
        ),
        ToolcallRequest(
            user_query="分析性能指标",
            session_id="batch_session_002", 
            user_id="performance_analyst"
        ),
        ToolcallRequest(
            user_query="生成使用报告",
            session_id="batch_session_003",
            user_id="report_generator"
        )
    ]
    
    # 批量处理
    responses = await manager.batch_process_requests(requests)
    
    print(f"批量处理完成，处理了 {len(responses)} 个请求:")
    for i, response in enumerate(responses):
        if response:
            print(f"  请求 {i+1}: {response.status} - {'成功' if response.success else '失败'}")
        else:
            print(f"  请求 {i+1}: 处理失败")


async def monitoring_example():
    """监控示例"""
    print("\n=== 系统监控示例 ===")
    
    registry = ToolcallRegistry()
    
    # 获取系统统计信息
    stats = await registry.get_system_statistics("default")
    if stats:
        print("系统统计信息:")
        print(f"  活跃请求数: {stats.get('active_requests', 0)}")
        print(f"  上下文数量: {stats.get('context_manager', {}).get('total_contexts', 0)}")
        print(f"  缓存命中率: {stats.get('performance_optimizer', {}).get('cache_hit_rate', 0):.2%}")
        print(f"  错误率: {stats.get('error_handler', {}).get('recent_error_rate', 0):.2%}")
    
    # 健康检查
    health = await registry.health_check("default")
    if health:
        print(f"\n健康检查状态: {health.get('status', 'unknown')}")
        components = health.get('components', {})
        for component, status in components.items():
            print(f"  {component}: {status.get('status', 'unknown')}")


async def configuration_optimization_example():
    """配置优化示例"""
    print("\n=== 配置优化示例 ===")
    
    registry = ToolcallRegistry()
    manager = registry.get_manager("default")
    if not manager:
        print("管理器不存在")
        return
    
    # 优化配置
    optimized_config = await manager.optimize_configuration()
    
    print("配置优化完成:")
    print(f"  缓存启用: {optimized_config.enable_caching}")
    print(f"  并行执行: {optimized_config.enable_parallel_execution}")
    print(f"  最大并行工具数: {optimized_config.max_parallel_tools}")
    print(f"  最大重试次数: {optimized_config.max_retries}")
    print(f"  基础超时时间: {optimized_config.base_timeout}s")


async def error_handling_example():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    registry = ToolcallRegistry()
    
    # 创建一个会失败的请求
    response = await process_toolcall_request(
        manager_id="default",
        user_query="执行一个不存在的操作",
        session_id="error_test_session",
        user_id="test_user"
    )
    
    if response:
        print(f"错误处理结果:")
        print(f"  状态: {response.status}")
        print(f"  成功: {response.success}")
        if response.error:
            print(f"  错误信息: {response.error}")
        
        # 检查是否有澄清信息
        metadata = response.metadata
        if "clarification_questions" in metadata:
            print("  需要澄清的问题:")
            for question in metadata["clarification_questions"]:
                print(f"    - {question}")


async def main():
    """主函数"""
    print("Toolcall抽象层演示程序")
    print("=" * 50)
    
    try:
        # 运行各种示例
        await basic_usage_example()
        await advanced_usage_example()
        await batch_processing_example()
        await monitoring_example()
        await configuration_optimization_example()
        await error_handling_example()
        
    except Exception as e:
        print(f"演示程序出错: {e}")
    
    finally:
        # 清理资源
        registry = ToolcallRegistry()
        await registry.shutdown_all()
        print("\n演示程序结束，资源已清理")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())