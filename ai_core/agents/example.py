#!/usr/bin/env python3
"""
MCP工具系统示例
演示如何使用MCP工具系统的各种功能
"""

import asyncio
import logging
import json
from typing import Dict, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_tool_registry():
    """演示工具注册功能"""
    print("\n" + "="*50)
    print("1. 工具注册功能演示")
    print("="*50)
    
    from mcp.core.tool_registry import global_registry
    from mcp.core.models import ToolCategory, ToolStatus
    
    # 获取工具统计
    stats = global_registry.get_tool_statistics()
    print(f"工具统计信息:")
    print(f"  总工具数: {stats['total_tools']}")
    print(f"  活跃工具数: {stats['active_tools']}")
    print(f"  分类统计: {stats['categories']}")
    print(f"  安全级别统计: {stats['security_levels']}")
    
    # 按分类列出工具
    categories = global_registry.list_tools_by_category()
    for category, tools in categories.items():
        if tools:
            print(f"\n{category.value} 分类工具:")
            for tool in tools[:3]:  # 只显示前3个
                print(f"  - {tool.name} ({tool.id})")
            if len(tools) > 3:
                print(f"  ... 还有 {len(tools) - 3} 个工具")


async def demo_tool_execution():
    """演示工具执行功能"""
    print("\n" + "="*50)
    print("2. 工具执行功能演示")
    print("="*50)
    
    from mcp.utils.tool_runner import tool_runner
    from mcp.core.tool_registry import global_registry
    from mcp.core.models import ToolStatus
    
    # 获取可用的工具
    tools = global_registry.list_tools(status=ToolStatus.ACTIVE)
    if not tools:
        print("没有可用的工具")
        return
    
    # 执行CPU监控工具
    cpu_tool = next((t for t in tools if t.id == "cpu_monitor"), None)
    if cpu_tool:
        print(f"\n执行工具: {cpu_tool.name}")
        result = await tool_runner.run_tool(
            tool_id="cpu_monitor",
            parameters={"duration": 5, "interval": 1.0},
            user_id="demo_user"
        )
        
        if result.get("success"):
            print(f"  执行成功!")
            print(f"  平均CPU使用率: {result.get('statistics', {}).get('average', 'N/A')}%")
            print(f"  监控样本数: {result.get('samples_count', 0)}")
        else:
            print(f"  执行失败: {result.get('error')}")
    
    # 执行内存监控工具
    memory_tool = next((t for t in tools if t.id == "memory_monitor"), None)
    if memory_tool:
        print(f"\n执行工具: {memory_tool.name}")
        result = await tool_runner.run_tool(
            tool_id="memory_monitor",
            parameters={"duration": 5},
            user_id="demo_user"
        )
        
        if result.get("success"):
            print(f"  执行成功!")
            print(f"  内存使用率: {result.get('statistics', {}).get('average_percentage', 'N/A')}%")
        else:
            print(f"  执行失败: {result.get('error')}")


async def demo_batch_execution():
    """演示批量执行功能"""
    print("\n" + "="*50)
    print("3. 批量执行功能演示")
    print("="*50)
    
    from mcp.utils.tool_runner import batch_processor
    from mcp.core.tool_registry import global_registry
    
    # 创建批量任务
    tasks = [
        {
            "id": "task1",
            "tool_id": "cpu_monitor",
            "parameters": {"duration": 3}
        },
        {
            "id": "task2", 
            "tool_id": "memory_monitor",
            "parameters": {"duration": 3}
        }
    ]
    
    print(f"执行 {len(tasks)} 个批量任务...")
    results = await batch_processor.process_batch(tasks)
    
    successful_tasks = sum(1 for r in results.values() if r.get("success"))
    print(f"批量执行完成:")
    print(f"  总任务数: {len(tasks)}")
    print(f"  成功任务: {successful_tasks}")
    print(f"  失败任务: {len(tasks) - successful_tasks}")
    
    # 显示结果摘要
    for task_id, result in results.items():
        status = "✓" if result.get("success") else "✗"
        print(f"  {status} {task_id}: {result.get('metric', 'unknown')}")


async def demo_skill_system():
    """演示技能系统"""
    print("\n" + "="*50)
    print("4. 技能系统演示")
    print("="*50)
    
    from mcp.skills import skill_manager, AtomicSkill, CompositeSkill, SkillStep, SkillExecutionContext
    
    # 创建原子技能
    print("创建原子技能...")
    cpu_skill = AtomicSkill("cpu_monitor")
    skill_manager.register_skill(cpu_skill)
    
    # 创建复合技能
    print("创建复合技能...")
    system_check = CompositeSkill(
        skill_id="system_check",
        steps=[
            SkillStep(
                id="cpu_check",
                name="CPU检查",
                tool_id="cpu_monitor",
                parameters={"duration": 3}
            ),
            SkillStep(
                id="memory_check",
                name="内存检查",
                tool_id="memory_monitor", 
                parameters={"duration": 3}
            )
        ]
    )
    skill_manager.register_skill(system_check)
    
    # 执行复合技能
    print("执行复合技能...")
    context = SkillExecutionContext(
        skill_id="system_check",
        parameters={},
        user_id="demo_user"
    )
    
    result = await skill_manager.execute_skill("system_check", {}, context)
    
    print(f"技能执行结果:")
    print(f"  成功: {result.success}")
    print(f"  执行步骤: {len(result.steps_executed)}")
    print(f"  执行时间: {result.execution_time:.2f}秒")
    
    if result.success:
        print("  步骤详情:")
        for step_id in result.steps_executed:
            step_result = result.step_results.get(step_id)
            if step_result:
                status = "✓" if step_result.success else "✗"
                print(f"    {status} {step_id}")
    
    # 获取技能统计
    stats = skill_manager.get_execution_statistics()
    print(f"\n技能统计:")
    print(f"  总技能数: {stats['total_skills']}")
    print(f"  总执行次数: {stats['total_executions']}")
    print(f"  成功率: {stats['success_rate']:.1f}%")


async def demo_security_system():
    """演示安全系统"""
    print("\n" + "="*50)
    print("5. 安全系统演示")
    print("="*50)
    
    from mcp.security.permission_manager import permission_manager, User, UserRole, PermissionType
    
    # 创建测试用户
    print("创建测试用户...")
    test_user = User(
        id="test_user",
        username="测试用户",
        role=UserRole.USER,
        permissions={PermissionType.READ, PermissionType.EXECUTE}
    )
    
    success = permission_manager.add_user(test_user)
    print(f"用户创建: {'成功' if success else '失败'}")
    
    # 测试权限检查
    print("\n权限检查测试:")
    
    # 检查CPU监控工具权限
    allowed = await permission_manager.check_permission(
        "cpu_monitor", {"duration": 10}, "test_user"
    )
    print(f"  CPU监控工具权限: {'允许' if allowed else '拒绝'}")
    
    # 检查端口扫描器权限（高危工具）
    allowed = await permission_manager.check_permission(
        "port_scanner", {"target": "127.0.0.1"}, "test_user"
    )
    print(f"  端口扫描器权限: {'允许' if allowed else '拒绝'}")
    
    # 获取安全统计
    stats = permission_manager.get_security_statistics()
    print(f"\n安全统计:")
    print(f"  总用户数: {stats['total_users']}")
    print(f"  活跃用户数: {stats['active_users']}")
    print(f"  安全规则数: {stats['total_rules']}")
    print(f"  24小时访问次数: {stats['recent_accesses_24h']}")
    print(f"  成功率: {stats['success_rate_24h']:.1f}%")


async def demo_mcp_server():
    """演示MCP服务器"""
    print("\n" + "="*50)
    print("6. MCP服务器演示")
    print("="*50)
    
    try:
        from mcp.core.mcp_server import mcp_server
        from mcp.config.config_manager import config_manager
        
        # 获取配置
        config = config_manager.get_config()
        print(f"服务器配置:")
        print(f"  主机: {config.server.host}")
        print(f"  端口: {config.server.port}")
        print(f"  调试模式: {config.server.debug}")
        print(f"  环境: {config.environment}")
        
        # 验证配置
        errors = config_manager.validate_config()
        if errors:
            print(f"配置验证错误:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("配置验证通过")
        
        print(f"\nMCP服务器功能:")
        print(f"  - RESTful API接口")
        print(f"  - 工具注册和管理")
        print(f"  - 权限控制和审计")
        print(f"  - 批量处理支持")
        print(f"  - 技能系统")
        print(f"  - 结果缓存")
        print(f"  - 统计和监控")
        
    except ImportError as e:
        print(f"服务器模块导入失败: {e}")


async def demo_configuration():
    """演示配置管理"""
    print("\n" + "="*50)
    print("7. 配置管理演示")
    print("="*50)
    
    from mcp.config.config_manager import config_manager
    
    # 获取当前配置
    config = config_manager.get_config()
    
    print("当前配置:")
    print(f"  服务器:")
    print(f"    主机: {config.server.host}")
    print(f"    端口: {config.server.port}")
    print(f"    调试: {config.server.debug}")
    
    print(f"  安全:")
    print(f"    启用: {config.security.enabled}")
    print(f"    策略: {config.security.default_policy}")
    print(f"    频率限制: {config.security.rate_limiting}")
    
    print(f"  缓存:")
    print(f"    启用: {config.cache.enabled}")
    print(f"    类型: {config.cache.type}")
    print(f"    TTL: {config.cache.ttl}秒")
    
    print(f"  日志:")
    print(f"    级别: {config.logging.level}")
    print(f"    控制台输出: {config.logging.console_output}")


async def demo_monitoring():
    """演示监控功能"""
    print("\n" + "="*50)
    print("8. 监控功能演示")
    print("="*50)
    
    from mcp.core.tool_registry import global_registry
    from mcp.skills import skill_manager
    from mcp.security.permission_manager import permission_manager
    from mcp.core.executor import ToolExecutor
    from mcp.core.result_handler import result_processor
    
    # 工具监控
    tool_stats = global_registry.get_tool_statistics()
    print("工具监控:")
    print(f"  活跃工具: {tool_stats['active_tools']}/{tool_stats['total_tools']}")
    
    # 技能监控
    skill_stats = skill_manager.get_execution_statistics()
    print("技能监控:")
    print(f"  注册技能: {skill_stats['total_skills']}")
    print(f"  执行次数: {skill_stats['total_executions']}")
    
    # 安全监控
    security_stats = permission_manager.get_security_statistics()
    print("安全监控:")
    print(f"  用户数: {security_stats['total_users']}")
    print(f"  访问日志: {len(permission_manager.access_logs)}")
    
    # 执行监控
    executor = ToolExecutor()
    exec_stats = executor.get_execution_statistics()
    print("执行监控:")
    print(f"  总执行次数: {exec_stats['total_executions']}")
    print(f"  成功率: {exec_stats['success_rate']:.1f}%")
    print(f"  平均执行时间: {exec_stats['average_execution_time']:.2f}秒")
    
    # 缓存监控
    cache_stats = result_processor.get_cache_statistics()
    print("缓存监控:")
    print(f"  缓存项数: {cache_stats['total_items']}")
    print(f"  有效项数: {cache_stats['valid_items']}")


async def setup_mcp_system():
    """设置MCP系统"""
    print("\n" + "="*50)
    print("MCP系统初始化")
    print("="*50)
    
    # 初始化工具注册
    from mcp.core.mcp_server import initialize_tools
    await initialize_tools()
    print("✓ 工具注册完成")
    
    # 初始化技能
    from mcp.core.mcp_server import initialize_skills
    await initialize_skills()
    print("✓ 技能初始化完成")
    
    # 加载配置
    from mcp.config.config_manager import config_manager
    config_manager.load_config()
    print("✓ 配置加载完成")


async def main():
    """主函数"""
    print("MCP工具系统演示")
    print("="*60)
    
    try:
        # 初始化系统
        await setup_mcp_system()
        
        # 各个功能演示
        await demo_tool_registry()
        await demo_tool_execution()
        await demo_batch_execution()
        await demo_skill_system()
        await demo_security_system()
        await demo_configuration()
        await demo_monitoring()
        await demo_mcp_server()
        
        print("\n" + "="*60)
        print("演示完成!")
        print("MCP工具系统已成功运行并展示了所有主要功能。")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        # 启动服务器模式
        from mcp.core.mcp_server import mcp_server
        import uvicorn
        
        config = {
            "host": "0.0.0.0",
            "port": 8000,
            "reload": True
        }
        
        print("启动MCP服务器...")
        uvicorn.run(mcp_server, **config)
    else:
        # 运行演示
        asyncio.run(main())