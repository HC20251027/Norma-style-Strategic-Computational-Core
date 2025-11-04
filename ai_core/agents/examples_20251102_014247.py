"""
任务分解规划系统使用示例

展示如何使用任务规划器的各个功能
"""

import asyncio
import json
from datetime import datetime

# 导入任务规划系统组件
from task_planning import (
    TaskPlanner, Task, TaskPriority, TaskStatus,
    create_sample_tasks, create_sample_execution_plan,
    generate_task_summary, format_task_duration,
    get_system_status, create_performance_report
)


class MockLLMInterface:
    """模拟LLM接口用于演示"""
    
    async def generate_response(self, prompt: str) -> str:
        """模拟LLM响应"""
        # 模拟任务分解的响应
        if "分解" in prompt or "subtasks" in prompt:
            return '''
{
    "subtasks": [
        {
            "name": "数据收集",
            "description": "从各个数据源收集销售数据",
            "priority": "high",
            "estimated_duration": 30,
            "task_type": "data_collection",
            "parameters": {"sources": ["database", "api", "files"]}
        },
        {
            "name": "数据清洗",
            "description": "清洗和预处理收集的数据",
            "priority": "high",
            "estimated_duration": 45,
            "task_type": "data_processing",
            "parameters": {"cleaning_rules": ["remove_nulls", "normalize"]}
        },
        {
            "name": "数据分析",
            "description": "进行趋势分析和异常检测",
            "priority": "normal",
            "estimated_duration": 60,
            "task_type": "analysis",
            "parameters": {"analysis_types": ["trend", "anomaly"]}
        },
        {
            "name": "报告生成",
            "description": "生成分析报告和可视化图表",
            "priority": "normal",
            "estimated_duration": 30,
            "task_type": "reporting",
            "parameters": {"format": "pdf", "include_charts": true}
        }
    ],
    "reasoning": "将复杂的数据分析任务分解为4个主要阶段，每个阶段都有明确的输入输出和预计时长"
}
'''
        else:
            return "模拟LLM响应"


async def demo_basic_usage():
    """演示基本使用方法"""
    print("=== 基本使用演示 ===")
    
    # 创建LLM接口和任务规划器
    llm = MockLLMInterface()
    planner = TaskPlanner(llm_interface=llm)
    
    # 创建并执行任务计划
    result = await planner.create_and_execute_plan(
        task_description="分析公司销售数据并生成月度报告",
        task_type="analysis",
        parameters={
            "data_source": "sales_database",
            "report_type": "monthly",
            "include_visualizations": True
        },
        priority=TaskPriority.HIGH,
        max_depth=2,
        min_task_duration=15
    )
    
    print(f"执行结果: {result}")
    
    # 获取计划状态
    if 'plan_id' in result:
        status = planner.get_plan_status(result['plan_id'])
        print(f"计划状态: {json.dumps(status, ensure_ascii=False, indent=2, default=str)}")
    
    # 清理资源
    planner.cleanup()


async def demo_advanced_features():
    """演示高级功能"""
    print("\n=== 高级功能演示 ===")
    
    llm = MockLLMInterface()
    planner = TaskPlanner(
        llm_interface=llm,
        max_parallel_tasks=3,
        enable_recovery=True,
        enable_monitoring=True
    )
    
    # 定义回调函数
    async def on_plan_start(plan):
        print(f"计划开始执行: {plan.name}")
    
    async def on_task_complete(task):
        print(f"任务完成: {task.name}")
    
    async def on_plan_progress(plan_or_task, milestone, progress):
        if isinstance(plan_or_task, str):
            print(f"里程碑到达: {milestone}, 进度: {progress:.1%}")
        else:
            print(f"任务进度更新: {plan_or_task.name}, 进度: {progress:.1%}")
    
    async def on_recovery_start(recovery_plan):
        print(f"启动恢复流程: 任务 {recovery_plan.task_id}")
    
    callbacks = {
        "on_plan_start": on_plan_start,
        "on_task_complete": on_task_complete,
        "on_plan_progress": on_plan_progress,
        "on_recovery_start": on_recovery_start
    }
    
    # 创建复杂任务计划
    plan = await planner.create_plan(
        task_description="开发一个完整的电商推荐系统",
        task_type="development",
        parameters={
            "system_type": "recommendation",
            "technologies": ["python", "tensorflow", "redis"],
            "features": ["collaborative_filtering", "content_based"],
            "deployment": "cloud"
        },
        priority=TaskPriority.CRITICAL,
        max_depth=3,
        min_task_duration=30
    )
    
    # 添加自定义依赖关系
    main_task = plan.tasks[0]
    subtasks = [t for t in plan.tasks if t.metadata.get("parent_task_id") == main_task.id]
    
    if len(subtasks) >= 2:
        planner.add_task_dependency(
            plan=plan,
            source_task_id=subtasks[0].id,
            target_task_id=subtasks[1].id,
            dependency_type="finish_to_start"
        )
    
    # 执行计划
    result = await planner.execute_plan(plan, callbacks)
    print(f"复杂计划执行结果: {result}")
    
    # 获取系统状态
    system_status = get_system_status(planner)
    print(f"系统状态: {json.dumps(system_status, ensure_ascii=False, indent=2, default=str)}")
    
    # 生成性能报告
    report = create_performance_report(planner)
    print(report)
    
    planner.cleanup()


async def demo_task_management():
    """演示任务管理功能"""
    print("\n=== 任务管理演示 ===")
    
    llm = MockLLMInterface()
    planner = TaskPlanner(llm_interface=llm)
    
    # 创建示例任务
    sample_tasks = create_sample_tasks()
    print(f"创建了 {len(sample_tasks)} 个示例任务")
    
    # 生成任务摘要
    summary = generate_task_summary(sample_tasks)
    print(f"任务摘要: {json.dumps(summary, ensure_ascii=False, indent=2, default=str)}")
    
    # 创建示例执行计划
    plan = create_sample_execution_plan()
    print(f"创建执行计划: {plan.name}")
    print(f"包含 {len(plan.tasks)} 个任务, {len(plan.dependencies)} 个依赖关系")
    
    # 导出计划
    exported_plan = plan.to_dict() if hasattr(plan, 'to_dict') else {"plan": "数据格式转换"}
    print(f"计划导出成功，包含 {len(exported_plan.get('tasks', []))} 个任务数据")
    
    planner.cleanup()


async def demo_error_handling():
    """演示错误处理和恢复机制"""
    print("\n=== 错误处理演示 ===")
    
    llm = MockLLMInterface()
    planner = TaskPlanner(
        llm_interface=llm,
        enable_recovery=True,
        enable_monitoring=True
    )
    
    # 创建容易失败的任务用于演示恢复机制
    result = await planner.create_and_execute_plan(
        task_description="测试恢复机制的任务（故意设计为可能失败）",
        task_type="testing",
        parameters={
            "failure_probability": 0.7,  # 70% 失败概率用于演示
            "enable_recovery": True
        },
        priority=TaskPriority.NORMAL
    )
    
    print(f"错误处理演示结果: {result}")
    
    # 检查恢复统计
    if planner.recovery_manager:
        recovery_stats = planner.recovery_manager.get_recovery_statistics()
        print(f"恢复统计: {json.dumps(recovery_stats, ensure_ascii=False, indent=2, default=str)}")
    
    planner.cleanup()


async def demo_performance_monitoring():
    """演示性能监控功能"""
    print("\n=== 性能监控演示 ===")
    
    llm = MockLLMInterface()
    planner = TaskPlanner(
        llm_interface=llm,
        max_parallel_tasks=2,
        enable_monitoring=True
    )
    
    # 创建多个计划进行性能测试
    plans_created = 0
    for i in range(3):
        try:
            plan = await planner.create_plan(
                task_description=f"性能测试任务 {i+1}",
                task_type="testing",
                parameters={"test_id": i+1},
                max_depth=1  # 简化分解以加快演示
            )
            plans_created += 1
        except Exception as e:
            print(f"创建计划 {i+1} 失败: {e}")
    
    print(f"成功创建 {plans_created} 个计划")
    
    # 获取性能指标
    metrics = planner.get_performance_metrics()
    print(f"性能指标: {json.dumps(metrics, ensure_ascii=False, indent=2, default=str)}")
    
    # 获取所有计划状态
    all_status = planner.get_all_plans_status()
    print(f"所有计划状态: {json.dumps(all_status['summary'], ensure_ascii=False, indent=2, default=str)}")
    
    planner.cleanup()


async def main():
    """主演示函数"""
    print("任务分解规划系统演示")
    print("=" * 50)
    
    try:
        # 演示基本功能
        await demo_basic_usage()
        
        # 演示高级功能
        await demo_advanced_features()
        
        # 演示任务管理
        await demo_task_management()
        
        # 演示错误处理
        await demo_error_handling()
        
        # 演示性能监控
        await demo_performance_monitoring()
        
        print("\n" + "=" * 50)
        print("所有演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())