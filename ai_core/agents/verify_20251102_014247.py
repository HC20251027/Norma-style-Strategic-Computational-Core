#!/usr/bin/env python3
"""
任务分解规划系统快速验证脚本

验证系统各个组件是否正常工作
"""

import sys
import os

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

def test_imports():
    """测试模块导入"""
    print("1. 测试模块导入...")
    try:
        from task_planning import (
            Task, TaskStatus, TaskPriority, TaskDependency,
            TaskDecomposer, DependencyAnalyzer, TaskScheduler,
            StateTracker, RecoveryManager, TaskPlanner
        )
        print("   ✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"   ✗ 模块导入失败: {e}")
        return False

def test_models():
    """测试数据模型"""
    print("\n2. 测试数据模型...")
    try:
        from task_planning import Task, TaskStatus, TaskPriority
        
        # 创建任务
        task = Task(
            name="测试任务",
            description="这是一个测试任务",
            priority=TaskPriority.HIGH
        )
        
        # 测试状态更新
        task.update_status(TaskStatus.RUNNING)
        assert task.status == TaskStatus.RUNNING
        
        # 测试进度更新
        task.update_progress(0.5)
        assert task.progress == 0.5
        
        print("   ✓ 数据模型测试通过")
        return True
    except Exception as e:
        print(f"   ✗ 数据模型测试失败: {e}")
        return False

def test_utils():
    """测试工具函数"""
    print("\n3. 测试工具函数...")
    try:
        from task_planning.utils import (
            create_sample_tasks, 
            generate_task_summary,
            format_task_duration,
            analyze_task_complexity
        )
        
        # 创建示例任务
        tasks = create_sample_tasks()
        assert len(tasks) > 0
        
        # 生成摘要
        summary = generate_task_summary(tasks)
        assert "total_tasks" in summary
        
        # 格式化时长
        duration = format_task_duration(90)
        assert "小时" in duration
        
        # 分析复杂度
        if tasks:
            analysis = analyze_task_complexity(tasks[0])
            assert "complexity_score" in analysis
        
        print("   ✓ 工具函数测试通过")
        return True
    except Exception as e:
        print(f"   ✗ 工具函数测试失败: {e}")
        return False

def test_dependency_analyzer():
    """测试依赖分析器"""
    print("\n4. 测试依赖分析器...")
    try:
        from task_planning import Task, TaskExecutionPlan, TaskDependency
        from task_planning.dependency_analyzer import DependencyAnalyzer
        
        analyzer = DependencyAnalyzer()
        
        # 创建测试任务
        task1 = Task(name="任务1")
        task2 = Task(name="任务2")
        
        plan = TaskExecutionPlan()
        plan.add_task(task1)
        plan.add_task(task2)
        
        # 添加依赖
        dep = TaskDependency(task1.id, task2.id)
        plan.add_dependency(dep)
        
        # 分析依赖
        result = analyzer.analyze_dependencies(plan)
        assert "topological_order" in result
        assert "execution_phases" in result
        
        print("   ✓ 依赖分析器测试通过")
        return True
    except Exception as e:
        print(f"   ✗ 依赖分析器测试失败: {e}")
        return False

def test_state_tracker():
    """测试状态跟踪器"""
    print("\n5. 测试状态跟踪器...")
    try:
        from task_planning import Task, TaskStatus
        from task_planning.state_tracker import StateTracker
        
        tracker = StateTracker()
        
        # 注册任务
        task = Task(name="测试任务")
        task_id = tracker.register_task(task)
        assert task_id == task.id
        
        # 更新状态
        success = tracker.update_task_status(task.id, TaskStatus.RUNNING)
        assert success
        assert tracker.tasks[task.id].status == TaskStatus.RUNNING
        
        # 获取事件
        events = tracker.get_task_events(task.id)
        assert len(events) > 0
        
        print("   ✓ 状态跟踪器测试通过")
        return True
    except Exception as e:
        print(f"   ✗ 状态跟踪器测试失败: {e}")
        return False

def test_recovery_manager():
    """测试恢复管理器"""
    print("\n6. 测试恢复管理器...")
    try:
        from task_planning import Task
        from task_planning.state_tracker import StateTracker
        from task_planning.recovery_manager import RecoveryManager, FailureType
        
        tracker = StateTracker()
        recovery_manager = RecoveryManager(tracker)
        
        # 测试失败类型分析
        task = Task(name="测试任务")
        error = ConnectionError("连接失败")
        failure_type = recovery_manager._analyze_failure_type(task, error)
        
        assert failure_type == FailureType.TEMPORARY
        
        print("   ✓ 恢复管理器测试通过")
        return True
    except Exception as e:
        print(f"   ✗ 恢复管理器测试失败: {e}")
        return False

def test_task_decomposer():
    """测试任务分解器"""
    print("\n7. 测试任务分解器...")
    try:
        from task_planning import Task
        from task_planning.task_decomposer import TaskDecomposer
        
        class MockLLM:
            async def generate_response(self, prompt):
                return '{"subtasks": [], "reasoning": "测试"}'
        
        decomposer = TaskDecomposer(MockLLM())
        
        # 测试简单任务检测
        simple_task = Task(name="简单任务", description="简单描述")
        is_simple = decomposer._is_simple_task(simple_task)
        assert isinstance(is_simple, bool)
        
        # 测试模板
        templates = decomposer.decomposition_templates
        assert "generic" in templates
        
        print("   ✓ 任务分解器测试通过")
        return True
    except Exception as e:
        print(f"   ✗ 任务分解器测试失败: {e}")
        return False

def test_integration():
    """测试集成功能"""
    print("\n8. 测试集成功能...")
    try:
        from task_planning import TaskPlanner, TaskPriority
        
        class MockLLM:
            async def generate_response(self, prompt):
                return '{"subtasks": [], "reasoning": "测试"}'
        
        planner = TaskPlanner(MockLLM())
        
        # 测试规划器创建
        assert planner is not None
        assert planner.task_decomposer is not None
        assert planner.dependency_analyzer is not None
        assert planner.scheduler is not None
        assert planner.state_tracker is not None
        
        print("   ✓ 集成测试通过")
        return True
    except Exception as e:
        print(f"   ✗ 集成测试失败: {e}")
        return False

def main():
    """主函数"""
    print("任务分解规划系统验证")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_models,
        test_utils,
        test_dependency_analyzer,
        test_state_tracker,
        test_recovery_manager,
        test_task_decomposer,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"验证结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有验证通过！系统已准备就绪。")
        return True
    else:
        print("❌ 部分验证失败，请检查错误信息。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)