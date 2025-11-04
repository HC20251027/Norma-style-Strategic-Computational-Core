"""
任务规划工具函数

提供常用的工具函数和辅助功能
"""

import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import asyncio
import uuid

from .models import Task, TaskStatus, TaskPriority, TaskExecutionPlan, TaskDependency
from .task_planner import TaskPlanner
from ..llm.interfaces.llm_interface import LLMInterface


def create_sample_tasks() -> List[Task]:
    """创建示例任务"""
    tasks = []
    
    # 示例任务1：数据分析
    task1 = Task(
        name="数据分析项目",
        description="对销售数据进行深度分析，包括趋势分析、异常检测和预测建模",
        task_type="analysis",
        priority=TaskPriority.HIGH,
        estimated_duration=120,
        parameters={
            "data_source": "sales_database",
            "analysis_types": ["trend", "anomaly", "prediction"],
            "output_format": "report"
        },
        tags=["数据分析", "销售", "预测"]
    )
    tasks.append(task1)
    
    # 示例任务2：软件开发
    task2 = Task(
        name="API开发项目",
        description="开发用户管理API，包括用户注册、登录、权限管理功能",
        task_type="development",
        priority=TaskPriority.CRITICAL,
        estimated_duration=240,
        parameters={
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "authentication": "JWT",
            "endpoints": ["register", "login", "profile", "permissions"]
        },
        tags=["后端开发", "API", "用户管理"]
    )
    tasks.append(task2)
    
    # 示例任务3：研究任务
    task3 = Task(
        name="市场调研报告",
        description="对AI行业市场趋势进行调研，撰写详细的分析报告",
        task_type="research",
        priority=TaskPriority.NORMAL,
        estimated_duration=180,
        parameters={
            "research_scope": "AI_industry",
            "report_length": "comprehensive",
            "target_audience": "executives",
            "deadline": "2024-12-31"
        },
        tags=["市场调研", "AI", "行业分析"]
    )
    tasks.append(task3)
    
    return tasks


def create_sample_execution_plan() -> TaskExecutionPlan:
    """创建示例执行计划"""
    plan = TaskExecutionPlan(
        name="示例项目执行计划",
        description="展示任务规划系统的完整功能",
        max_parallel_tasks=3,
        execution_mode="hybrid"
    )
    
    # 添加示例任务
    tasks = create_sample_tasks()
    for task in tasks:
        plan.add_task(task)
    
    # 添加依赖关系
    # 数据分析任务依赖于API开发的数据
    dep1 = TaskDependency(
        source_task_id=tasks[1].id,  # API开发
        target_task_id=tasks[0].id,  # 数据分析
        dependency_type="finish_to_start"
    )
    plan.add_dependency(dep1)
    
    # 市场调研可以并行进行
    # 没有依赖关系
    
    return plan


def format_task_duration(minutes: int) -> str:
    """格式化任务时长"""
    if minutes < 60:
        return f"{minutes}分钟"
    else:
        hours = minutes // 60
        remaining_minutes = minutes % 60
        if remaining_minutes == 0:
            return f"{hours}小时"
        else:
            return f"{hours}小时{remaining_minutes}分钟"


def format_task_priority(priority: TaskPriority) -> str:
    """格式化任务优先级"""
    priority_map = {
        TaskPriority.LOW: "低",
        TaskPriority.NORMAL: "普通",
        TaskPriority.HIGH: "高",
        TaskPriority.CRITICAL: "关键"
    }
    return priority_map.get(priority, "未知")


def format_task_status(status: TaskStatus) -> str:
    """格式化任务状态"""
    status_map = {
        TaskStatus.PENDING: "等待中",
        TaskStatus.PLANNING: "规划中",
        TaskStatus.READY: "准备就绪",
        TaskStatus.RUNNING: "执行中",
        TaskStatus.COMPLETED: "已完成",
        TaskStatus.FAILED: "执行失败",
        TaskStatus.CANCELLED: "已取消",
        TaskStatus.RETRYING: "重试中",
        TaskStatus.SKIPPED: "已跳过"
    }
    return status_map.get(status, "未知")


def calculate_estimated_completion(
    tasks: List[Task],
    start_time: Optional[datetime] = None
) -> datetime:
    """计算预计完成时间"""
    if not tasks:
        return datetime.now()
    
    if start_time is None:
        start_time = datetime.now()
    
    # 计算总耗时（简化计算，实际应该考虑并行执行）
    total_minutes = sum(task.estimated_duration or 0 for task in tasks)
    
    return start_time + timedelta(minutes=total_minutes)


def analyze_task_complexity(task: Task) -> Dict[str, Any]:
    """分析任务复杂度"""
    complexity_score = 0
    factors = []
    
    # 基于描述长度
    desc_length = len(task.description)
    if desc_length > 500:
        complexity_score += 2
        factors.append("详细描述")
    elif desc_length > 200:
        complexity_score += 1
        factors.append("中等描述")
    
    # 基于参数数量
    param_count = len(task.parameters)
    if param_count > 5:
        complexity_score += 2
        factors.append("复杂参数")
    elif param_count > 2:
        complexity_score += 1
        factors.append("中等参数")
    
    # 基于预计时长
    if task.estimated_duration:
        if task.estimated_duration > 240:  # 4小时
            complexity_score += 2
            factors.append("长时间任务")
        elif task.estimated_duration > 60:  # 1小时
            complexity_score += 1
            factors.append("中等时长")
    
    # 基于任务类型
    complex_types = ["development", "analysis", "research"]
    if task.task_type in complex_types:
        complexity_score += 1
        factors.append(f"{task.task_type}类型")
    
    # 确定复杂度等级
    if complexity_score >= 5:
        complexity_level = "高"
    elif complexity_score >= 3:
        complexity_level = "中"
    else:
        complexity_level = "低"
    
    return {
        "complexity_score": complexity_score,
        "complexity_level": complexity_level,
        "factors": factors,
        "recommendation": get_complexity_recommendation(complexity_score)
    }


def get_complexity_recommendation(score: int) -> str:
    """获取复杂度建议"""
    if score >= 5:
        return "建议深度分解任务，增加中间检查点，考虑并行执行"
    elif score >= 3:
        return "建议适度分解，增加进度监控"
    else:
        return "任务相对简单，可以直接执行"


def generate_task_summary(tasks: List[Task]) -> Dict[str, Any]:
    """生成任务摘要"""
    if not tasks:
        return {"total_tasks": 0}
    
    # 统计信息
    total_tasks = len(tasks)
    status_counts = {}
    priority_counts = {}
    total_duration = 0
    
    for task in tasks:
        # 状态统计
        status = task.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
        
        # 优先级统计
        priority = task.priority.value
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        # 时长统计
        if task.estimated_duration:
            total_duration += task.estimated_duration
    
    # 计算平均时长
    tasks_with_duration = [t for t in tasks if t.estimated_duration]
    avg_duration = total_duration / len(tasks_with_duration) if tasks_with_duration else 0
    
    return {
        "total_tasks": total_tasks,
        "status_distribution": status_counts,
        "priority_distribution": priority_counts,
        "total_estimated_duration": total_duration,
        "average_duration": avg_duration,
        "formatted_duration": format_task_duration(int(avg_duration)),
        "completion_rate": status_counts.get("completed", 0) / total_tasks,
        "failure_rate": status_counts.get("failed", 0) / total_tasks
    }


def validate_task_dependencies(tasks: List[Task], dependencies: List[TaskDependency]) -> Tuple[bool, List[str]]:
    """验证任务依赖关系"""
    errors = []
    
    # 获取所有任务ID
    task_ids = {task.id for task in tasks}
    
    # 检查依赖的任务是否存在
    for dep in dependencies:
        if dep.source_task_id not in task_ids:
            errors.append(f"依赖的源任务不存在: {dep.source_task_id}")
        
        if dep.target_task_id not in task_ids:
            errors.append(f"依赖的目标任务不存在: {dep.target_task_id}")
    
    # 检查自依赖
    for dep in dependencies:
        if dep.source_task_id == dep.target_task_id:
            errors.append(f"任务存在自依赖: {dep.source_task_id}")
    
    # 检查循环依赖
    try:
        from .dependency_analyzer import DependencyAnalyzer
        analyzer = DependencyAnalyzer()
        
        # 创建临时计划进行验证
        temp_plan = TaskExecutionPlan()
        for task in tasks:
            temp_plan.add_task(task)
        for dep in dependencies:
            temp_plan.add_dependency(dep)
        
        is_valid, validation_errors = analyzer.validate_execution_plan(temp_plan)
        if not is_valid:
            errors.extend(validation_errors)
    
    except Exception as e:
        errors.append(f"依赖关系验证失败: {str(e)}")
    
    return len(errors) == 0, errors


def export_plan_to_dict(plan: TaskExecutionPlan) -> Dict[str, Any]:
    """将执行计划导出为字典"""
    return {
        "id": plan.id,
        "name": plan.name,
        "description": plan.description,
        "created_at": plan.created_at.isoformat(),
        "status": plan.status.value,
        "progress": plan.progress,
        "max_parallel_tasks": plan.max_parallel_tasks,
        "execution_mode": plan.execution_mode,
        "tasks": [task.to_dict() for task in plan.tasks],
        "dependencies": [
            {
                "id": dep.id,
                "source_task_id": dep.source_task_id,
                "target_task_id": dep.target_task_id,
                "dependency_type": dep.dependency_type,
                "delay": dep.delay,
                "condition": dep.condition
            }
            for dep in plan.dependencies
        ],
        "summary": generate_task_summary(plan.tasks)
    }


def import_plan_from_dict(data: Dict[str, Any]) -> TaskExecutionPlan:
    """从字典导入执行计划"""
    plan = TaskExecutionPlan(
        id=data["id"],
        name=data["name"],
        description=data["description"],
        created_at=datetime.fromisoformat(data["created_at"]),
        status=TaskStatus(data["status"]),
        progress=data["progress"],
        max_parallel_tasks=data["max_parallel_tasks"],
        execution_mode=data["execution_mode"]
    )
    
    # 导入任务
    for task_data in data["tasks"]:
        task = Task.from_dict(task_data)
        plan.add_task(task)
    
    # 导入依赖关系
    for dep_data in data["dependencies"]:
        dependency = TaskDependency(
            source_task_id=dep_data["source_task_id"],
            target_task_id=dep_data["target_task_id"],
            dependency_type=dep_data["dependency_type"],
            delay=dep_data["delay"],
            condition=dep_data["condition"]
        )
        # 设置ID
        dependency.id = dep_data["id"]
        plan.add_dependency(dependency)
    
    return plan


async def create_quick_demo_plan(llm_interface: LLMInterface) -> TaskExecutionPlan:
    """创建快速演示计划"""
    planner = TaskPlanner(llm_interface)
    
    # 创建一个简单的演示任务
    plan = await planner.create_plan(
        task_description="分析公司销售数据并生成月度报告",
        task_type="analysis",
        parameters={
            "data_source": "sales_db",
            "report_type": "monthly",
            "include_charts": True
        },
        priority=TaskPriority.NORMAL,
        max_depth=2,
        min_task_duration=15,
        max_subtasks=5
    )
    
    return plan


def get_system_status(planner: TaskPlanner) -> Dict[str, Any]:
    """获取系统状态"""
    return {
        "timestamp": datetime.now().isoformat(),
        "planner_status": {
            "active_plans": len(planner.active_plans),
            "completed_plans": len(planner.completed_plans),
            "failed_plans": len(planner.failed_plans),
            "recovery_enabled": planner.enable_recovery,
            "monitoring_enabled": planner.enable_monitoring
        },
        "performance_metrics": planner.get_performance_metrics(),
        "system_health": "healthy"  # 可以扩展为更复杂的健康检查
    }


def create_performance_report(planner: TaskPlanner) -> str:
    """创建性能报告"""
    metrics = planner.get_performance_metrics()
    
    report = f"""
# 任务规划系统性能报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 总体统计
- 总计划数: {metrics['task_planner']['total_plans_created']}
- 成功率: {metrics['task_planner']['success_rate']:.2%}
- 当前活跃计划: {metrics['task_planner']['active_plans_count']}

## 调度配置
- 最大并行任务: {metrics['scheduler']['max_parallel_tasks']}
- 调度策略: {metrics['scheduler']['strategy']}

## 性能指标
"""
    
    if metrics.get('state_tracker'):
        st_metrics = metrics['state_tracker']
        report += f"""
- 平均任务时长: {st_metrics.get('average_task_duration', 0):.1f} 分钟
- 成功率: {st_metrics.get('success_rate', 0):.2%}
- 吞吐量: {st_metrics.get('throughput', 0):.2f} 任务/小时
"""
    
    if metrics.get('recovery_manager') and metrics['recovery_manager']:
        rm_metrics = metrics['recovery_manager']
        report += f"""
## 恢复统计
- 恢复尝试次数: {rm_metrics['recovery_stats']['total_recoveries']}
- 成功恢复次数: {rm_metrics['recovery_stats']['successful_recoveries']}
- 失败恢复次数: {rm_metrics['recovery_stats']['failed_recoveries']}
"""
    
    return report