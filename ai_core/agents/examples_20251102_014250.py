#!/usr/bin/env python3
"""
诺玛Agent监控和优化系统使用示例
演示如何使用各个监控组件
"""

import asyncio
import time
import json
from pathlib import Path

# 导入监控组件
from .monitoring_manager import MonitoringManager, create_monitoring_manager
from .dashboard.monitoring_dashboard import MonitoringDashboard, MetricType
from .metrics.performance_collector import MetricsCollector, PerformanceMetric
from .alerts.alert_system import AlertSystem, Alert, AlertSeverity, AlertStatus
from .tuning.auto_tuner import AutoTuner, TuningAction, ResourceType
from .analytics.user_analytics import UserAnalytics, UserAction, UserSegment
from .health.health_monitor import HealthMonitor, HealthStatus, ComponentType
from . import quick_start, get_monitoring_status, get_dashboard_data, get_active_alerts, get_health_status

async def basic_monitoring_example():
    """基本监控示例"""
    print("=== 基本监控示例 ===")
    
    # 使用快速启动函数
    manager = quick_start()
    
    try:
        # 启动监控
        await manager.start_monitoring()
        print("监控已启动")
        
        # 运行一段时间观察数据
        for i in range(10):
            await asyncio.sleep(10)
            
            # 获取状态
            status = manager.get_monitoring_status()
            print(f"第 {i+1} 次检查 - 运行时间: {status['uptime']:.0f}秒")
            
            # 获取仪表板数据
            dashboard_data = manager.get_dashboard_data()
            if 'system_status' in dashboard_data:
                sys_status = dashboard_data['system_status']
                print(f"  CPU: {sys_status.get('cpu_percent', 0):.1f}%")
                print(f"  内存: {sys_status.get('memory_percent', 0):.1f}%")
                print(f"  磁盘: {sys_status.get('disk_percent', 0):.1f}%")
            
            # 获取活跃告警
            alerts = manager.get_active_alerts()
            if alerts:
                print(f"  活跃告警: {len(alerts)} 个")
                for alert in alerts[:3]:  # 显示前3个
                    print(f"    - {alert.rule_name}: {alert.message}")
            
            # 获取健康状态
            health = manager.get_health_status()
            healthy_count = sum(1 for status in health.values() if status == HealthStatus.HEALTHY)
            print(f"  健康组件: {healthy_count}/{len(health)}")
    
    finally:
        await manager.stop_monitoring()
        print("监控已停止")

async def user_behavior_tracking_example():
    """用户行为跟踪示例"""
    print("\n=== 用户行为跟踪示例 ===")
    
    # 创建用户分析实例
    from monitoring.analytics.user_analytics import UserAnalytics
    analytics = UserAnalytics()
    
    try:
        await analytics.start_analytics()
        print("用户行为分析已启动")
        
        # 模拟用户行为
        user_id = "demo_user_123"
        session_id = "session_001"
        
        # 用户登录
        analytics.track_user_action(user_id, UserAction.LOGIN, session_id)
        print("用户登录")
        
        await asyncio.sleep(2)
        
        # 用户发送消息
        analytics.track_user_action(
            user_id, 
            UserAction.SEND_MESSAGE, 
            session_id,
            metadata={"message_length": 150, "message_type": "question"}
        )
        print("发送消息")
        
        await asyncio.sleep(1)
        
        # 系统响应
        analytics.track_user_action(
            user_id,
            UserAction.RECEIVE_RESPONSE,
            session_id,
            duration=1.5,
            metadata={"response_quality": "good"}
        )
        print("接收响应")
        
        await asyncio.sleep(2)
        
        # 用户浏览对话
        analytics.track_user_action(user_id, UserAction.VIEW_CONVERSATION, session_id)
        print("浏览对话")
        
        await asyncio.sleep(5)
        
        # 获取用户画像
        profile = analytics.get_user_profile(user_id)
        if profile:
            print(f"用户画像: {profile.segment.value}")
            print(f"  会话数: {profile.total_sessions}")
            print(f"  行为数: {profile.total_actions}")
            print(f"  满意度: {profile.satisfaction_score:.1f}")
        
        # 获取体验洞察
        insights = analytics.get_experience_insights(hours=1)
        if insights:
            print("体验洞察:")
            for insight in insights[:3]:
                print(f"  - {insight.metric_name}: {insight.trend}")
    
    finally:
        await analytics.stop_analytics()
        print("用户行为分析已停止")

async def custom_monitoring_example():
    """自定义监控示例"""
    print("\n=== 自定义监控示例 ===")
    
    # 创建自定义监控管理器
    manager = create_monitoring_manager("custom_config.json")
    
    # 添加自定义事件回调
    async def custom_event_callback(status):
        print(f"组件状态变更: {status}")
        
        # 检查是否有组件停止运行
        stopped_components = [name for name, running in status.items() if not running]
        if stopped_components:
            print(f"警告: 以下组件已停止: {stopped_components}")
    
    manager.add_event_callback(custom_event_callback)
    
    try:
        await manager.start_monitoring()
        print("自定义监控已启动")
        
        # 运行一段时间
        await asyncio.sleep(60)
        
        # 获取综合数据
        analytics_data = manager.get_user_analytics()
        performance_data = manager.get_performance_metrics()
        
        print("用户分析摘要:")
        if 'user_segments' in analytics_data:
            for segment, data in analytics_data['user_segments'].items():
                print(f"  {segment}: {data['count']} 用户")
        
        print("性能指标摘要:")
        if 'recent_metrics' in performance_data:
            print(f"  指标数量: {len(performance_data['recent_metrics'])}")
        
        if 'tuning_history' in performance_data:
            print(f"  调优历史: {len(performance_data['tuning_history'])} 条记录")
        
        # 导出监控数据
        export_path = "monitoring_export.json"
        manager.export_monitoring_data(export_path, hours=1)
        print(f"监控数据已导出到: {export_path}")
    
    finally:
        await manager.stop_monitoring()
        print("自定义监控已停止")

async def health_monitoring_example():
    """健康监控示例"""
    print("\n=== 健康监控示例 ===")
    
    from monitoring.health.health_monitor import HealthMonitor, HealthCheck, ComponentType
    
    # 创建健康监控实例
    health_monitor = HealthMonitor()
    
    # 添加自定义健康检查
    custom_check = HealthCheck(
        component_name="custom_service",
        component_type=ComponentType.SERVICE,
        check_type="process",
        check_value="python",
        expected_result="running",
        timeout=5.0,
        interval=30.0
    )
    
    health_monitor.add_health_check(custom_check)
    print("添加自定义健康检查")
    
    # 添加健康状态回调
    async def health_callback(component_name: str, status: HealthStatus, result: dict):
        status_emoji = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "❌",
            HealthStatus.UNKNOWN: "❓"
        }
        emoji = status_emoji.get(status, "❓")
        print(f"{emoji} {component_name}: {status.value} - {result.get('message', '')}")
    
    health_monitor.add_health_callback(health_callback)
    
    try:
        await health_monitor.start_monitoring()
        print("健康监控已启动")
        
        # 运行一段时间观察健康状态
        for i in range(5):
            await asyncio.sleep(15)
            
            # 获取当前健康状态
            current_status = health_monitor.get_component_status()
            print(f"健康检查 {i+1}:")
            
            for component, status in current_status.items():
                status_emoji = {
                    HealthStatus.HEALTHY: "✅",
                    HealthStatus.WARNING: "⚠️", 
                    HealthStatus.CRITICAL: "❌"
                }
                emoji = status_emoji.get(status, "❓")
                print(f"  {emoji} {component}: {status.value}")
        
        # 获取健康历史
        health_history = health_monitor.get_health_history(hours=1)
        print(f"健康历史记录: {len(health_history)} 条")
        
    finally:
        await health_monitor.stop_monitoring()
        print("健康监控已停止")

async def alert_system_example():
    """告警系统示例"""
    print("\n=== 告警系统示例 ===")
    
    from monitoring.alerts.alert_system import AlertSystem, AlertRule, AlertSeverity
    
    # 创建告警系统实例
    alert_system = AlertSystem()
    
    # 添加自定义告警规则
    custom_rule = AlertRule(
        name="high_response_time",
        metric_name="app.response_time.avg",
        condition="greater_than",
        threshold=3.0,
        severity=AlertSeverity.WARNING,
        duration=60,
        description="响应时间过长告警"
    )
    
    alert_system.add_alert_rule(custom_rule)
    print("添加自定义告警规则")
    
    # 添加告警回调
    async def alert_callback(alert):
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🆘"
        }
        emoji = severity_emoji.get(alert.severity, "❓")
        print(f"{emoji} 告警: {alert.message}")
        
        # 模拟告警确认
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            alert_system.acknowledge_alert(alert.id, "demo_user")
            print(f"  告警已确认: {alert.id}")
    
    alert_system.add_alert_callback(alert_callback)
    
    try:
        await alert_system.start_monitoring()
        print("告警系统已启动")
        
        # 运行一段时间
        await asyncio.sleep(90)
        
        # 获取告警历史
        alert_history = alert_system.get_alert_history(hours=1)
        print(f"告警历史: {len(alert_history)} 条记录")
        
        # 获取活跃告警
        active_alerts = alert_system.get_active_alerts()
        print(f"活跃告警: {len(active_alerts)} 个")
        
    finally:
        await alert_system.stop_monitoring()
        print("告警系统已停止")

async def performance_tuning_example():
    """性能调优示例"""
    print("\n=== 性能调优示例 ===")
    
    from monitoring.tuning.auto_tuner import AutoTuner, TuningRule, TuningAction, ResourceType
    
    # 创建自动调优实例
    auto_tuner = AutoTuner()
    
    # 添加自定义调优规则
    tuning_rule = TuningRule(
        name="memory_optimization",
        resource_type=ResourceType.MEMORY,
        metric_name="memory.usage.percent",
        condition="greater_than",
        threshold=80.0,
        action=TuningAction.CLEANUP_RESOURCES,
        target_value=70.0,
        cooldown_period=300,
        description="内存使用率过高时清理资源"
    )
    
    auto_tuner.add_tuning_rule(tuning_rule)
    print("添加自定义调优规则")
    
    # 添加调优回调
    async def tuning_callback(action: TuningAction):
        print(f"🔧 调优动作: {action.rule_name} - {action.action_type.value}")
        print(f"  目标: {action.target_value}")
        print(f"  资源类型: {action.resource_type.value}")
    
    auto_tuner.add_tuning_callback(tuning_callback)
    
    try:
        await auto_tuner.start_tuning()
        print("自动调优已启动")
        
        # 运行一段时间
        await asyncio.sleep(120)
        
        # 获取调优历史
        tuning_history = auto_tuner.get_tuning_history(hours=1)
        print(f"调优历史: {len(tuning_history)} 条记录")
        
        # 获取推荐
        recommendations = auto_tuner.get_recommendations(hours=1)
        print(f"资源推荐: {len(recommendations)} 条")
        
        for rec in recommendations[:3]:
            print(f"  - {rec.resource_type.value}: {rec.current_usage:.1f}% -> {rec.recommended_usage:.1f}%")
            print(f"    理由: {rec.reasoning}")
    
    finally:
        await auto_tuner.stop_tuning()
        print("自动调优已停止")

async def comprehensive_integration_example():
    """综合集成示例"""
    print("\n=== 综合集成示例 ===")
    
    # 创建完整的监控管理系统
    manager = MonitoringManager()
    
    # 添加综合事件回调
    async def comprehensive_callback(status):
        print(f"🔄 状态更新: {len([s for s in status.values() if s])}/{len(status)} 组件运行中")
        
        # 获取实时数据
        dashboard_data = manager.get_dashboard_data()
        alerts = manager.get_active_alerts()
        health = manager.get_health_status()
        
        # 打印摘要
        if 'system_status' in dashboard_data:
            sys_status = dashboard_data['system_status']
            cpu = sys_status.get('cpu_percent', 0)
            memory = sys_status.get('memory_percent', 0)
            print(f"  📊 系统状态 - CPU: {cpu:.1f}%, 内存: {memory:.1f}%")
        
        print(f"  🚨 活跃告警: {len(alerts)} 个")
        print(f"  ❤️ 健康组件: {sum(1 for s in health.values() if s == HealthStatus.HEALTHY)}/{len(health)}")
    
    manager.add_event_callback(comprehensive_callback)
    
    try:
        await manager.start_monitoring()
        print("综合监控系统已启动")
        
        # 运行5分钟，每30秒输出一次状态
        for i in range(10):
            await asyncio.sleep(30)
            print(f"\n--- 第 {i+1} 次状态检查 ---")
            
            # 获取完整状态
            full_status = manager.get_monitoring_status()
            print(f"运行时间: {full_status['uptime']:.0f}秒")
            print(f"组件状态: {full_status['component_status']}")
            
            # 获取用户分析数据
            analytics = manager.get_user_analytics()
            if 'user_segments' in analytics:
                print(f"用户分群: {analytics['user_segments']}")
            
            # 获取性能指标
            performance = manager.get_performance_metrics()
            if 'tuning_history' in performance:
                print(f"调优记录: {len(performance['tuning_history'])} 条")
        
        # 生成最终报告
        print("\n📋 生成最终监控报告...")
        manager.export_monitoring_data("final_monitoring_report.json", hours=1)
        print("报告已保存到: final_monitoring_report.json")
        
    finally:
        await manager.stop_monitoring()
        print("综合监控系统已停止")

async def main():
    """主函数 - 运行所有示例"""
    print("诺玛Agent监控和优化系统示例")
    print("=" * 50)
    
    examples = [
        ("基本监控", basic_monitoring_example),
        ("用户行为跟踪", user_behavior_tracking_example),
        ("健康监控", health_monitoring_example),
        ("告警系统", alert_system_example),
        ("性能调优", performance_tuning_example),
        ("自定义监控", custom_monitoring_example),
        ("综合集成", comprehensive_integration_example)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n🚀 开始运行: {name}")
            await example_func()
            print(f"✅ {name} 完成")
            
            # 等待一段时间再运行下一个示例
            await asyncio.sleep(5)
            
        except Exception as e:
            print(f"❌ {name} 出错: {e}")
            continue
    
    print("\n🎉 所有示例运行完成！")
    print("\n查看生成的文件:")
    for file_path in ["monitoring_export.json", "final_monitoring_report.json", "dashboard_data.json"]:
        if Path(file_path).exists():
            print(f"  📄 {file_path}")

if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())