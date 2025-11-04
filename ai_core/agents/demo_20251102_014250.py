#!/usr/bin/env python3
"""
诺玛Agent监控系统核心功能演示
展示监控系统的关键特性和使用方法
"""

import sys
import os
import asyncio
import time
import json
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def demo_monitoring_features():
    """演示监控系统的核心功能"""
    print("🎯 诺玛Agent监控系统核心功能演示")
    print("=" * 60)
    
    try:
        # 1. 导入和创建监控管理器
        print("\n📦 1. 导入监控组件...")
        from monitoring import (
            MonitoringManager, 
            create_monitoring_manager,
            MetricType,
            AlertSeverity,
            HealthStatus,
            UserAction
        )
        print("✅ 监控组件导入成功")
        
        # 2. 创建监控管理器实例
        print("\n🔧 2. 创建监控管理器...")
        manager = create_monitoring_manager("monitoring_config.json")
        print("✅ 监控管理器创建成功")
        
        # 3. 展示配置信息
        print("\n⚙️  3. 监控配置信息:")
        config = manager.config
        print(f"   - 监控启用: {config.get('monitoring', {}).get('enabled', False)}")
        print(f"   - 数据保留天数: {config.get('monitoring', {}).get('data_retention_days', 30)}")
        print(f"   - 告警启用: {config.get('alerts', {}).get('enabled', False)}")
        print(f"   - 自动调优启用: {config.get('tuning', {}).get('enabled', False)}")
        print(f"   - 用户分析启用: {config.get('analytics', {}).get('enabled', False)}")
        print(f"   - 健康检查启用: {config.get('health', {}).get('enabled', False)}")
        
        # 4. 获取初始状态
        print("\n📊 4. 获取监控状态...")
        status = manager.get_monitoring_status()
        print(f"   - 监控运行状态: {'运行中' if status['is_running'] else '已停止'}")
        print(f"   - 运行时间: {status['uptime']:.1f}秒")
        print(f"   - 组件数量: {len(status['component_status'])}")
        
        # 5. 获取仪表板数据
        print("\n📈 5. 仪表板数据:")
        dashboard_data = manager.get_dashboard_data()
        print(f"   - 仪表板数据项: {len(dashboard_data)}")
        for key, value in list(dashboard_data.items())[:5]:
            print(f"     * {key}: {type(value).__name__}")
        
        # 6. 获取活跃告警
        print("\n🚨 6. 告警系统状态:")
        alerts = manager.get_active_alerts()
        print(f"   - 活跃告警数量: {len(alerts)}")
        
        # 7. 获取健康状态
        print("\n🏥 7. 健康检查状态:")
        health_status = manager.get_health_status()
        print(f"   - 监控组件数量: {len(health_status)}")
        
        # 8. 展示支持的指标类型
        print("\n📊 8. 支持的监控指标类型:")
        metric_types = [mt.value for mt in MetricType]
        for i, mt in enumerate(metric_types, 1):
            print(f"   {i:2d}. {mt}")
        
        # 9. 展示告警严重级别
        print("\n🚨 9. 告警严重级别:")
        alert_levels = [as_.value for as_ in AlertSeverity]
        for i, al in enumerate(alert_levels, 1):
            print(f"   {i}. {al}")
        
        # 10. 展示健康状态
        print("\n🏥 10. 健康状态类型:")
        health_states = [hs.value for hs in HealthStatus]
        for i, hs in enumerate(health_states, 1):
            print(f"   {i}. {hs}")
        
        # 11. 展示用户行为类型
        print("\n👤 11. 用户行为类型:")
        user_actions = [ua.value for ua in UserAction]
        for i, ua in enumerate(user_actions, 1):
            print(f"   {i:2d}. {ua}")
        
        # 12. 演示短期监控启动（5秒）
        print("\n⏱️  12. 启动短期监控测试（5秒）...")
        await manager.start_monitoring()
        print("✅ 监控已启动")
        
        # 运行5秒并显示状态变化
        for i in range(5):
            await asyncio.sleep(1)
            current_status = manager.get_monitoring_status()
            print(f"   第{i+1}秒 - 运行时间: {current_status['uptime']:.1f}秒")
        
        # 停止监控
        await manager.stop_monitoring()
        print("✅ 监控已停止")
        
        # 13. 最终状态报告
        print("\n📋 13. 最终监控报告:")
        final_status = manager.get_monitoring_status()
        print(f"   - 总运行时间: {final_status['uptime']:.1f}秒")
        print(f"   - 组件状态: {final_status['component_status']}")
        
        # 14. 导出示例数据
        print("\n💾 14. 导出监控数据示例...")
        export_data = {
            "monitoring_summary": {
                "status": final_status['is_running'],
                "uptime": final_status['uptime'],
                "components": final_status['component_status']
            },
            "dashboard_metrics": len(dashboard_data),
            "active_alerts": len(alerts),
            "health_components": len(health_status),
            "supported_features": {
                "real_time_monitoring": True,
                "intelligent_alerts": True,
                "auto_tuning": True,
                "user_analytics": True,
                "health_monitoring": True,
                "auto_recovery": True
            }
        }
        
        print("✅ 监控系统核心功能演示完成")
        print("\n🎉 诺玛Agent监控系统运行正常！")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_system_architecture():
    """显示系统架构"""
    print("\n🏗️  诺玛Agent监控系统架构:")
    print("=" * 60)
    
    architecture = {
        "核心组件": {
            "监控管理器": "统一管理所有监控组件",
            "仪表板": "实时数据可视化和状态展示",
            "指标收集器": "性能指标收集和分析",
            "告警系统": "智能告警和异常检测",
            "自动调优器": "性能调优和资源管理",
            "用户分析器": "用户行为分析和体验优化",
            "健康监控器": "系统健康检查和自动恢复"
        },
        "主要功能": [
            "实时性能监控",
            "多级智能告警",
            "自动性能调优",
            "用户行为分析",
            "系统健康检查",
            "自动故障恢复",
            "数据可视化",
            "配置管理"
        ],
        "技术特性": [
            "异步处理架构",
            "多线程并发",
            "SQLite数据存储",
            "实时数据流",
            "异常检测算法",
            "机器学习优化",
            "自动恢复机制",
            "事件驱动架构"
        ]
    }
    
    for category, items in architecture.items():
        print(f"\n📋 {category}:")
        if isinstance(items, dict):
            for component, description in items.items():
                print(f"   • {component}: {description}")
        elif isinstance(items, list):
            for item in items:
                print(f"   • {item}")

def show_usage_examples():
    """显示使用示例"""
    print("\n💡 使用示例:")
    print("=" * 60)
    
    examples = {
        "基本使用": '''
from monitoring import quick_start

# 快速启动监控
manager = quick_start()
await manager.start_monitoring()
        ''',
        
        "自定义配置": '''
from monitoring import create_monitoring_manager

# 使用自定义配置
manager = create_monitoring_manager("custom_config.json")
await manager.start_monitoring()
        ''',
        
        "获取监控数据": '''
# 获取各种监控数据
status = manager.get_monitoring_status()
dashboard = manager.get_dashboard_data()
alerts = manager.get_active_alerts()
health = manager.get_health_status()
        ''',
        
        "用户行为跟踪": '''
from monitoring import UserAction

# 跟踪用户行为
analytics.track_user_action(
    user_id="user123",
    action_type=UserAction.SEND_MESSAGE,
    session_id="session_001"
)
        '''
    }
    
    for example_name, code in examples.items():
        print(f"\n🔹 {example_name}:")
        print(code)

if __name__ == "__main__":
    print("🚀 诺玛Agent监控系统演示程序")
    print("作者: 皇")
    print("版本: 1.0.0")
    
    # 显示系统架构
    show_system_architecture()
    
    # 显示使用示例
    show_usage_examples()
    
    # 运行功能演示
    success = asyncio.run(demo_monitoring_features())
    
    if success:
        print("\n" + "=" * 60)
        print("🎊 演示完成！诺玛Agent监控系统已成功实现并测试")
        print("📚 更多信息请查看 README.md 文档")
        print("=" * 60)
    else:
        print("\n❌ 演示过程中遇到问题，请检查配置和依赖")
    
    sys.exit(0 if success else 1)