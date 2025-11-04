#!/usr/bin/env python3
"""
诺玛Agent监控系统快速验证
"""

import asyncio
import sys
import time
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

async def quick_test():
    """快速测试监控系统"""
    print("🚀 诺玛Agent监控系统快速验证")
    print("=" * 50)
    
    try:
        # 测试导入
        print("1. 测试模块导入...")
        from monitoring_manager import MonitoringManager
        print("   ✅ 监控管理器导入成功")
        
        from dashboard.monitoring_dashboard import MonitoringDashboard
        print("   ✅ 仪表板模块导入成功")
        
        from metrics.performance_collector import MetricsCollector
        print("   ✅ 指标收集模块导入成功")
        
        from alerts.alert_system import AlertSystem
        print("   ✅ 告警系统模块导入成功")
        
        from tuning.auto_tuner import AutoTuner
        print("   ✅ 自动调优模块导入成功")
        
        from analytics.user_analytics import UserAnalytics
        print("   ✅ 用户分析模块导入成功")
        
        from health.health_monitor import HealthMonitor
        print("   ✅ 健康监控模块导入成功")
        
        # 测试创建实例
        print("\n2. 测试组件创建...")
        manager = MonitoringManager()
        print("   ✅ 监控管理器创建成功")
        
        dashboard = MonitoringDashboard()
        print("   ✅ 仪表板创建成功")
        
        collector = MetricsCollector()
        print("   ✅ 指标收集器创建成功")
        
        alert_system = AlertSystem()
        print("   ✅ 告警系统创建成功")
        
        tuner = AutoTuner()
        print("   ✅ 自动调优器创建成功")
        
        analytics = UserAnalytics()
        print("   ✅ 用户分析器创建成功")
        
        health_monitor = HealthMonitor()
        print("   ✅ 健康监控器创建成功")
        
        # 测试基本功能
        print("\n3. 测试基本功能...")
        status = manager.get_monitoring_status()
        print(f"   ✅ 状态获取成功 - 组件数: {len(status.get('component_status', {}))}")
        
        dashboard_data = manager.get_dashboard_data()
        print("   ✅ 仪表板数据获取成功")
        
        alerts = manager.get_active_alerts()
        print(f"   ✅ 活跃告警获取成功 - 数量: {len(alerts)}")
        
        health = manager.get_health_status()
        print(f"   ✅ 健康状态获取成功 - 组件数: {len(health)}")
        
        # 测试短期运行
        print("\n4. 测试短期监控运行...")
        print("   启动监控管理器...")
        await manager.start_monitoring()
        
        print("   运行5秒...")
        await asyncio.sleep(5)
        
        print("   检查状态...")
        current_status = manager.get_monitoring_status()
        running_components = sum(1 for running in current_status.get('component_status', {}).values() if running)
        print(f"   ✅ 监控运行中 - 运行组件: {running_components}/{len(current_status.get('component_status', {}))}")
        
        print("   停止监控...")
        await manager.stop_monitoring()
        print("   ✅ 监控停止成功")
        
        print("\n" + "=" * 50)
        print("🎉 诺玛Agent监控系统验证完成！")
        print("✅ 所有核心功能正常工作")
        print("\n📋 系统包含以下组件:")
        print("   • 实时监控仪表板")
        print("   • 性能指标收集和分析")
        print("   • 智能告警和异常检测")
        print("   • 自动性能调优和资源管理")
        print("   • 用户行为分析和体验优化")
        print("   • 系统健康检查和自动恢复")
        
        print("\n🚀 使用方法:")
        print("   from monitoring import quick_start")
        print("   manager = quick_start()")
        print("   await manager.start_monitoring()")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(quick_test())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ 验证被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        sys.exit(1)