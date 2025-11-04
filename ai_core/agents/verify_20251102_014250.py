#!/usr/bin/env python3
"""
诺玛Agent监控系统快速验证脚本
"""

import sys
from pathlib import Path

def quick_verify():
    """快速验证监控系统"""
    print("🔍 诺玛Agent监控系统快速验证")
    print("-" * 40)
    
    # 检查文件是否存在
    required_files = [
        "__init__.py",
        "monitoring_manager.py", 
        "monitoring_config.json",
        "dashboard/monitoring_dashboard.py",
        "metrics/performance_collector.py",
        "alerts/alert_system.py",
        "tuning/auto_tuner.py",
        "analytics/user_analytics.py",
        "health/health_monitor.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少以下文件:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ 所有必需文件存在")
    
    # 测试导入
    try:
        from monitoring import quick_start, MonitoringManager
        print("✅ 模块导入成功")
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    
    # 测试创建管理器
    try:
        manager = quick_start()
        print("✅ 监控管理器创建成功")
    except Exception as e:
        print(f"❌ 监控管理器创建失败: {e}")
        return False
    
    print("-" * 40)
    print("🎉 监控系统验证通过！")
    print("\n使用说明:")
    print("1. 运行完整测试: python test_system.py")
    print("2. 查看使用示例: python examples.py")
    print("3. 快速开始使用:")
    print("   from monitoring import quick_start")
    print("   manager = quick_start()")
    print("   await manager.start_monitoring()")
    
    return True

if __name__ == "__main__":
    success = quick_verify()
    sys.exit(0 if success else 1)