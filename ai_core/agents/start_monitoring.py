#!/usr/bin/env python3
"""
诺玛Agent监控系统快速启动脚本
一键启动完整的监控和优化系统
"""

import sys
import os
import asyncio
import time
import signal
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class NormaMonitoringLauncher:
    """诺玛监控系统启动器"""
    
    def __init__(self):
        self.manager = None
        self.is_running = False
        
    async def start_monitoring(self):
        """启动监控系统"""
        print("🚀 启动诺玛Agent监控系统...")
        print("=" * 60)
        
        try:
            # 导入监控组件
            from monitoring import create_monitoring_manager
            
            # 创建监控管理器
            print("📦 初始化监控管理器...")
            self.manager = create_monitoring_manager()
            print("✅ 监控管理器初始化完成")
            
            # 启动监控
            print("🔧 启动监控组件...")
            await self.manager.start_monitoring()
            self.is_running = True
            print("✅ 监控系统启动成功")
            
            # 显示启动信息
            self._show_startup_info()
            
            # 主监控循环
            await self._main_loop()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  接收到中断信号，正在停止监控...")
        except Exception as e:
            print(f"\n❌ 启动过程中出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self._cleanup()
    
    def _show_startup_info(self):
        """显示启动信息"""
        print("\n" + "=" * 60)
        print("🎉 诺玛Agent监控系统已启动")
        print("=" * 60)
        print("📊 监控组件:")
        print("   ✓ 实时监控仪表板")
        print("   ✓ 性能指标收集器")
        print("   ✓ 智能告警系统")
        print("   ✓ 自动性能调优器")
        print("   ✓ 用户行为分析器")
        print("   ✓ 系统健康监控器")
        print("\n💡 提示:")
        print("   - 按 Ctrl+C 停止监控")
        print("   - 查看 logs/monitoring.log 获取详细日志")
        print("   - 配置文件: monitoring_config.json")
        print("=" * 60)
    
    async def _main_loop(self):
        """主监控循环"""
        print("\n📈 开始监控 (按 Ctrl+C 停止)...")
        
        try:
            while self.is_running:
                # 获取监控状态
                status = self.manager.get_monitoring_status()
                
                # 显示状态摘要
                uptime = status['uptime']
                components = status['component_status']
                running_count = sum(1 for v in components.values() if v)
                
                print(f"\r⏱️  运行时间: {uptime:.0f}s | 组件: {running_count}/{len(components)} | "
                      f"时间: {time.strftime('%H:%M:%S')}", end="", flush=True)
                
                # 每分钟生成详细报告
                if int(uptime) % 60 == 0 and uptime > 0:
                    await self._generate_minute_report()
                
                await asyncio.sleep(10)  # 每10秒更新一次
                
        except asyncio.CancelledError:
            pass
    
    async def _generate_minute_report(self):
        """生成每分钟报告"""
        try:
            # 获取各项数据
            dashboard_data = self.manager.get_dashboard_data()
            alerts = self.manager.get_active_alerts()
            health_status = self.manager.get_health_status()
            
            print(f"\n📊 {time.strftime('%H:%M:%S')} - 监控报告")
            print("-" * 40)
            
            # 系统状态
            if 'system_status' in dashboard_data:
                sys_status = dashboard_data['system_status']
                print(f"CPU: {sys_status.get('cpu_percent', 0):.1f}% | "
                      f"内存: {sys_status.get('memory_percent', 0):.1f}% | "
                      f"磁盘: {sys_status.get('disk_percent', 0):.1f}%")
            
            # 告警状态
            print(f"活跃告警: {len(alerts)} 个")
            if alerts:
                for alert in alerts[:3]:  # 显示前3个告警
                    print(f"  🚨 {alert.severity.value}: {alert.message[:50]}...")
            
            # 健康状态
            healthy_count = sum(1 for status in health_status.values() 
                              if str(status) == 'HealthStatus.HEALTHY')
            print(f"健康组件: {healthy_count}/{len(health_status)}")
            
        except Exception as e:
            print(f"\n⚠️  生成报告时出错: {e}")
    
    async def _cleanup(self):
        """清理资源"""
        print("\n🛑 正在停止监控系统...")
        
        try:
            if self.manager:
                await self.manager.stop_monitoring()
                print("✅ 监控系统已停止")
        except Exception as e:
            print(f"⚠️  停止监控时出错: {e}")
        
        print("👋 感谢使用诺玛Agent监控系统！")
    
    def setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            print(f"\n\n📡 接收到信号 {signum}")
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

async def quick_demo():
    """快速演示模式"""
    print("🎯 诺玛Agent监控系统快速演示")
    print("=" * 50)
    
    try:
        from monitoring import create_monitoring_manager
        
        # 创建监控管理器
        manager = create_monitoring_manager()
        
        # 启动监控
        print("🚀 启动监控 (演示模式，30秒)...")
        await manager.start_monitoring()
        
        # 运行30秒演示
        for i in range(30):
            await asyncio.sleep(1)
            if i % 5 == 0:
                status = manager.get_monitoring_status()
                print(f"⏱️  第{i+1}秒 - 运行时间: {status['uptime']:.1f}秒")
        
        # 停止监控
        await manager.stop_monitoring()
        print("\n✅ 演示完成！")
        
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")

def show_help():
    """显示帮助信息"""
    print("""
🎯 诺玛Agent监控系统启动器

用法:
  python start_monitoring.py [选项]

选项:
  --demo          运行30秒快速演示
  --help          显示此帮助信息
  --version       显示版本信息

示例:
  python start_monitoring.py          # 启动完整监控系统
  python start_monitoring.py --demo   # 运行快速演示

功能特性:
  ✓ 实时性能监控
  ✓ 智能告警系统
  ✓ 自动性能调优
  ✓ 用户行为分析
  ✓ 系统健康检查
  ✓ 自动故障恢复

更多信息请查看 README.md 文档。
""")

def show_version():
    """显示版本信息"""
    print("""
诺玛Agent监控系统
版本: 1.0.0
作者: 皇
构建时间: 2025-10-31
""")

def main():
    """主函数"""
    print("🚀 诺玛Agent监控系统启动器")
    print("作者: 皇")
    print("版本: 1.0.0")
    print("=" * 60)
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == "--help" or arg == "-h":
            show_help()
            return
        elif arg == "--version" or arg == "-v":
            show_version()
            return
        elif arg == "--demo":
            asyncio.run(quick_demo())
            return
        else:
            print(f"❌ 未知参数: {arg}")
            print("使用 --help 查看帮助信息")
            return
    
    # 启动完整监控系统
    launcher = NormaMonitoringLauncher()
    launcher.setup_signal_handlers()
    
    try:
        asyncio.run(launcher.start_monitoring())
    except KeyboardInterrupt:
        print("\n👋 用户中断，监控系统已停止")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()