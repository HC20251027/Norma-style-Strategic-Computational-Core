"""
诺玛Agent监控管理器
统一管理所有监控组件
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import signal
import sys

# 导入所有监控组件 - 使用相对导入避免模块路径问题
from .dashboard.monitoring_dashboard import MonitoringDashboard, MetricType
from .metrics.performance_collector import MetricsCollector, PerformanceMetric
from .alerts.alert_system import AlertSystem, Alert, AlertSeverity
from .tuning.auto_tuner import AutoTuner, TuningAction, TuningActionRecord, ResourceType
from .analytics.user_analytics import UserAnalytics, UserAction
from .health.health_monitor import HealthMonitor, HealthStatus

class MonitoringManager:
    """诺玛Agent监控管理器"""
    
    def __init__(self, config_path: str = "monitoring_config.json"):
        self.config_path = config_path
        self.logger = self._setup_logging()
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化组件
        self.dashboard = MonitoringDashboard()
        self.metrics_collector = MetricsCollector()
        self.alert_system = AlertSystem()
        self.auto_tuner = AutoTuner()
        self.user_analytics = UserAnalytics()
        self.health_monitor = HealthMonitor()
        
        # 状态管理
        self.is_running = False
        self.start_time = None
        self.component_status = {}
        
        # 回调函数
        self.event_callbacks: List[Callable] = []
        
        # 设置组件间集成
        self._setup_integration()
        
        self.logger.info("监控管理器初始化完成")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("NormaMonitoring")
        logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 文件处理器
        file_handler = logging.FileHandler(log_dir / "monitoring.log")
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "monitoring": {
                "enabled": True,
                "log_level": "INFO",
                "data_retention_days": 30,
                "check_interval": 30
            },
            "dashboard": {
                "refresh_interval": 5,
                "metrics_buffer_size": 1000,
                "alert_buffer_size": 100
            },
            "metrics": {
                "collection_interval": 1,
                "aggregation_interval": 60,
                "anomaly_threshold": 2.5,
                "trend_window": 300
            },
            "alerts": {
                "enabled": True,
                "email_notifications": False,
                "webhook_notifications": False,
                "severity_levels": ["info", "warning", "critical"]
            },
            "tuning": {
                "enabled": True,
                "auto_recovery": True,
                "max_concurrent_actions": 3,
                "action_timeout": 300
            },
            "analytics": {
                "enabled": True,
                "session_timeout": 1800,
                "behavior_tracking": True,
                "user_segmentation": True
            },
            "health": {
                "enabled": True,
                "auto_recovery": True,
                "check_interval": 30,
                "status_retention_days": 7
            }
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # 合并配置
                self._merge_config(default_config, loaded_config)
                
                self.logger.info(f"配置文件加载成功: {self.config_path}")
            else:
                # 保存默认配置
                self._save_config(default_config)
                self.logger.info("创建默认配置文件")
                
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}，使用默认配置")
        
        return default_config
    
    def _merge_config(self, default: Dict, loaded: Dict):
        """合并配置"""
        for key, value in loaded.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self._merge_config(default[key], value)
                else:
                    default[key] = value
    
    def _save_config(self, config: Dict[str, Any]):
        """保存配置"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {e}")
    
    def _setup_integration(self):
        """设置组件间集成"""
        # 指标收集器 -> 告警系统
        async def metrics_to_alerts(metric: PerformanceMetric):
            """指标数据触发告警检查"""
            # 指标数据会自动触发告警检查
            pass
        
        self.metrics_collector.add_metric_callback(metrics_to_alerts)
        
        # 告警系统 -> 自动调优器
        async def alerts_to_tuning(alert: Alert):
            """告警触发调优"""
            if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.WARNING]:
                # 根据告警创建调优建议
                tuning_action = TuningActionRecord(
                    id=None,
                    rule_name=f"alert_{alert.rule_name}",
                    action_type=TuningAction.OPTIMIZE_CONFIG,
                    resource_type=ResourceType.AGENT_INSTANCES,  # 默认类型
                    target_value=alert.threshold,
                    timestamp=time.time()
                )
                
                # 这里可以触发调优逻辑
                pass
        
        self.alert_system.add_alert_callback(alerts_to_tuning)
        
        # 用户分析 -> 告警系统
        async def analytics_to_alerts(data: Any):
            """用户行为数据触发告警"""
            # 用户行为异常可以触发告警
            pass
        
        self.user_analytics.add_analytics_callback(analytics_to_alerts)
        
        # 健康监控 -> 自动调优器
        async def health_to_tuning(component_name: str, status: HealthStatus, result: Dict):
            """健康状态触发调优"""
            if status == HealthStatus.CRITICAL:
                # 健康状态严重时触发调优
                pass
        
        self.health_monitor.add_health_callback(health_to_tuning)
        
        # 健康监控 -> 告警系统
        async def health_to_alerts(component_name: str, status: HealthStatus, result: Dict):
            """健康状态触发告警"""
            if status == HealthStatus.CRITICAL:
                # 创建告警
                alert_message = f"组件 {component_name} 状态严重: {result.get('message', '')}"
                # 这里可以创建告警
                pass
        
        self.health_monitor.add_health_callback(health_to_alerts)
    
    async def start_monitoring(self):
        """启动监控"""
        if self.is_running:
            self.logger.warning("监控已在运行中")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        self.logger.info("启动诺玛Agent监控管理系统")
        
        try:
            # 并发启动所有组件
            tasks = [
                self.dashboard.start_monitoring(),
                self.metrics_collector.start_collection(),
                self.alert_system.start_monitoring(),
                self.auto_tuner.start_tuning(),
                self.user_analytics.start_analytics(),
                self.health_monitor.start_monitoring(),
                self._monitoring_loop()
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            self.logger.error(f"启动监控时出错: {e}")
            await self.stop_monitoring()
    
    async def stop_monitoring(self):
        """停止监控"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.logger.info("停止诺玛Agent监控管理系统")
        
        try:
            # 并发停止所有组件
            tasks = [
                self.dashboard.stop_monitoring(),
                self.metrics_collector.stop_collection(),
                self.alert_system.stop_monitoring(),
                self.auto_tuner.stop_tuning(),
                self.user_analytics.stop_analytics(),
                self.health_monitor.stop_monitoring()
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            self.logger.info("监控管理系统已停止")
            
        except Exception as e:
            self.logger.error(f"停止监控时出错: {e}")
    
    async def _monitoring_loop(self):
        """监控主循环"""
        while self.is_running:
            try:
                # 更新组件状态
                await self._update_component_status()
                
                # 生成综合报告
                await self._generate_comprehensive_report()
                
                # 触发事件回调
                await self._trigger_event_callbacks()
                
                await asyncio.sleep(60)  # 每分钟执行一次
                
            except Exception as e:
                self.logger.error(f"监控主循环出错: {e}")
                await asyncio.sleep(60)
    
    async def _update_component_status(self):
        """更新组件状态"""
        try:
            # 检查各组件运行状态
            self.component_status = {
                "dashboard": self.dashboard.is_monitoring,
                "metrics_collector": self.metrics_collector.is_collecting,
                "alert_system": self.alert_system.is_running,
                "auto_tuner": self.auto_tuner.is_running,
                "user_analytics": self.user_analytics.is_running,
                "health_monitor": self.health_monitor.is_running
            }
            
            # 检查是否有组件异常停止
            for component, is_running in self.component_status.items():
                if not is_running:
                    self.logger.warning(f"组件 {component} 已停止运行")
            
        except Exception as e:
            self.logger.error(f"更新组件状态时出错: {e}")
    
    async def _generate_comprehensive_report(self):
        """生成综合报告"""
        try:
            # 获取各组件数据
            dashboard_data = self.dashboard.get_dashboard_data()
            active_alerts = self.alert_system.get_active_alerts()
            health_status = self.health_monitor.get_component_status()
            
            # 生成综合报告
            report = {
                "timestamp": time.time(),
                "uptime": time.time() - self.start_time if self.start_time else 0,
                "component_status": self.component_status,
                "summary": {
                    "total_components": len(self.component_status),
                    "running_components": sum(1 for status in self.component_status.values() if status),
                    "active_alerts": len(active_alerts),
                    "critical_components": sum(1 for status in health_status.values() 
                                             if status == HealthStatus.CRITICAL),
                    "system_health_score": self._calculate_overall_health_score()
                },
                "dashboard": dashboard_data,
                "alerts": {
                    "active_count": len(active_alerts),
                    "recent_alerts": [
                        {
                            "rule_name": alert.rule_name,
                            "severity": alert.severity.value,
                            "message": alert.message,
                            "timestamp": alert.timestamp
                        }
                        for alert in active_alerts[:10]  # 最近10个告警
                    ]
                },
                "health": {
                    "component_count": len(health_status),
                    "healthy_components": sum(1 for status in health_status.values() 
                                            if status == HealthStatus.HEALTHY),
                    "critical_components": sum(1 for status in health_status.values() 
                                             if status == HealthStatus.CRITICAL)
                }
            }
            
            # 保存报告
            report_file = Path("comprehensive_report.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            self.logger.error(f"生成综合报告时出错: {e}")
    
    def _calculate_overall_health_score(self) -> float:
        """计算整体健康分数"""
        try:
            health_status = self.health_monitor.get_component_status()
            if not health_status:
                return 0.0
            
            # 根据组件状态计算分数
            total_components = len(health_status)
            healthy_components = sum(1 for status in health_status.values() 
                                   if status == HealthStatus.HEALTHY)
            warning_components = sum(1 for status in health_status.values() 
                                   if status == HealthStatus.WARNING)
            
            # 加权计算
            score = (healthy_components * 100 + warning_components * 70) / total_components
            return min(100.0, score)
            
        except Exception as e:
            self.logger.error(f"计算整体健康分数时出错: {e}")
            return 0.0
    
    async def _trigger_event_callbacks(self):
        """触发事件回调"""
        try:
            for callback in self.event_callbacks:
                await callback(self.component_status)
        except Exception as e:
            self.logger.error(f"触发事件回调时出错: {e}")
    
    def add_event_callback(self, callback: Callable):
        """添加事件回调函数"""
        self.event_callbacks.append(callback)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            "is_running": self.is_running,
            "start_time": self.start_time,
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "component_status": self.component_status,
            "config": self.config
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        return self.dashboard.get_dashboard_data()
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return self.alert_system.get_active_alerts()
    
    def get_health_status(self) -> Dict[str, HealthStatus]:
        """获取健康状态"""
        return self.health_monitor.get_component_status()
    
    def get_user_analytics(self) -> Dict[str, Any]:
        """获取用户分析数据"""
        return {
            "user_segments": self.user_analytics.get_user_segments_summary(),
            "recent_insights": [
                {
                    "metric_name": insight.metric_name,
                    "trend": insight.trend,
                    "insights": insight.insights,
                    "recommendations": insight.recommendations
                }
                for insight in self.user_analytics.get_experience_insights(hours=24)
            ]
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            "recent_metrics": self.metrics_collector.get_analysis_results(hours=1),
            "tuning_history": self.auto_tuner.get_tuning_history(hours=24),
            "recommendations": self.auto_tuner.get_recommendations(hours=24)
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        try:
            self._merge_config(self.config, new_config)
            self._save_config(self.config)
            self.logger.info("配置更新成功")
        except Exception as e:
            self.logger.error(f"更新配置失败: {e}")
    
    def export_monitoring_data(self, export_path: str, hours: int = 24):
        """导出监控数据"""
        try:
            export_data = {
                "export_time": time.time(),
                "period_hours": hours,
                "dashboard_data": self.get_dashboard_data(),
                "active_alerts": [
                    {
                        "rule_name": alert.rule_name,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "timestamp": alert.timestamp
                    }
                    for alert in self.get_active_alerts()
                ],
                "health_status": {
                    component: status.value 
                    for component, status in self.get_health_status().items()
                },
                "user_analytics": self.get_user_analytics(),
                "performance_metrics": self.get_performance_metrics()
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"监控数据导出成功: {export_path}")
            
        except Exception as e:
            self.logger.error(f"导出监控数据失败: {e}")

# 信号处理
def signal_handler(signum, frame):
    """信号处理器"""
    print(f"\n接收到信号 {signum}，正在停止监控...")
    # 这里应该调用监控管理器的停止方法
    sys.exit(0)

# 全局监控管理器实例
monitoring_manager = None

def create_monitoring_manager(config_path: str = "monitoring_config.json") -> MonitoringManager:
    """创建监控管理器实例"""
    global monitoring_manager
    monitoring_manager = MonitoringManager(config_path)
    return monitoring_manager

async def main():
    """主函数"""
    global monitoring_manager
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建监控管理器
    monitoring_manager = create_monitoring_manager()
    
    # 添加事件回调
    async def status_callback(status: Dict[str, bool]):
        print(f"组件状态更新: {status}")
    
    monitoring_manager.add_event_callback(status_callback)
    
    try:
        # 启动监控
        await monitoring_manager.start_monitoring()
        
        # 保持运行
        while monitoring_manager.is_running:
            await asyncio.sleep(10)
            
            # 打印状态摘要
            status = monitoring_manager.get_monitoring_status()
            print(f"监控运行中 - 运行时间: {status['uptime']:.0f}秒")
        
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在停止...")
    finally:
        if monitoring_manager:
            await monitoring_manager.stop_monitoring()

if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())