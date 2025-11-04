"""
系统监控智能体 - 负责系统性能和健康监控
"""
import asyncio
import psutil
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
import json
import logging
from ..core.base_agent import BaseAgent, Task, TaskPriority


class SystemMonitorAgent(BaseAgent):
    """系统监控智能体"""
    
    def __init__(self, agent_id: str = None, config: Dict[str, Any] = None):
        super().__init__(
            agent_id=agent_id or f"system-monitor-{int(time.time())}",
            agent_type="system_monitor",
            config=config or {}
        )
        self.monitoring_interval = self.config.get("interval", 30)  # 监控间隔秒
        self.alert_thresholds = self.config.get("thresholds", {
            "cpu": 80.0,
            "memory": 85.0,
            "disk": 90.0,
            "network": 1000.0  # MB/s
        })
        self.monitoring_data = []
        self.alerts = []
        self.logger = logging.getLogger("agent.system.monitor")
        
    async def initialize(self) -> bool:
        """初始化系统监控智能体"""
        try:
            self.logger.info("初始化系统监控智能体...")
            
            # 设置监控能力
            self.capabilities = [
                "cpu_monitoring",
                "memory_monitoring", 
                "disk_monitoring",
                "network_monitoring",
                "process_monitoring",
                "alert_generation",
                "performance_analysis"
            ]
            
            # 启动监控循环
            asyncio.create_task(self._monitoring_loop())
            
            # 启动告警检查
            asyncio.create_task(self._alert_checker())
            
            self.logger.info("系统监控智能体初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            return False
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """处理监控任务"""
        try:
            task_type = task.payload.get("type")
            
            if task_type == "get_system_status":
                return await self._get_system_status()
            elif task_type == "get_performance_metrics":
                return await self._get_performance_metrics()
            elif task_type == "get_process_info":
                return await self._get_process_info(task.payload.get("pid"))
            elif task_type == "get_disk_usage":
                return await self._get_disk_usage()
            elif task_type == "get_network_stats":
                return await self._get_network_stats()
            elif task_type == "set_alert_threshold":
                return await self._set_alert_threshold(task.payload)
            elif task_type == "get_monitoring_history":
                return await self._get_monitoring_history(task.payload.get("duration", 3600))
            else:
                return {"status": "error", "message": f"未知的任务类型: {task_type}"}
                
        except Exception as e:
            self.logger.error(f"任务处理失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def shutdown(self) -> bool:
        """关闭系统监控智能体"""
        try:
            self.logger.info("关闭系统监控智能体...")
            await super().stop()
            self.logger.info("系统监控智能体已关闭")
            return True
        except Exception as e:
            self.logger.error(f"关闭失败: {e}")
            return False
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            # CPU信息
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # 内存信息
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # 磁盘信息
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # 网络信息
            network_io = psutil.net_io_counters()
            
            # 进程信息
            processes = len(psutil.pids())
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count,
                    "frequency": cpu_freq.current if cpu_freq else None
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "swap": {
                    "total": swap.total,
                    "used": swap.used,
                    "free": swap.free,
                    "percent": swap.percent
                },
                "disk": {
                    "total": disk_usage.total,
                    "used": disk_usage.used,
                    "free": disk_usage.free,
                    "percent": (disk_usage.used / disk_usage.total) * 100
                },
                "network": {
                    "bytes_sent": network_io.bytes_sent,
                    "bytes_recv": network_io.bytes_recv,
                    "packets_sent": network_io.packets_sent,
                    "packets_recv": network_io.packets_recv
                },
                "processes": processes,
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"获取系统状态失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        try:
            # 获取历史数据
            recent_data = self.monitoring_data[-60:]  # 最近60个数据点
            
            if not recent_data:
                return {"status": "error", "message": "没有监控数据"}
            
            # 计算平均值
            avg_cpu = sum(d["cpu_percent"] for d in recent_data) / len(recent_data)
            avg_memory = sum(d["memory_percent"] for d in recent_data) / len(recent_data)
            avg_disk = sum(d["disk_percent"] for d in recent_data) / len(recent_data)
            
            return {
                "status": "success",
                "metrics": {
                    "cpu": {
                        "average": avg_cpu,
                        "current": recent_data[-1]["cpu_percent"],
                        "max": max(d["cpu_percent"] for d in recent_data),
                        "min": min(d["cpu_percent"] for d in recent_data)
                    },
                    "memory": {
                        "average": avg_memory,
                        "current": recent_data[-1]["memory_percent"],
                        "max": max(d["memory_percent"] for d in recent_data),
                        "min": min(d["memory_percent"] for d in recent_data)
                    },
                    "disk": {
                        "average": avg_disk,
                        "current": recent_data[-1]["disk_percent"],
                        "max": max(d["disk_percent"] for d in recent_data),
                        "min": min(d["disk_percent"] for d in recent_data)
                    }
                },
                "data_points": len(recent_data),
                "time_range": f"{len(recent_data) * self.monitoring_interval} seconds"
            }
            
        except Exception as e:
            self.logger.error(f"获取性能指标失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _get_process_info(self, pid: int = None) -> Dict[str, Any]:
        """获取进程信息"""
        try:
            if pid:
                # 获取特定进程
                process = psutil.Process(pid)
                return {
                    "status": "success",
                    "process": {
                        "pid": process.pid,
                        "name": process.name(),
                        "status": process.status(),
                        "cpu_percent": process.cpu_percent(),
                        "memory_percent": process.memory_percent(),
                        "memory_info": process.memory_info()._asdict(),
                        "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
                        "num_threads": process.num_threads()
                    }
                }
            else:
                # 获取所有进程
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        processes.append(proc.info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # 按CPU使用率排序
                processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
                
                return {
                    "status": "success",
                    "processes": processes[:20],  # 返回前20个
                    "total": len(processes)
                }
                
        except Exception as e:
            self.logger.error(f"获取进程信息失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _get_disk_usage(self) -> Dict[str, Any]:
        """获取磁盘使用情况"""
        try:
            disk_usage = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage.append({
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": (usage.used / usage.total) * 100
                    })
                except PermissionError:
                    continue
            
            return {
                "status": "success",
                "disks": disk_usage
            }
            
        except Exception as e:
            self.logger.error(f"获取磁盘使用情况失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _get_network_stats(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        try:
            net_io = psutil.net_io_counters()
            net_connections = len(psutil.net_connections())
            
            # 网络接口统计
            net_if_stats = {}
            for interface, stats in psutil.net_if_stats().items():
                net_if_stats[interface] = {
                    "is_up": stats.isup,
                    "mtu": stats.mtu,
                    "speed": stats.speed
                }
            
            return {
                "status": "success",
                "network_io": {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                    "errin": net_io.errin,
                    "errout": net_io.errout,
                    "dropin": net_io.dropin,
                    "dropout": net_io.dropout
                },
                "connections": net_connections,
                "interfaces": net_if_stats
            }
            
        except Exception as e:
            self.logger.error(f"获取网络统计失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _set_alert_threshold(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """设置告警阈值"""
        try:
            metric = payload.get("metric")
            threshold = payload.get("threshold")
            
            if metric and threshold is not None:
                self.alert_thresholds[metric] = threshold
                return {
                    "status": "success",
                    "message": f"已更新 {metric} 阈值为 {threshold}"
                }
            else:
                return {"status": "error", "message": "缺少必要参数"}
                
        except Exception as e:
            self.logger.error(f"设置告警阈值失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _get_monitoring_history(self, duration: int) -> Dict[str, Any]:
        """获取监控历史数据"""
        try:
            cutoff_time = datetime.now() - timedelta(seconds=duration)
            history = [
                data for data in self.monitoring_data 
                if data["timestamp"] >= cutoff_time
            ]
            
            return {
                "status": "success",
                "history": history,
                "count": len(history),
                "duration": duration
            }
            
        except Exception as e:
            self.logger.error(f"获取监控历史失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 收集系统指标
                status = await self._get_system_status()
                
                if status["status"] == "success":
                    # 存储监控数据
                    monitoring_point = {
                        "timestamp": datetime.now(),
                        "cpu_percent": status["cpu"]["usage_percent"],
                        "memory_percent": status["memory"]["percent"],
                        "disk_percent": status["disk"]["percent"],
                        "network_bytes": status["network"]["bytes_sent"] + status["network"]["bytes_recv"]
                    }
                    
                    self.monitoring_data.append(monitoring_point)
                    
                    # 保持数据在合理范围内（最近1000个数据点）
                    if len(self.monitoring_data) > 1000:
                        self.monitoring_data = self.monitoring_data[-1000:]
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _alert_checker(self):
        """告警检查器"""
        while self.is_running:
            try:
                if self.monitoring_data:
                    latest = self.monitoring_data[-1]
                    
                    # 检查CPU告警
                    if latest["cpu_percent"] > self.alert_thresholds["cpu"]:
                        alert = {
                            "type": "cpu_high",
                            "value": latest["cpu_percent"],
                            "threshold": self.alert_thresholds["cpu"],
                            "timestamp": latest["timestamp"]
                        }
                        self.alerts.append(alert)
                        self.logger.warning(f"CPU使用率告警: {latest['cpu_percent']:.1f}%")
                    
                    # 检查内存告警
                    if latest["memory_percent"] > self.alert_thresholds["memory"]:
                        alert = {
                            "type": "memory_high",
                            "value": latest["memory_percent"],
                            "threshold": self.alert_thresholds["memory"],
                            "timestamp": latest["timestamp"]
                        }
                        self.alerts.append(alert)
                        self.logger.warning(f"内存使用率告警: {latest['memory_percent']:.1f}%")
                    
                    # 检查磁盘告警
                    if latest["disk_percent"] > self.alert_thresholds["disk"]:
                        alert = {
                            "type": "disk_high",
                            "value": latest["disk_percent"],
                            "threshold": self.alert_thresholds["disk"],
                            "timestamp": latest["timestamp"]
                        }
                        self.alerts.append(alert)
                        self.logger.warning(f"磁盘使用率告警: {latest['disk_percent']:.1f}%")
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"告警检查错误: {e}")
                await asyncio.sleep(self.monitoring_interval)