"""
系统监控工具实现
"""

import psutil
import logging
import time
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta

from .base_tool import SystemMonitorTool
from ..core.models import ToolParameter, ToolCategory, SecurityLevel


logger = logging.getLogger(__name__)


class CPUMonitorTool(SystemMonitorTool):
    """CPU监控工具"""
    
    def get_tool_definition(self):
        from ..core.models import ToolDefinition
        return ToolDefinition(
            id="cpu_monitor",
            name="CPU监控",
            description="监控系统CPU使用率",
            category=ToolCategory.SYSTEM_MONITOR,
            security_level=SecurityLevel.LOW,
            parameters=[
                ToolParameter("duration", "int", "监控时长（秒）", False, 60),
                ToolParameter("interval", "float", "采样间隔（秒）", False, 1.0),
                ToolParameter("cores", "bool", "是否监控单个核心", False, False)
            ],
            timeout=120,
            tags=["cpu", "monitor", "performance"]
        )
    
    async def execute(self, duration: int = 60, interval: float = 1.0, cores: bool = False) -> Dict[str, Any]:
        """执行CPU监控"""
        self._update_execution_stats()
        
        start_time = time.time()
        cpu_samples = []
        
        try:
            while time.time() - start_time < duration:
                # 获取CPU使用率
                if cores:
                    cpu_per_core = psutil.cpu_percent(interval=interval, percpu=True)
                    cpu_samples.append({
                        "timestamp": datetime.now().isoformat(),
                        "total": psutil.cpu_percent(),
                        "cores": cpu_per_core
                    })
                else:
                    cpu_percent = psutil.cpu_percent(interval=interval)
                    cpu_samples.append({
                        "timestamp": datetime.now().isoformat(),
                        "total": cpu_percent
                    })
            
            # 计算统计信息
            if cpu_samples:
                cpu_values = [sample["total"] for sample in cpu_samples]
                avg_cpu = sum(cpu_values) / len(cpu_values)
                max_cpu = max(cpu_values)
                min_cpu = min(cpu_values)
            else:
                avg_cpu = max_cpu = min_cpu = 0
            
            return {
                "success": True,
                "metric": "cpu",
                "samples_count": len(cpu_samples),
                "statistics": {
                    "average": round(avg_cpu, 2),
                    "maximum": round(max_cpu, 2),
                    "minimum": round(min_cpu, 2)
                },
                "samples": cpu_samples[:100],  # 限制返回样本数量
                "monitoring_duration": duration,
                "cores_monitored": cores
            }
            
        except Exception as e:
            logger.error(f"CPU监控失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "metric": "cpu"
            }


class MemoryMonitorTool(SystemMonitorTool):
    """内存监控工具"""
    
    def get_tool_definition(self):
        from ..core.models import ToolDefinition
        return ToolDefinition(
            id="memory_monitor",
            name="内存监控",
            description="监控系统内存使用情况",
            category=ToolCategory.SYSTEM_MONITOR,
            security_level=SecurityLevel.LOW,
            parameters=[
                ToolParameter("duration", "int", "监控时长（秒）", False, 60),
                ToolParameter("interval", "float", "采样间隔（秒）", False, 1.0),
                ToolParameter("detailed", "bool", "是否显示详细信息", False, False)
            ],
            timeout=120,
            tags=["memory", "monitor", "performance"]
        )
    
    async def execute(self, duration: int = 60, interval: float = 1.0, detailed: bool = False) -> Dict[str, Any]:
        """执行内存监控"""
        self._update_execution_stats()
        
        start_time = time.time()
        memory_samples = []
        
        try:
            # 获取初始内存信息
            initial_memory = psutil.virtual_memory()
            
            while time.time() - start_time < duration:
                memory = psutil.virtual_memory()
                
                sample = {
                    "timestamp": datetime.now().isoformat(),
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percentage": memory.percent,
                    "free": memory.free,
                    "cached": getattr(memory, 'cached', 0)
                }
                
                if detailed:
                    # 获取进程内存信息
                    processes = []
                    for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'memory_info']):
                        try:
                            processes.append(proc.info)
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    
                    # 按内存使用率排序，取前10个
                    top_processes = sorted(processes, 
                                         key=lambda x: x.get('memory_percent', 0), 
                                         reverse=True)[:10]
                    
                    sample["top_processes"] = top_processes
                
                memory_samples.append(sample)
                await asyncio.sleep(interval)
            
            # 计算统计信息
            if memory_samples:
                memory_percentages = [sample["percentage"] for sample in memory_samples]
                avg_memory = sum(memory_percentages) / len(memory_percentages)
                max_memory = max(memory_percentages)
                min_memory = min(memory_percentages)
            else:
                avg_memory = max_memory = min_memory = 0
            
            return {
                "success": True,
                "metric": "memory",
                "initial_state": {
                    "total": initial_memory.total,
                    "available": initial_memory.available,
                    "percentage": initial_memory.percent
                },
                "samples_count": len(memory_samples),
                "statistics": {
                    "average_percentage": round(avg_memory, 2),
                    "maximum_percentage": round(max_memory, 2),
                    "minimum_percentage": round(min_memory, 2)
                },
                "samples": memory_samples[:100],  # 限制返回样本数量
                "monitoring_duration": duration,
                "detailed": detailed
            }
            
        except Exception as e:
            logger.error(f"内存监控失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "metric": "memory"
            }


class DiskMonitorTool(SystemMonitorTool):
    """磁盘监控工具"""
    
    def get_tool_definition(self):
        from ..core.models import ToolDefinition
        return ToolDefinition(
            id="disk_monitor",
            name="磁盘监控",
            description="监控系统磁盘使用情况",
            category=ToolCategory.SYSTEM_MONITOR,
            security_level=SecurityLevel.LOW,
            parameters=[
                ToolParameter("path", "str", "监控路径", False, "/"),
                ToolParameter("io_stats", "bool", "是否监控IO统计", False, False)
            ],
            timeout=60,
            tags=["disk", "monitor", "storage"]
        )
    
    async def execute(self, path: str = "/", io_stats: bool = False) -> Dict[str, Any]:
        """执行磁盘监控"""
        self._update_execution_stats()
        
        try:
            # 获取磁盘使用情况
            disk_usage = psutil.disk_usage(path)
            
            # 获取磁盘IO统计
            io_counters = None
            if io_stats:
                io_counters = psutil.disk_io_counters(perdisk=True)
            
            # 获取磁盘挂载点信息
            disk_partitions = psutil.disk_partitions()
            
            result = {
                "success": True,
                "metric": "disk",
                "path": path,
                "usage": {
                    "total": disk_usage.total,
                    "used": disk_usage.used,
                    "free": disk_usage.free,
                    "percentage": round((disk_usage.used / disk_usage.total) * 100, 2)
                },
                "partitions": [
                    {
                        "device": partition.device,
                        "mountpoint": partition.mountpoint,
                        "fstype": partition.fstype,
                        "opts": partition.opts
                    }
                    for partition in disk_partitions
                ]
            }
            
            if io_counters:
                result["io_counters"] = {
                    device: {
                        "read_count": counter.read_count,
                        "write_count": counter.write_count,
                        "read_bytes": counter.read_bytes,
                        "write_bytes": counter.write_bytes,
                        "read_time": counter.read_time,
                        "write_time": counter.write_time
                    }
                    for device, counter in io_counters.items()
                }
            
            return result
            
        except Exception as e:
            logger.error(f"磁盘监控失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "metric": "disk"
            }


class NetworkMonitorTool(SystemMonitorTool):
    """网络监控工具"""
    
    def get_tool_definition(self):
        from ..core.models import ToolDefinition
        return ToolDefinition(
            id="network_monitor",
            name="网络监控",
            description="监控系统网络使用情况",
            category=ToolCategory.SYSTEM_MONITOR,
            security_level=SecurityLevel.LOW,
            parameters=[
                ToolParameter("duration", "int", "监控时长（秒）", False, 60),
                ToolParameter("interface", "str", "网络接口", False, None),
                ToolParameter("detailed", "bool", "是否显示详细信息", False, False)
            ],
            timeout=120,
            tags=["network", "monitor", "traffic"]
        )
    
    async def execute(self, duration: int = 60, interface: str = None, detailed: bool = False) -> Dict[str, Any]:
        """执行网络监控"""
        self._update_execution_stats()
        
        start_time = time.time()
        network_samples = []
        
        try:
            # 获取初始网络统计
            initial_counters = psutil.net_io_counters(pernic=True)
            
            while time.time() - start_time < duration:
                current_counters = psutil.net_io_counters(pernic=True)
                
                for nic, counter in current_counters.items():
                    if interface and nic != interface:
                        continue
                    
                    sample = {
                        "timestamp": datetime.now().isoformat(),
                        "interface": nic,
                        "bytes_sent": counter.bytes_sent,
                        "bytes_recv": counter.bytes_recv,
                        "packets_sent": counter.packets_sent,
                        "packets_recv": counter.packets_recv,
                        "errin": counter.errin,
                        "errout": counter.errout,
                        "dropin": counter.dropin,
                        "dropout": counter.dropout
                    }
                    
                    # 计算速率（如果有初始数据）
                    if nic in initial_counters:
                        time_diff = time.time() - start_time
                        if time_diff > 0:
                            bytes_sent_rate = (counter.bytes_sent - initial_counters[nic].bytes_sent) / time_diff
                            bytes_recv_rate = (counter.bytes_recv - initial_counters[nic].bytes_recv) / time_diff
                            sample["bytes_sent_rate"] = bytes_sent_rate
                            sample["bytes_recv_rate"] = bytes_recv_rate
                    
                    network_samples.append(sample)
                
                await asyncio.sleep(1)
            
            return {
                "success": True,
                "metric": "network",
                "samples_count": len(network_samples),
                "monitoring_duration": duration,
                "interface": interface,
                "detailed": detailed,
                "samples": network_samples[:200]  # 限制返回样本数量
            }
            
        except Exception as e:
            logger.error(f"网络监控失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "metric": "network"
            }


class ProcessMonitorTool(SystemMonitorTool):
    """进程监控工具"""
    
    def get_tool_definition(self):
        from ..core.models import ToolDefinition
        return ToolDefinition(
            id="process_monitor",
            name="进程监控",
            description="监控系统进程信息",
            category=ToolCategory.SYSTEM_MONITOR,
            security_level=SecurityLevel.MEDIUM,
            parameters=[
                ToolParameter("sort_by", "str", "排序字段", False, "cpu_percent", options=["cpu_percent", "memory_percent", "pid", "name"]),
                ToolParameter("limit", "int", "返回进程数量限制", False, 10),
                ToolParameter("detailed", "bool", "是否显示详细信息", False, False)
            ],
            timeout=60,
            tags=["process", "monitor", "system"]
        )
    
    async def execute(self, sort_by: str = "cpu_percent", limit: int = 10, detailed: bool = False) -> Dict[str, Any]:
        """执行进程监控"""
        self._update_execution_stats()
        
        try:
            processes = []
            
            for proc in psutil.process_iter([
                'pid', 'name', 'cpu_percent', 'memory_percent', 'memory_info',
                'create_time', 'status', 'num_threads', 'connections'
            ]):
                try:
                    proc_info = proc.info
                    
                    if detailed:
                        # 获取更详细的信息
                        try:
                            proc_info['cmdline'] = proc.cmdline()
                            proc_info['cwd'] = proc.cwd()
                            proc_info['exe'] = proc.exe()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    
                    processes.append(proc_info)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # 排序
            processes.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
            top_processes = processes[:limit]
            
            # 计算系统总体统计
            total_processes = len(processes)
            running_processes = len([p for p in processes if p.get('status') == 'running'])
            
            return {
                "success": True,
                "metric": "process",
                "total_processes": total_processes,
                "running_processes": running_processes,
                "sort_by": sort_by,
                "limit": limit,
                "top_processes": top_processes,
                "detailed": detailed
            }
            
        except Exception as e:
            logger.error(f"进程监控失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "metric": "process"
            }