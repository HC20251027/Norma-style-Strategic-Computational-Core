"""
资源管理器
负责管理任务执行所需的资源，包括CPU、内存、GPU、存储等
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import psutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import os

# 配置日志
logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """资源类型枚举"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"
    CUSTOM = "custom"


class ResourceStatus(Enum):
    """资源状态枚举"""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    RESERVED = "reserved"
    UNAVAILABLE = "unavailable"


@dataclass
class Resource:
    """资源定义"""
    id: str
    name: str
    type: ResourceType
    capacity: float
    current_usage: float = 0.0
    status: ResourceStatus = ResourceStatus.AVAILABLE
    metadata: dict = field(default_factory=dict)
    allocated_tasks: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    
    @property
    def available_capacity(self) -> float:
        """获取可用容量"""
        return max(0, self.capacity - self.current_usage)
    
    @property
    def usage_percentage(self) -> float:
        """获取使用率百分比"""
        if self.capacity == 0:
            return 0.0
        return (self.current_usage / self.capacity) * 100


@dataclass
class ResourceRequest:
    """资源请求"""
    task_id: str
    resource_requirements: Dict[ResourceType, float]
    priority: int = 1  # 1-10, 10为最高优先级
    timeout: float = 30.0
    created_at: float = field(default_factory=time.time)
    allocated_resources: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending, allocated, failed, timeout


class ResourcePool:
    """资源池"""
    
    def __init__(self, name: str, resource_type: ResourceType):
        self.name = name
        self.resource_type = resource_type
        self.resources: Dict[str, Resource] = {}
        self.total_capacity = 0.0
        self.available_capacity = 0.0
        
    def add_resource(self, resource: Resource):
        """添加资源到池中"""
        self.resources[resource.id] = resource
        self.total_capacity += resource.capacity
        self.available_capacity += resource.available_capacity
        
        logger.info(f"资源已添加到池 {self.name}: {resource.name}")
    
    def remove_resource(self, resource_id: str) -> bool:
        """从池中移除资源"""
        if resource_id in self.resources:
            resource = self.resources[resource_id]
            self.total_capacity -= resource.capacity
            self.available_capacity -= resource.available_capacity
            del self.resources[resource_id]
            
            logger.info(f"资源已从池 {self.name} 移除: {resource.name}")
            return True
        return False
    
    def allocate(self, amount: float) -> Optional[Resource]:
        """分配资源"""
        # 找到最适合的资源
        best_resource = None
        for resource in self.resources.values():
            if (resource.status == ResourceStatus.AVAILABLE and 
                resource.available_capacity >= amount):
                if best_resource is None or resource.available_capacity < best_resource.available_capacity:
                    best_resource = resource
        
        if best_resource:
            best_resource.current_usage += amount
            best_resource.available_capacity -= amount
            
            if best_resource.available_capacity <= 0:
                best_resource.status = ResourceStatus.ALLOCATED
            
            self.available_capacity -= amount
            
            logger.debug(f"资源分配成功: {amount} 单位从 {best_resource.name}")
            return best_resource
        
        return None
    
    def deallocate(self, resource_id: str, amount: float):
        """释放资源"""
        if resource_id in self.resources:
            resource = self.resources[resource_id]
            resource.current_usage = max(0, resource.current_usage - amount)
            resource.available_capacity = min(resource.capacity, resource.available_capacity + amount)
            
            if resource.status == ResourceStatus.ALLOCATED and resource.available_capacity > 0:
                resource.status = ResourceStatus.AVAILABLE
            
            self.available_capacity += amount
            
            logger.debug(f"资源释放成功: {amount} 单位从 {resource.name}")
    
    def get_statistics(self) -> dict:
        """获取资源池统计信息"""
        return {
            'name': self.name,
            'resource_type': self.resource_type.value,
            'total_capacity': self.total_capacity,
            'available_capacity': self.available_capacity,
            'usage_percentage': ((self.total_capacity - self.available_capacity) / self.total_capacity * 100) if self.total_capacity > 0 else 0,
            'resource_count': len(self.resources),
            'available_resources': len([r for r in self.resources.values() if r.status == ResourceStatus.AVAILABLE])
        }


class ResourceManager:
    """资源管理器"""
    
    def __init__(self, enable_monitoring: bool = True):
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.pending_requests: Dict[str, ResourceRequest] = {}
        self.allocated_resources: Dict[str, Dict[str, float]] = {}  # task_id -> {resource_id: amount}
        
        self.monitoring_enabled = enable_monitoring
        self.monitoring_interval = 5.0  # 监控间隔（秒）
        self._monitoring_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        
        # 统计信息
        self.stats = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'failed_allocations': 0,
            'pending_requests': 0,
            'active_allocations': 0
        }
        
        # 初始化系统资源
        self._initialize_system_resources()
        
        logger.info("资源管理器初始化完成")
    
    def _initialize_system_resources(self):
        """初始化系统资源"""
        # CPU资源
        cpu_count = psutil.cpu_count()
        cpu_pool = ResourcePool("CPU Pool", ResourceType.CPU)
        
        # 为每个CPU核心创建一个资源
        for i in range(cpu_count):
            cpu_resource = Resource(
                id=f"cpu_{i}",
                name=f"CPU Core {i}",
                type=ResourceType.CPU,
                capacity=1.0,  # 每个核心100%
                metadata={'core_id': i, 'frequency': '2.5GHz'}
            )
            cpu_pool.add_resource(cpu_resource)
        
        self.resource_pools['cpu'] = cpu_pool
        
        # 内存资源
        memory = psutil.virtual_memory()
        memory_pool = ResourcePool("Memory Pool", ResourceType.MEMORY)
        
        memory_resource = Resource(
            id="memory_main",
            name="System Memory",
            type=ResourceType.MEMORY,
            capacity=memory.total / (1024**3),  # 转换为GB
            metadata={'total_gb': memory.total / (1024**3)}
        )
        memory_pool.add_resource(memory_resource)
        
        self.resource_pools['memory'] = memory_pool
        
        # 存储资源
        disk = psutil.disk_usage('/')
        storage_pool = ResourcePool("Storage Pool", ResourceType.STORAGE)
        
        storage_resource = Resource(
            id="storage_main",
            name="System Storage",
            type=ResourceType.STORAGE,
            capacity=disk.total / (1024**3),  # 转换为GB
            metadata={'total_gb': disk.total / (1024**3)}
        )
        storage_pool.add_resource(storage_resource)
        
        self.resource_pools['storage'] = storage_pool
        
        logger.info("系统资源初始化完成")
    
    def create_resource_pool(self, name: str, resource_type: ResourceType) -> ResourcePool:
        """创建资源池"""
        if name in self.resource_pools:
            raise ValueError(f"资源池 {name} 已存在")
        
        pool = ResourcePool(name, resource_type)
        self.resource_pools[name] = pool
        
        logger.info(f"资源池已创建: {name}")
        return pool
    
    def add_custom_resource(self, pool_name: str, resource: Resource):
        """添加自定义资源"""
        if pool_name not in self.resource_pools:
            raise ValueError(f"资源池 {pool_name} 不存在")
        
        self.resource_pools[pool_name].add_resource(resource)
    
    async def request_resources(self, request: ResourceRequest) -> bool:
        """请求资源"""
        with self._lock:
            self.pending_requests[request.task_id] = request
            self.stats['pending_requests'] += 1
            
            logger.info(f"收到资源请求: 任务 {request.task_id}")
            
            # 尝试立即分配
            success = await self._allocate_resources(request)
            
            if success:
                logger.info(f"资源分配成功: 任务 {request.task_id}")
            else:
                logger.warning(f"资源分配失败: 任务 {request.task_id}")
            
            return success
    
    async def _allocate_resources(self, request: ResourceRequest) -> bool:
        """分配资源"""
        allocated = {}
        
        try:
            # 按资源类型分配
            for resource_type, amount in request.resource_requirements.items():
                # 查找对应的资源池
                pool = self._find_resource_pool(resource_type)
                if not pool:
                    raise ValueError(f"未找到资源类型 {resource_type} 的资源池")
                
                # 分配资源
                resource = pool.allocate(amount)
                if not resource:
                    raise ValueError(f"资源不足: 需要 {amount} 单位 {resource_type}")
                
                allocated[resource.id] = amount
            
            # 所有资源分配成功
            request.allocated_resources = allocated
            request.status = "allocated"
            self.allocated_resources[request.task_id] = allocated
            self.stats['active_allocations'] += 1
            self.stats['total_allocations'] += 1
            
            # 更新资源状态
            for resource_id, amount in allocated.items():
                self._update_resource_allocation(request.task_id, resource_id, amount, True)
            
            return True
            
        except Exception as e:
            # 分配失败，需要回滚已分配的资源
            for resource_id, amount in allocated.items():
                self._deallocate_from_pool(resource_id, amount)
            
            request.status = "failed"
            self.stats['failed_allocations'] += 1
            logger.error(f"资源分配失败: {e}")
            return False
    
    def _find_resource_pool(self, resource_type: ResourceType) -> Optional[ResourcePool]:
        """查找资源池"""
        for pool in self.resource_pools.values():
            if pool.resource_type == resource_type:
                return pool
        return None
    
    def _update_resource_allocation(self, task_id: str, resource_id: str, amount: float, allocated: bool):
        """更新资源分配状态"""
        for pool in self.resource_pools.values():
            if resource_id in pool.resources:
                resource = pool.resources[resource_id]
                if allocated:
                    resource.allocated_tasks.add(task_id)
                else:
                    resource.allocated_tasks.discard(task_id)
                break
    
    def _deallocate_from_pool(self, resource_id: str, amount: float):
        """从资源池释放资源"""
        for pool in self.resource_pools.values():
            if resource_id in pool.resources:
                pool.deallocate(resource_id, amount)
                break
    
    async def release_resources(self, task_id: str) -> bool:
        """释放资源"""
        with self._lock:
            if task_id not in self.allocated_resources:
                return False
            
            allocated = self.allocated_resources[task_id]
            
            # 释放所有分配的资源
            for resource_id, amount in allocated.items():
                self._deallocate_from_pool(resource_id, amount)
                self._update_resource_allocation(task_id, resource_id, amount, False)
            
            # 清理记录
            del self.allocated_resources[task_id]
            
            # 清理待处理请求
            if task_id in self.pending_requests:
                del self.pending_requests[task_id]
                self.stats['pending_requests'] -= 1
            
            self.stats['active_allocations'] -= 1
            self.stats['total_deallocations'] += 1
            
            logger.info(f"资源已释放: 任务 {task_id}")
            return True
    
    def get_resource_usage(self, task_id: str) -> Dict[str, float]:
        """获取任务资源使用情况"""
        return self.allocated_resources.get(task_id, {}).copy()
    
    def get_resource_status(self, resource_id: str) -> Optional[dict]:
        """获取资源状态"""
        for pool in self.resource_pools.values():
            if resource_id in pool.resources:
                resource = pool.resources[resource_id]
                return {
                    'id': resource.id,
                    'name': resource.name,
                    'type': resource.type.value,
                    'capacity': resource.capacity,
                    'current_usage': resource.current_usage,
                    'available_capacity': resource.available_capacity,
                    'usage_percentage': resource.usage_percentage,
                    'status': resource.status.value,
                    'allocated_tasks': list(resource.allocated_tasks)
                }
        return None
    
    def get_pool_statistics(self, pool_name: str) -> Optional[dict]:
        """获取资源池统计信息"""
        if pool_name in self.resource_pools:
            return self.resource_pools[pool_name].get_statistics()
        return None
    
    def get_all_statistics(self) -> dict:
        """获取所有统计信息"""
        pools_stats = {}
        for name, pool in self.resource_pools.items():
            pools_stats[name] = pool.get_statistics()
        
        return {
            'resource_pools': pools_stats,
            'manager_stats': self.stats.copy(),
            'total_pools': len(self.resource_pools),
            'pending_requests': len(self.pending_requests),
            'active_allocations': len(self.allocated_resources)
        }
    
    async def start_monitoring(self):
        """启动资源监控"""
        if self.monitoring_enabled and not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("资源监控已启动")
    
    async def stop_monitoring(self):
        """停止资源监控"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            logger.info("资源监控已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while True:
            try:
                await self._check_resource_health()
                await asyncio.sleep(self.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"资源监控错误: {e}")
                await asyncio.sleep(1)
    
    async def _check_resource_health(self):
        """检查资源健康状态"""
        current_time = time.time()
        
        # 检查待处理请求是否超时
        timeout_requests = []
        for task_id, request in self.pending_requests.items():
            if current_time - request.created_at > request.timeout:
                timeout_requests.append(task_id)
        
        # 处理超时请求
        for task_id in timeout_requests:
            request = self.pending_requests[task_id]
            request.status = "timeout"
            self.stats['failed_allocations'] += 1
            self.stats['pending_requests'] -= 1
            
            logger.warning(f"资源请求超时: 任务 {task_id}")
            
            # 从待处理列表中移除
            del self.pending_requests[task_id]
        
        # 更新系统资源使用情况
        await self._update_system_resources()
    
    async def _update_system_resources(self):
        """更新系统资源使用情况"""
        # 更新CPU使用率
        if 'cpu' in self.resource_pools:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_pool = self.resource_pools['cpu']
            
            # 假设所有核心平均使用
            for resource in cpu_pool.resources.values():
                resource.current_usage = cpu_percent / 100.0
                resource.available_capacity = resource.capacity - resource.current_usage
        
        # 更新内存使用率
        if 'memory' in self.resource_pools:
            memory = psutil.virtual_memory()
            memory_pool = self.resource_pools['memory']
            
            for resource in memory_pool.resources.values():
                resource.current_usage = memory.used / (1024**3)  # GB
                resource.available_capacity = resource.capacity - resource.current_usage
        
        # 更新存储使用率
        if 'storage' in self.resource_pools:
            disk = psutil.disk_usage('/')
            storage_pool = self.resource_pools['storage']
            
            for resource in storage_pool.resources.values():
                resource.current_usage = disk.used / (1024**3)  # GB
                resource.available_capacity = resource.capacity - resource.current_usage
    
    def save_configuration(self, filepath: str):
        """保存配置到文件"""
        config = {
            'resource_pools': {},
            'statistics': self.stats
        }
        
        for name, pool in self.resource_pools.items():
            config['resource_pools'][name] = {
                'name': pool.name,
                'resource_type': pool.resource_type.value,
                'resources': []
            }
            
            for resource in pool.resources.values():
                config['resource_pools'][name]['resources'].append({
                    'id': resource.id,
                    'name': resource.name,
                    'capacity': resource.capacity,
                    'metadata': resource.metadata
                })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"资源管理器配置已保存到: {filepath}")
    
    def load_configuration(self, filepath: str):
        """从文件加载配置"""
        if not os.path.exists(filepath):
            logger.warning(f"配置文件不存在: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 清理现有配置
        self.resource_pools.clear()
        
        # 加载配置
        for pool_name, pool_config in config['resource_pools'].items():
            resource_type = ResourceType(pool_config['resource_type'])
            pool = ResourcePool(pool_config['name'], resource_type)
            
            for resource_config in pool_config['resources']:
                resource = Resource(
                    id=resource_config['id'],
                    name=resource_config['name'],
                    type=resource_type,
                    capacity=resource_config['capacity'],
                    metadata=resource_config.get('metadata', {})
                )
                pool.add_resource(resource)
            
            self.resource_pools[pool_name] = pool
        
        logger.info(f"资源管理器配置已从 {filepath} 加载")


# 辅助函数
def create_cpu_resource(core_id: int, capacity: float = 1.0) -> Resource:
    """创建CPU资源"""
    return Resource(
        id=f"cpu_{core_id}",
        name=f"CPU Core {core_id}",
        type=ResourceType.CPU,
        capacity=capacity,
        metadata={'core_id': core_id}
    )


def create_memory_resource(name: str, capacity_gb: float) -> Resource:
    """创建内存资源"""
    return Resource(
        id=f"memory_{name}",
        name=name,
        type=ResourceType.MEMORY,
        capacity=capacity_gb,
        metadata={'unit': 'GB'}
    )


def create_gpu_resource(gpu_id: int, memory_gb: float) -> Resource:
    """创建GPU资源"""
    return Resource(
        id=f"gpu_{gpu_id}",
        name=f"GPU {gpu_id}",
        type=ResourceType.GPU,
        capacity=memory_gb,
        metadata={'memory_gb': memory_gb}
    )


# 示例用法
async def example_usage():
    """示例用法"""
    # 创建资源管理器
    manager = ResourceManager()
    
    # 创建自定义资源池
    gpu_pool = manager.create_resource_pool("GPU Pool", ResourceType.GPU)
    
    # 添加GPU资源
    gpu1 = create_gpu_resource(0, 8.0)
    gpu2 = create_gpu_resource(1, 8.0)
    gpu_pool.add_resource(gpu1)
    gpu_pool.add_resource(gpu2)
    
    # 创建资源请求
    request = ResourceRequest(
        task_id="task_001",
        resource_requirements={
            ResourceType.CPU: 2.0,
            ResourceType.MEMORY: 4.0,
            ResourceType.GPU: 1.0
        },
        priority=5
    )
    
    # 请求资源
    success = await manager.request_resources(request)
    if success:
        print("资源分配成功")
        print("分配的资源:", request.allocated_resources)
    
    # 启动监控
    await manager.start_monitoring()
    
    # 等待一段时间
    await asyncio.sleep(10)
    
    # 释放资源
    await manager.release_resources("task_001")
    
    # 获取统计信息
    stats = manager.get_all_statistics()
    print("资源管理器统计:", stats)
    
    # 停止监控
    await manager.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(example_usage())