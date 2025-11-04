"""
资源管理器

提供资源分配、释放、监控和优化功能
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import psutil
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    DATABASE = "database"
    CACHE = "cache"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"


class AllocationStrategy(Enum):
    """分配策略"""
    FAIR = "fair"
    PRIORITY = "priority"
    FIRST_FIT = "first_fit"
    BEST_FIT = "best_fit"
    ROUND_ROBIN = "round_robin"
    LOAD_BASED = "load_based"


@dataclass
class Resource:
    """资源定义"""
    resource_id: str
    resource_type: ResourceType
    name: str
    capacity: float
    current_usage: float = 0.0
    available: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self.available = self.capacity - self.current_usage
    
    @property
    def utilization_rate(self) -> float:
        """资源利用率"""
        return (self.current_usage / self.capacity) * 100 if self.capacity > 0 else 0.0
    
    @property
    def is_available(self) -> bool:
        """是否可用"""
        return self.available > 0


@dataclass
class ResourceAllocation:
    """资源分配"""
    allocation_id: str
    resource_id: str
    consumer_id: str
    allocated_amount: float
    allocated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """是否过期"""
        return self.expires_at and time.time() > self.expires_at


@dataclass
class ResourceRequest:
    """资源请求"""
    request_id: str
    consumer_id: str
    resource_type: ResourceType
    amount: float
    priority: int = 0
    duration: Optional[float] = None  # 预计使用时长
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class ResourcePool:
    """资源池"""
    pool_id: str
    resource_type: ResourceType
    resources: Dict[str, Resource] = field(default_factory=dict)
    allocation_strategy: AllocationStrategy = AllocationStrategy.FAIR
    max_allocation_size: float = 1000.0
    min_allocation_size: float = 1.0
    auto_scaling: bool = False
    scaling_threshold: float = 80.0  # 扩容阈值（利用率%）


class ResourceManager:
    """资源管理器"""
    
    def __init__(self):
        # 资源池管理
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.default_pools: Dict[ResourceType, str] = {}
        
        # 分配管理
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.consumer_allocations: Dict[str, List[str]] = defaultdict(list)
        
        # 请求队列
        self.pending_requests: List[ResourceRequest] = []
        
        # 统计信息
        self.stats = {
            'total_resources': 0,
            'total_allocations': 0,
            'active_allocations': 0,
            'failed_allocations': 0,
            'average_allocation_time': 0.0,
            'resource_efficiency': 0.0
        }
        
        # 监控和清理
        self._monitoring_active = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        
        # 初始化系统资源
        self._initialize_system_resources()
    
    def _initialize_system_resources(self):
        """初始化系统资源"""
        try:
            # CPU资源
            cpu_count = psutil.cpu_count()
            self.create_resource_pool(
                pool_id="cpu_pool",
                resource_type=ResourceType.CPU,
                resources=[Resource(
                    resource_id="cpu_main",
                    resource_type=ResourceType.CPU,
                    name="主CPU",
                    capacity=cpu_count
                )]
            )
            
            # 内存资源
            memory_info = psutil.virtual_memory()
            self.create_resource_pool(
                pool_id="memory_pool",
                resource_type=ResourceType.MEMORY,
                resources=[Resource(
                    resource_id="memory_main",
                    resource_type=ResourceType.MEMORY,
                    name="主内存",
                    capacity=memory_info.total / (1024**3)  # 转换为GB
                )]
            )
            
            # 磁盘资源
            disk_info = psutil.disk_usage('/')
            self.create_resource_pool(
                pool_id="disk_pool",
                resource_type=ResourceType.DISK,
                resources=[Resource(
                    resource_id="disk_main",
                    resource_type=ResourceType.DISK,
                    name="主磁盘",
                    capacity=disk_info.total / (1024**3)  # 转换为GB
                )]
            )
            
            logger.info("系统资源初始化完成")
            
        except Exception as e:
            logger.error(f"系统资源初始化失败: {e}")
    
    def create_resource_pool(self, pool_id: str, resource_type: ResourceType,
                           resources: List[Resource],
                           allocation_strategy: AllocationStrategy = AllocationStrategy.FAIR,
                           auto_scaling: bool = False) -> bool:
        """创建资源池"""
        try:
            with self._lock:
                pool = ResourcePool(
                    pool_id=pool_id,
                    resource_type=resource_type,
                    resources={r.resource_id: r for r in resources},
                    allocation_strategy=allocation_strategy,
                    auto_scaling=auto_scaling
                )
                
                self.resource_pools[pool_id] = pool
                self.default_pools[resource_type] = pool_id
                
                self.stats['total_resources'] += len(resources)
                
                logger.info(f"资源池 {pool_id} 创建成功，资源类型: {resource_type.value}, 资源数: {len(resources)}")
                return True
                
        except Exception as e:
            logger.error(f"创建资源池失败: {e}")
            return False
    
    def add_resource_to_pool(self, pool_id: str, resource: Resource) -> bool:
        """向资源池添加资源"""
        try:
            with self._lock:
                if pool_id not in self.resource_pools:
                    return False
                
                pool = self.resource_pools[pool_id]
                pool.resources[resource.resource_id] = resource
                self.stats['total_resources'] += 1
                
                logger.info(f"资源 {resource.resource_id} 已添加到资源池 {pool_id}")
                return True
                
        except Exception as e:
            logger.error(f"添加资源失败: {e}")
            return False
    
    async def request_resource(self, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """请求资源"""
        try:
            with self._lock:
                # 查找合适的资源池
                pool_id = self.default_pools.get(request.resource_type)
                if not pool_id or pool_id not in self.resource_pools:
                    logger.error(f"未找到资源类型 {request.resource_type.value} 对应的资源池")
                    return None
                
                pool = self.resource_pools[pool_id]
                
                # 选择资源
                selected_resource = await self._select_resource(pool, request)
                if not selected_resource:
                    logger.warning(f"无法为请求 {request.request_id} 分配资源")
                    return None
                
                # 分配资源
                allocation = await self._allocate_resource(selected_resource, request)
                
                if allocation:
                    self.allocations[allocation.allocation_id] = allocation
                    self.consumer_allocations[request.consumer_id].append(allocation.allocation_id)
                    
                    self.stats['total_allocations'] += 1
                    self.stats['active_allocations'] += 1
                    
                    logger.info(f"资源分配成功: {allocation.allocation_id}")
                    return allocation
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"请求资源失败: {e}")
            return None
    
    async def _select_resource(self, pool: ResourcePool, request: ResourceRequest) -> Optional[Resource]:
        """选择资源"""
        try:
            available_resources = [
                resource for resource in pool.resources.values()
                if resource.available >= request.amount
            ]
            
            if not available_resources:
                return None
            
            # 根据分配策略选择资源
            if pool.allocation_strategy == AllocationStrategy.FAIR:
                # 公平分配：选择利用率最低的资源
                return min(available_resources, key=lambda r: r.utilization_rate)
            
            elif pool.allocation_strategy == AllocationStrategy.PRIORITY:
                # 优先级分配：优先选择容量大的资源
                return max(available_resources, key=lambda r: r.capacity)
            
            elif pool.allocation_strategy == AllocationStrategy.FIRST_FIT:
                # 首次适应：选择第一个满足条件的资源
                return available_resources[0]
            
            elif pool.allocation_strategy == AllocationStrategy.BEST_FIT:
                # 最佳适应：选择最接近需求容量的资源
                return min(available_resources, key=lambda r: r.capacity - request.amount)
            
            elif pool.allocation_strategy == AllocationStrategy.ROUND_ROBIN:
                # 轮询分配（简化实现）
                return available_resources[hash(request.request_id) % len(available_resources)]
            
            elif pool.allocation_strategy == AllocationStrategy.LOAD_BASED:
                # 基于负载分配
                return min(available_resources, key=lambda r: r.current_usage)
            
            else:
                return available_resources[0]
                
        except Exception as e:
            logger.error(f"选择资源失败: {e}")
            return None
    
    async def _allocate_resource(self, resource: Resource, request: ResourceRequest) -> Optional[ResourceAllocation]:
        """分配资源"""
        try:
            if resource.available < request.amount:
                return None
            
            # 创建分配记录
            allocation = ResourceAllocation(
                allocation_id=f"alloc_{int(time.time() * 1000)}_{request.consumer_id}",
                resource_id=resource.resource_id,
                consumer_id=request.consumer_id,
                allocated_amount=request.amount,
                expires_at=time.time() + request.duration if request.duration else None,
                metadata=request.metadata.copy()
            )
            
            # 更新资源状态
            resource.current_usage += request.amount
            resource.available = resource.capacity - resource.current_usage
            
            # 检查是否需要扩容
            if resource.utilization_rate > 80 and request.priority > 5:
                await self._trigger_scaling(resource.resource_type)
            
            return allocation
            
        except Exception as e:
            logger.error(f"分配资源失败: {e}")
            return None
    
    async def release_resource(self, allocation_id: str) -> bool:
        """释放资源"""
        try:
            with self._lock:
                if allocation_id not in self.allocations:
                    return False
                
                allocation = self.allocations[allocation_id]
                
                # 查找资源
                resource = None
                for pool in self.resource_pools.values():
                    if allocation.resource_id in pool.resources:
                        resource = pool.resources[allocation.resource_id]
                        break
                
                if resource:
                    # 释放资源
                    resource.current_usage = max(0, resource.current_usage - allocation.allocated_amount)
                    resource.available = resource.capacity - resource.current_usage
                    
                    # 清理分配记录
                    del self.allocations[allocation_id]
                    
                    # 从消费者分配列表中移除
                    consumer_allocs = self.consumer_allocations.get(allocation.consumer_id, [])
                    if allocation_id in consumer_allocs:
                        consumer_allocs.remove(allocation_id)
                    
                    self.stats['active_allocations'] -= 1
                    
                    logger.info(f"资源释放成功: {allocation_id}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"释放资源失败: {e}")
            return False
    
    async def release_consumer_resources(self, consumer_id: str) -> List[str]:
        """释放消费者所有资源"""
        try:
            released_allocations = []
            
            if consumer_id in self.consumer_allocations:
                allocation_ids = self.consumer_allocations[consumer_id].copy()
                
                for allocation_id in allocation_ids:
                    if await self.release_resource(allocation_id):
                        released_allocations.append(allocation_id)
                
                del self.consumer_allocations[consumer_id]
            
            logger.info(f"消费者 {consumer_id} 的 {len(released_allocations)} 个资源分配已释放")
            return released_allocations
            
        except Exception as e:
            logger.error(f"释放消费者资源失败: {e}")
            return []
    
    async def _trigger_scaling(self, resource_type: ResourceType):
        """触发资源扩容"""
        try:
            pool_id = self.default_pools.get(resource_type)
            if not pool_id or pool_id not in self.resource_pools:
                return
            
            pool = self.resource_pools[pool_id]
            if not pool.auto_scaling:
                return
            
            logger.info(f"资源类型 {resource_type.value} 触发自动扩容")
            
            # 这里可以添加实际的扩容逻辑
            # 例如：创建新的虚拟机、添加物理资源等
            
        except Exception as e:
            logger.error(f"触发资源扩容失败: {e}")
    
    async def start_monitoring(self, interval: float = 30.0):
        """启动资源监控"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._cleanup_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info("资源监控已启动")
    
    async def stop_monitoring(self):
        """停止资源监控"""
        self._monitoring_active = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("资源监控已停止")
    
    async def _monitoring_loop(self, interval: float):
        """监控循环"""
        while self._monitoring_active:
            try:
                await asyncio.sleep(interval)
                
                # 更新系统资源使用情况
                await self._update_system_resources()
                
                # 清理过期分配
                await self._cleanup_expired_allocations()
                
                # 更新统计信息
                await self._update_stats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"资源监控异常: {e}")
    
    async def _update_system_resources(self):
        """更新系统资源使用情况"""
        try:
            # 更新CPU使用情况
            cpu_usage = psutil.cpu_percent(interval=1)
            if "cpu_pool" in self.resource_pools:
                pool = self.resource_pools["cpu_pool"]
                for resource in pool.resources.values():
                    resource.current_usage = (cpu_usage / 100) * resource.capacity
                    resource.available = resource.capacity - resource.current_usage
            
            # 更新内存使用情况
            memory_info = psutil.virtual_memory()
            if "memory_pool" in self.resource_pools:
                pool = self.resource_pools["memory_pool"]
                for resource in pool.resources.values():
                    resource.current_usage = memory_info.used / (1024**3)  # 转换为GB
                    resource.available = resource.capacity - resource.current_usage
            
            # 更新磁盘使用情况
            disk_info = psutil.disk_usage('/')
            if "disk_pool" in self.resource_pools:
                pool = self.resource_pools["disk_pool"]
                for resource in pool.resources.values():
                    resource.current_usage = disk_info.used / (1024**3)  # 转换为GB
                    resource.available = resource.capacity - resource.current_usage
                    
        except Exception as e:
            logger.error(f"更新系统资源失败: {e}")
    
    async def _cleanup_expired_allocations(self):
        """清理过期分配"""
        try:
            current_time = time.time()
            expired_allocations = [
                alloc_id for alloc_id, allocation in self.allocations.items()
                if allocation.is_expired()
            ]
            
            for allocation_id in expired_allocations:
                await self.release_resource(allocation_id)
            
            if expired_allocations:
                logger.info(f"清理了 {len(expired_allocations)} 个过期分配")
                
        except Exception as e:
            logger.error(f"清理过期分配失败: {e}")
    
    async def _update_stats(self):
        """更新统计信息"""
        try:
            total_capacity = 0
            total_usage = 0
            
            for pool in self.resource_pools.values():
                for resource in pool.resources.values():
                    total_capacity += resource.capacity
                    total_usage += resource.current_usage
            
            if total_capacity > 0:
                self.stats['resource_efficiency'] = (total_usage / total_capacity) * 100
            
        except Exception as e:
            logger.error(f"更新统计失败: {e}")
    
    def get_resource_status(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """获取资源状态"""
        try:
            for pool in self.resource_pools.values():
                if resource_id in pool.resources:
                    resource = pool.resources[resource_id]
                    return {
                        'resource_id': resource_id,
                        'resource_type': resource.resource_type.value,
                        'capacity': resource.capacity,
                        'current_usage': resource.current_usage,
                        'available': resource.available,
                        'utilization_rate': resource.utilization_rate,
                        'is_available': resource.is_available
                    }
            return None
        except Exception as e:
            logger.error(f"获取资源状态失败: {e}")
            return None
    
    def get_pool_status(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """获取资源池状态"""
        try:
            if pool_id not in self.resource_pools:
                return None
            
            pool = self.resource_pools[pool_id]
            resources_status = []
            
            for resource in pool.resources.values():
                resources_status.append({
                    'resource_id': resource.resource_id,
                    'name': resource.name,
                    'capacity': resource.capacity,
                    'current_usage': resource.current_usage,
                    'available': resource.available,
                    'utilization_rate': resource.utilization_rate
                })
            
            return {
                'pool_id': pool_id,
                'resource_type': pool.resource_type.value,
                'allocation_strategy': pool.allocation_strategy.value,
                'auto_scaling': pool.auto_scaling,
                'total_resources': len(pool.resources),
                'resources': resources_status
            }
        except Exception as e:
            logger.error(f"获取资源池状态失败: {e}")
            return None
    
    def get_consumer_allocations(self, consumer_id: str) -> List[Dict[str, Any]]:
        """获取消费者资源分配"""
        try:
            allocations = []
            allocation_ids = self.consumer_allocations.get(consumer_id, [])
            
            for alloc_id in allocation_ids:
                if alloc_id in self.allocations:
                    allocation = self.allocations[alloc_id]
                    resource_status = self.get_resource_status(allocation.resource_id)
                    
                    allocations.append({
                        'allocation_id': alloc_id,
                        'resource_id': allocation.resource_id,
                        'allocated_amount': allocation.allocated_amount,
                        'allocated_at': allocation.allocated_at,
                        'expires_at': allocation.expires_at,
                        'is_expired': allocation.is_expired(),
                        'resource_status': resource_status
                    })
            
            return allocations
        except Exception as e:
            logger.error(f"获取消费者分配失败: {e}")
            return []
    
    def get_resource_manager_stats(self) -> Dict[str, Any]:
        """获取资源管理器统计"""
        return {
            'total_pools': len(self.resource_pools),
            'total_resources': self.stats['total_resources'],
            'total_allocations': self.stats['total_allocations'],
            'active_allocations': self.stats['active_allocations'],
            'failed_allocations': self.stats['failed_allocations'],
            'resource_efficiency': self.stats['resource_efficiency'],
            'pools': {
                pool_id: {
                    'resource_type': pool.resource_type.value,
                    'total_resources': len(pool.resources),
                    'allocation_strategy': pool.allocation_strategy.value
                }
                for pool_id, pool in self.resource_pools.items()
            }
        }