"""
负载均衡器

提供多种负载均衡算法和健康检查功能
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import random
import hashlib
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"
    RESOURCE_BASED = "resource_based"


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


@dataclass
class BackendServer:
    """后端服务器"""
    server_id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    response_time: float = 0.0
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_health_check: float = 0.0
    failure_count: int = 0
    success_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        return self.health_status == HealthStatus.HEALTHY
    
    @property
    def is_available(self) -> bool:
        return self.is_healthy and self.current_connections < self.max_connections
    
    @property
    def utilization(self) -> float:
        return (self.current_connections / self.max_connections) * 100 if self.max_connections > 0 else 0.0


@dataclass
class LoadBalanceRequest:
    """负载均衡请求"""
    request_id: str
    client_ip: str
    request_data: Any
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class LoadBalanceResult:
    """负载均衡结果"""
    request_id: str
    selected_server: Optional[BackendServer]
    strategy_used: LoadBalanceStrategy
    selection_time: float
    reasoning: str


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, name: str = "load_balancer"):
        self.name = name
        
        # 服务器管理
        self.servers: Dict[str, BackendServer] = {}
        self.server_groups: Dict[str, List[str]] = defaultdict(list)
        
        # 负载均衡策略
        self.strategies = {
            LoadBalanceStrategy.ROUND_ROBIN: self._round_robin_strategy,
            LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN: self._weighted_round_robin_strategy,
            LoadBalanceStrategy.LEAST_CONNECTIONS: self._least_connections_strategy,
            LoadBalanceStrategy.LEAST_RESPONSE_TIME: self._least_response_time_strategy,
            LoadBalanceStrategy.IP_HASH: self._ip_hash_strategy,
            LoadBalanceStrategy.RANDOM: self._random_strategy,
            LoadBalanceStrategy.CONSISTENT_HASH: self._consistent_hash_strategy,
            LoadBalanceStrategy.RESOURCE_BASED: self._resource_based_strategy
        }
        
        self.current_strategy = LoadBalanceStrategy.ROUND_ROBIN
        self.round_robin_index = 0
        self.weighted_round_robin_index = 0
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_selections': 0,
            'failed_selections': 0,
            'average_selection_time': 0.0,
            'server_utilization': {},
            'request_distribution': {}
        }
        
        # 健康检查
        self.health_check_interval = 30.0
        self.health_check_timeout = 5.0
        self.health_check_url = ""
        self.health_check_enabled = True
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        # 监控和回调
        self.selection_callbacks: List[Callable] = []
        self.health_callbacks: List[Callable] = []
    
    async def start(self):
        """启动负载均衡器"""
        self._running = True
        
        if self.health_check_enabled:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"负载均衡器 {self.name} 已启动")
    
    async def stop(self):
        """停止负载均衡器"""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"负载均衡器 {self.name} 已停止")
    
    def add_server(self, server: BackendServer, group: str = "default") -> bool:
        """添加服务器"""
        try:
            self.servers[server.server_id] = server
            self.server_groups[group].append(server.server_id)
            
            logger.info(f"服务器 {server.server_id} 已添加到负载均衡器 {self.name}")
            return True
            
        except Exception as e:
            logger.error(f"添加服务器失败: {e}")
            return False
    
    def remove_server(self, server_id: str) -> bool:
        """移除服务器"""
        try:
            if server_id in self.servers:
                del self.servers[server_id]
                
                # 从所有组中移除
                for group_servers in self.server_groups.values():
                    if server_id in group_servers:
                        group_servers.remove(server_id)
                
                logger.info(f"服务器 {server_id} 已从负载均衡器 {self.name} 移除")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"移除服务器失败: {e}")
            return False
    
    async def select_server(self, request: LoadBalanceRequest, 
                          group: str = "default") -> LoadBalanceResult:
        """选择服务器"""
        start_time = time.time()
        
        try:
            self.stats['total_requests'] += 1
            
            # 获取组内的健康服务器
            available_servers = await self._get_available_servers(group)
            
            if not available_servers:
                result = LoadBalanceResult(
                    request_id=request.request_id,
                    selected_server=None,
                    strategy_used=self.current_strategy,
                    selection_time=time.time() - start_time,
                    reasoning="没有可用的服务器"
                )
                
                self.stats['failed_selections'] += 1
                return result
            
            # 使用选择的策略选择服务器
            strategy_func = self.strategies.get(self.current_strategy, self._round_robin_strategy)
            selected_server = await strategy_func(available_servers, request)
            
            selection_time = time.time() - start_time
            
            # 更新统计
            self.stats['successful_selections'] += 1
            self.stats['average_selection_time'] = (
                (self.stats['average_selection_time'] * 
                 (self.stats['successful_selections'] - 1) + selection_time) /
                self.stats['successful_selections']
            )
            
            # 更新服务器连接数
            if selected_server:
                selected_server.current_connections += 1
                
                # 更新请求分布统计
                if selected_server.server_id not in self.stats['request_distribution']:
                    self.stats['request_distribution'][selected_server.server_id] = 0
                self.stats['request_distribution'][selected_server.server_id] += 1
            
            result = LoadBalanceResult(
                request_id=request.request_id,
                selected_server=selected_server,
                strategy_used=self.current_strategy,
                selection_time=selection_time,
                reasoning=f"使用{self.current_strategy.value}策略选择"
            )
            
            # 触发回调
            await self._trigger_selection_callbacks(result)
            
            return result
            
        except Exception as e:
            logger.error(f"选择服务器失败: {e}")
            self.stats['failed_selections'] += 1
            
            return LoadBalanceResult(
                request_id=request.request_id,
                selected_server=None,
                strategy_used=self.current_strategy,
                selection_time=time.time() - start_time,
                reasoning=f"选择失败: {str(e)}"
            )
    
    async def _get_available_servers(self, group: str) -> List[BackendServer]:
        """获取可用的服务器"""
        try:
            group_server_ids = self.server_groups.get(group, [])
            available_servers = []
            
            for server_id in group_server_ids:
                if server_id in self.servers:
                    server = self.servers[server_id]
                    if server.is_available:
                        available_servers.append(server)
            
            return available_servers
            
        except Exception as e:
            logger.error(f"获取可用服务器失败: {e}")
            return []
    
    async def _round_robin_strategy(self, servers: List[BackendServer], 
                                  request: LoadBalanceRequest) -> Optional[BackendServer]:
        """轮询策略"""
        if not servers:
            return None
        
        server = servers[self.round_robin_index % len(servers)]
        self.round_robin_index += 1
        return server
    
    async def _weighted_round_robin_strategy(self, servers: List[BackendServer], 
                                           request: LoadBalanceRequest) -> Optional[BackendServer]:
        """加权轮询策略"""
        if not servers:
            return None
        
        # 计算总权重
        total_weight = sum(server.weight for server in servers)
        
        if total_weight <= 0:
            return servers[0]
        
        # 生成随机数
        random_weight = random.uniform(0, total_weight)
        current_weight = 0
        
        for server in servers:
            current_weight += server.weight
            if random_weight <= current_weight:
                return server
        
        return servers[-1]
    
    async def _least_connections_strategy(self, servers: List[BackendServer], 
                                        request: LoadBalanceRequest) -> Optional[BackendServer]:
        """最少连接策略"""
        if not servers:
            return None
        
        return min(servers, key=lambda s: s.current_connections)
    
    async def _least_response_time_strategy(self, servers: List[BackendServer], 
                                          request: LoadBalanceRequest) -> Optional[BackendServer]:
        """最少响应时间策略"""
        if not servers:
            return None
        
        return min(servers, key=lambda s: s.response_time)
    
    async def _ip_hash_strategy(self, servers: List[BackendServer], 
                              request: LoadBalanceRequest) -> Optional[BackendServer]:
        """IP哈希策略"""
        if not servers:
            return None
        
        # 计算客户端IP的哈希值
        client_ip = request.client_ip
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        
        # 选择服务器
        server_index = hash_value % len(servers)
        return servers[server_index]
    
    async def _random_strategy(self, servers: List[BackendServer], 
                             request: LoadBalanceRequest) -> Optional[BackendServer]:
        """随机策略"""
        if not servers:
            return None
        
        return random.choice(servers)
    
    async def _consistent_hash_strategy(self, servers: List[BackendServer], 
                                      request: LoadBalanceRequest) -> Optional[BackendServer]:
        """一致性哈希策略"""
        if not servers:
            return None
        
        # 使用请求ID作为哈希键
        hash_key = request.request_id
        hash_value = int(hashlib.md5(hash_key.encode()).hexdigest(), 16)
        
        # 选择服务器
        server_index = hash_value % len(servers)
        return servers[server_index]
    
    async def _resource_based_strategy(self, servers: List[BackendServer], 
                                     request: LoadBalanceRequest) -> Optional[BackendServer]:
        """基于资源的策略"""
        if not servers:
            return None
        
        # 综合考虑多个因素
        scored_servers = []
        
        for server in servers:
            # 计算综合分数
            connection_score = 1.0 - (server.current_connections / server.max_connections)
            response_score = 1.0 / (1.0 + server.response_time)
            health_score = 1.0 if server.is_healthy else 0.0
            
            total_score = connection_score * 0.4 + response_score * 0.3 + health_score * 0.3
            scored_servers.append((server, total_score))
        
        # 选择分数最高的服务器
        best_server, _ = max(scored_servers, key=lambda x: x[1])
        return best_server
    
    async def release_server_connection(self, server_id: str):
        """释放服务器连接"""
        try:
            if server_id in self.servers:
                server = self.servers[server_id]
                server.current_connections = max(0, server.current_connections - 1)
                logger.debug(f"服务器 {server_id} 连接数减少，当前: {server.current_connections}")
        except Exception as e:
            logger.error(f"释放服务器连接失败: {e}")
    
    async def update_server_health(self, server_id: str, is_healthy: bool, 
                                 response_time: float = 0.0):
        """更新服务器健康状态"""
        try:
            if server_id in self.servers:
                server = self.servers[server_id]
                
                old_status = server.health_status
                server.health_status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
                server.last_health_check = time.time()
                server.response_time = response_time
                
                if is_healthy:
                    server.success_count += 1
                    server.failure_count = 0
                else:
                    server.failure_count += 1
                
                # 触发健康检查回调
                if old_status != server.health_status:
                    await self._trigger_health_callbacks(server, is_healthy)
                
                logger.debug(f"服务器 {server_id} 健康状态更新: {old_status.value} -> {server.health_status.value}")
                
        except Exception as e:
            logger.error(f"更新服务器健康状态失败: {e}")
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # 对所有服务器进行健康检查
                for server_id, server in self.servers.items():
                    try:
                        await self._perform_health_check(server)
                    except Exception as e:
                        logger.error(f"服务器 {server_id} 健康检查失败: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查循环异常: {e}")
    
    async def _perform_health_check(self, server: BackendServer):
        """执行健康检查"""
        try:
            import aiohttp
            
            health_url = f"http://{server.host}:{server.port}/health"
            
            timeout = aiohttp.ClientTimeout(total=self.health_check_timeout)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                start_time = time.time()
                async with session.get(health_url) as response:
                    response_time = time.time() - start_time
                    
                    is_healthy = response.status == 200
                    await self.update_server_health(server.server_id, is_healthy, response_time)
                    
        except Exception as e:
            logger.debug(f"健康检查失败: {server.server_id}, 错误: {e}")
            await self.update_server_health(server.server_id, False)
    
    async def _trigger_selection_callbacks(self, result: LoadBalanceResult):
        """触发选择回调"""
        try:
            for callback in self.selection_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
        except Exception as e:
            logger.error(f"触发选择回调失败: {e}")
    
    async def _trigger_health_callbacks(self, server: BackendServer, is_healthy: bool):
        """触发健康检查回调"""
        try:
            for callback in self.health_callbacks:
                if asyncio.iscoroutinefunction(callback):
                    await callback(server, is_healthy)
                else:
                    callback(server, is_healthy)
        except Exception as e:
            logger.error(f"触发健康检查回调失败: {e}")
    
    def set_load_balance_strategy(self, strategy: LoadBalanceStrategy):
        """设置负载均衡策略"""
        self.current_strategy = strategy
        logger.info(f"负载均衡策略已设置为: {strategy.value}")
    
    def add_selection_callback(self, callback: Callable):
        """添加选择回调"""
        self.selection_callbacks.append(callback)
    
    def add_health_callback(self, callback: Callable):
        """添加健康检查回调"""
        self.health_callbacks.append(callback)
    
    def get_server_stats(self, server_id: str) -> Optional[Dict[str, Any]]:
        """获取服务器统计"""
        try:
            if server_id not in self.servers:
                return None
            
            server = self.servers[server_id]
            return {
                'server_id': server_id,
                'host': server.host,
                'port': server.port,
                'weight': server.weight,
                'max_connections': server.max_connections,
                'current_connections': server.current_connections,
                'utilization': server.utilization,
                'response_time': server.response_time,
                'health_status': server.health_status.value,
                'failure_count': server.failure_count,
                'success_count': server.success_count,
                'is_available': server.is_available
            }
        except Exception as e:
            logger.error(f"获取服务器统计失败: {e}")
            return None
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """获取负载均衡器统计"""
        try:
            # 计算服务器利用率统计
            server_utilizations = {}
            for server_id, server in self.servers.items():
                server_utilizations[server_id] = {
                    'utilization': server.utilization,
                    'response_time': server.response_time,
                    'health_status': server.health_status.value
                }
            
            # 计算平均响应时间
            response_times = [s.response_time for s in self.servers.values() if s.response_time > 0]
            avg_response_time = statistics.mean(response_times) if response_times else 0.0
            
            # 计算健康服务器比例
            healthy_servers = len([s for s in self.servers.values() if s.is_healthy])
            total_servers = len(self.servers)
            health_ratio = (healthy_servers / total_servers * 100) if total_servers > 0 else 0
            
            return {
                'name': self.name,
                'current_strategy': self.current_strategy.value,
                'total_servers': total_servers,
                'healthy_servers': healthy_servers,
                'health_ratio': health_ratio,
                'total_requests': self.stats['total_requests'],
                'successful_selections': self.stats['successful_selections'],
                'failed_selections': self.stats['failed_selections'],
                'average_selection_time': self.stats['average_selection_time'],
                'average_response_time': avg_response_time,
                'server_stats': server_utilizations,
                'request_distribution': self.stats['request_distribution'].copy()
            }
        except Exception as e:
            logger.error(f"获取负载均衡器统计失败: {e}")
            return {}