"""
负载均衡系统
支持多种负载均衡算法和健康检查
"""

import asyncio
import time
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from ..utils import measure_time

logger = logging.getLogger(__name__)

class LoadBalanceAlgorithm(Enum):
    """负载均衡算法枚举"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    RANDOM = "random"
    CONSISTENT_HASH = "consistent_hash"

@dataclass
class ServerNode:
    """服务器节点"""
    id: str
    host: str
    port: int
    weight: int = 1
    current_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_health_check: float = 0.0
    is_healthy: bool = True
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def error_rate(self) -> float:
        """错误率"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def load_score(self) -> float:
        """负载得分"""
        # 综合考虑连接数、响应时间和错误率
        connection_score = self.current_connections / max(self.weight, 1)
        response_score = self.avg_response_time
        error_score = self.error_rate * 100
        
        return connection_score + response_score + error_score

class LoadBalancer:
    """智能负载均衡器"""
    
    def __init__(self, algorithm: LoadBalanceAlgorithm = LoadBalanceAlgorithm.WEIGHTED_ROUND_ROBIN):
        self.algorithm = algorithm
        self.nodes: Dict[str, ServerNode] = {}
        self.round_robin_index = 0
        self.health_check_interval = 30
        self.failure_threshold = 3
        self.recovery_threshold = 2
        self.health_check_task: Optional[asyncio.Task] = None
        self._start_health_checker()
    
    def _start_health_checker(self):
        """启动健康检查器"""
        if self.health_check_task is None or self.health_check_task.done():
            self.health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"健康检查循环错误: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_check(self):
        """执行健康检查"""
        current_time = time.time()
        
        for node_id, node in self.nodes.items():
            try:
                # 简单的健康检查（实际实现中应该发送实际的健康检查请求）
                is_healthy = await self._check_node_health(node)
                
                if is_healthy:
                    node.consecutive_failures = 0
                    node.is_healthy = True
                else:
                    node.consecutive_failures += 1
                    node.last_failure_time = current_time
                    
                    if node.consecutive_failures >= self.failure_threshold:
                        node.is_healthy = False
                
                node.last_health_check = current_time
                
            except Exception as e:
                logger.error(f"节点 {node_id} 健康检查失败: {e}")
                node.consecutive_failures += 1
                node.is_healthy = False
    
    async def _check_node_health(self, node: ServerNode) -> bool:
        """检查单个节点健康状态"""
        # 这里应该实现实际的健康检查逻辑
        # 例如：发送HTTP请求到节点的健康检查端点
        
        # 模拟健康检查
        await asyncio.sleep(0.1)  # 模拟网络延迟
        
        # 根据错误率和响应时间判断健康状态
        if node.error_rate > 0.5:  # 错误率超过50%
            return False
        
        if node.avg_response_time > 10.0:  # 平均响应时间超过10秒
            return False
        
        # 连续失败次数过多
        if node.consecutive_failures >= self.failure_threshold:
            return False
        
        return True
    
    def add_node(self, node: ServerNode):
        """添加节点"""
        self.nodes[node.id] = node
        logger.info(f"添加节点: {node.id} ({node.host}:{node.port})")
    
    def remove_node(self, node_id: str):
        """移除节点"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"移除节点: {node_id}")
    
    def update_node(self, node_id: str, **kwargs):
        """更新节点信息"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            for key, value in kwargs.items():
                if hasattr(node, key):
                    setattr(node, key, value)
    
    @measure_time
    async def select_node(
        self,
        client_ip: Optional[str] = None,
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[ServerNode]:
        """
        选择节点
        
        Args:
            client_ip: 客户端IP（用于IP哈希）
            request_context: 请求上下文
            
        Returns:
            选中的节点，如果没有可用节点则返回None
        """
        available_nodes = self._get_available_nodes()
        
        if not available_nodes:
            logger.warning("没有可用的节点")
            return None
        
        if len(available_nodes) == 1:
            return available_nodes[0]
        
        # 根据算法选择节点
        if self.algorithm == LoadBalanceAlgorithm.ROUND_ROBIN:
            return self._round_robin_select(available_nodes)
        elif self.algorithm == LoadBalanceAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(available_nodes)
        elif self.algorithm == LoadBalanceAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections_select(available_nodes)
        elif self.algorithm == LoadBalanceAlgorithm.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(available_nodes)
        elif self.algorithm == LoadBalanceAlgorithm.IP_HASH:
            return self._ip_hash_select(available_nodes, client_ip)
        elif self.algorithm == LoadBalanceAlgorithm.RANDOM:
            return self._random_select(available_nodes)
        else:  # CONSISTENT_HASH
            return self._consistent_hash_select(available_nodes, request_context)
    
    def _get_available_nodes(self) -> List[ServerNode]:
        """获取可用节点列表"""
        current_time = time.time()
        available = []
        
        for node in self.nodes.values():
            # 检查节点是否健康
            if not node.is_healthy:
                # 尝试恢复健康节点
                if (current_time - node.last_failure_time > 60 and  # 1分钟后尝试恢复
                    node.consecutive_failures < self.recovery_threshold):
                    node.is_healthy = True
                    logger.info(f"节点 {node.id} 自动恢复")
                else:
                    continue
            
            available.append(node)
        
        return available
    
    def _round_robin_select(self, nodes: List[ServerNode]) -> ServerNode:
        """轮询选择"""
        selected = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index += 1
        return selected
    
    def _weighted_round_robin_select(self, nodes: List[ServerNode]) -> ServerNode:
        """加权轮询选择"""
        total_weight = sum(node.weight for node in nodes)
        if total_weight == 0:
            return random.choice(nodes)
        
        random_weight = random.randint(1, total_weight)
        current_weight = 0
        
        for node in nodes:
            current_weight += node.weight
            if random_weight <= current_weight:
                return node
        
        return nodes[-1]  # 兜底选择
    
    def _least_connections_select(self, nodes: List[ServerNode]) -> ServerNode:
        """最少连接选择"""
        return min(nodes, key=lambda n: n.current_connections)
    
    def _least_response_time_select(self, nodes: List[ServerNode]) -> ServerNode:
        """最少响应时间选择"""
        return min(nodes, key=lambda n: n.avg_response_time)
    
    def _ip_hash_select(self, nodes: List[ServerNode], client_ip: Optional[str]) -> ServerNode:
        """IP哈希选择"""
        if not client_ip:
            return random.choice(nodes)
        
        hash_value = hash(client_ip) % len(nodes)
        return nodes[hash_value]
    
    def _random_select(self, nodes: List[ServerNode]) -> ServerNode:
        """随机选择"""
        return random.choice(nodes)
    
    def _consistent_hash_select(self, nodes: List[ServerNode], request_context: Optional[Dict[str, Any]]) -> ServerNode:
        """一致性哈希选择"""
        if not request_context:
            return random.choice(nodes)
        
        # 使用请求的关键信息作为哈希键
        hash_key = str(request_context.get('user_id', '')) + str(request_context.get('session_id', ''))
        hash_value = hash(hash_key) % len(nodes)
        return nodes[hash_value]
    
    async def record_request_start(self, node_id: str):
        """记录请求开始"""
        if node_id in self.nodes:
            self.nodes[node_id].current_connections += 1
            self.nodes[node_id].total_requests += 1
    
    async def record_request_end(
        self,
        node_id: str,
        success: bool,
        response_time: float
    ):
        """记录请求结束"""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.current_connections = max(0, node.current_connections - 1)
        
        if success:
            node.successful_requests += 1
        else:
            node.failed_requests += 1
        
        # 更新平均响应时间
        if node.avg_response_time == 0:
            node.avg_response_time = response_time
        else:
            node.avg_response_time = (node.avg_response_time + response_time) / 2
    
    def get_node_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取节点统计信息"""
        stats = {}
        
        for node_id, node in self.nodes.items():
            stats[node_id] = {
                'host': node.host,
                'port': node.port,
                'weight': node.weight,
                'current_connections': node.current_connections,
                'total_requests': node.total_requests,
                'successful_requests': node.successful_requests,
                'failed_requests': node.failed_requests,
                'avg_response_time': node.avg_response_time,
                'success_rate': node.success_rate,
                'error_rate': node.error_rate,
                'load_score': node.load_score,
                'is_healthy': node.is_healthy,
                'consecutive_failures': node.consecutive_failures,
                'last_health_check': node.last_health_check
            }
        
        return stats
    
    def get_load_balance_stats(self) -> Dict[str, Any]:
        """获取负载均衡统计信息"""
        total_requests = sum(node.total_requests for node in self.nodes.values())
        total_successful = sum(node.successful_requests for node in self.nodes.values())
        total_failed = sum(node.failed_requests for node in self.nodes.values())
        total_connections = sum(node.current_connections for node in self.nodes.values())
        
        avg_response_time = 0
        if self.nodes:
            avg_response_time = sum(node.avg_response_time for node in self.nodes.values()) / len(self.nodes)
        
        healthy_nodes = sum(1 for node in self.nodes.values() if node.is_healthy)
        
        return {
            'algorithm': self.algorithm.value,
            'total_nodes': len(self.nodes),
            'healthy_nodes': healthy_nodes,
            'unhealthy_nodes': len(self.nodes) - healthy_nodes,
            'total_requests': total_requests,
            'total_successful': total_successful,
            'total_failed': total_failed,
            'overall_success_rate': total_successful / total_requests if total_requests > 0 else 0,
            'total_connections': total_connections,
            'avg_response_time': avg_response_time,
            'round_robin_index': self.round_robin_index
        }
    
    def set_algorithm(self, algorithm: LoadBalanceAlgorithm):
        """设置负载均衡算法"""
        self.algorithm = algorithm
        logger.info(f"负载均衡算法已更新为: {algorithm.value}")
    
    def set_health_check_config(
        self,
        interval: int = None,
        failure_threshold: int = None,
        recovery_threshold: int = None
    ):
        """设置健康检查配置"""
        if interval is not None:
            self.health_check_interval = interval
        if failure_threshold is not None:
            self.failure_threshold = failure_threshold
        if recovery_threshold is not None:
            self.recovery_threshold = recovery_threshold
        
        # 重启健康检查器
        if self.health_check_task and not self.health_check_task.done():
            self.health_check_task.cancel()
        
        self._start_health_checker()
        
        logger.info(f"健康检查配置已更新: interval={self.health_check_interval}, "
                   f"failure_threshold={self.failure_threshold}, recovery_threshold={self.recovery_threshold}")
    
    async def shutdown(self):
        """关闭负载均衡器"""
        if self.health_check_task and not self.health_check_task.done():
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("负载均衡器已关闭")