"""
Agent层指挥中枢主启动文件
演示完整的智能体系统
"""
import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

# 导入所有智能体组件
from src.agents import (
    NormaCommandCenter,
    SystemMonitorAgent,
    NetworkSecurityAgent, 
    BloodlineAnalysisAgent,
    ConversationCoordinatorAgent,
    LoadBalancer,
    AutoScaler,
    TaskScheduler,
    CollaborationManager,
    TaskDistributor,
    PoolConfig,
    LoadBalancingStrategy,
    CollaborationMode,
    TaskNode,
    AgentHealthMonitor,
    AgentMetricsCollector,
    AgentUtility
)


class AgentSystemOrchestrator:
    """智能体系统编排器"""
    
    def __init__(self):
        self.norma_center = None
        self.specialist_agents = {}
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler(self.load_balancer)
        self.task_scheduler = TaskScheduler()
        self.collaboration_manager = CollaborationManager(self.task_scheduler)
        self.task_distributor = TaskDistributor(self.collaboration_manager)
        self.health_monitor = AgentHealthMonitor()
        self.metrics_collector = AgentMetricsCollector()
        self.logger = logging.getLogger("agent.orchestrator")
        
    async def initialize(self):
        """初始化智能体系统"""
        try:
            self.logger.info("初始化智能体系统...")
            
            # 1. 创建诺玛指挥中枢
            self.norma_center = NormaCommandCenter()
            await self.norma_center.initialize()
            
            # 2. 创建专业智能体
            await self._create_specialist_agents()
            
            # 3. 设置负载均衡
            await self._setup_load_balancing()
            
            # 4. 配置任务分发
            await self._setup_task_distribution()
            
            # 5. 启动健康监控
            await self._start_health_monitoring()
            
            # 6. 启动自动扩缩容
            asyncio.create_task(self.auto_scaler.start_auto_scaling())
            
            self.logger.info("智能体系统初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            return False
    
    async def _create_specialist_agents(self):
        """创建专业智能体"""
        # 系统监控智能体
        system_monitor = SystemMonitorAgent(
            agent_id="system-monitor-001",
            config={
                "interval": 30,
                "thresholds": {
                    "cpu": 80.0,
                    "memory": 85.0,
                    "disk": 90.0
                }
            }
        )
        await system_monitor.initialize()
        self.specialist_agents["system_monitor"] = system_monitor
        self.norma_center.register_specialist_agent("system_monitor", system_monitor)
        
        # 网络安全智能体
        security_agent = NetworkSecurityAgent(
            agent_id="security-001",
            config={
                "interval": 60,
                "blacklist_ips": ["192.168.1.100", "10.0.0.50"],
                "suspicious_ports": [22, 23, 135, 139, 445]
            }
        )
        await security_agent.initialize()
        self.specialist_agents["network_security"] = security_agent
        self.norma_center.register_specialist_agent("network_security", security_agent)
        
        # 血统分析智能体
        bloodline_agent = BloodlineAnalysisAgent(
            agent_id="bloodline-001",
            config={
                "confidence_threshold": 0.8,
                "analysis_depth": 5
            }
        )
        await bloodline_agent.initialize()
        self.specialist_agents["bloodline_analysis"] = bloodline_agent
        self.norma_center.register_specialist_agent("bloodline_analysis", bloodline_agent)
        
        # 对话协调智能体
        conversation_agent = ConversationCoordinatorAgent(
            agent_id="conversation-001",
            config={
                "coordination_rules": {
                    "resource_contention_resolution": "arbitration",
                    "message_conflict_resolution": "timestamp_ordering"
                }
            }
        )
        await conversation_agent.initialize()
        self.specialist_agents["conversation_coordinator"] = conversation_agent
        self.norma_center.register_specialist_agent("conversation_coordinator", conversation_agent)
        
        self.logger.info(f"创建了 {len(self.specialist_agents)} 个专业智能体")
    
    async def _setup_load_balancing(self):
        """设置负载均衡"""
        # 为每个智能体类型创建池
        for agent_type, agent in self.specialist_agents.items():
            pool_config = PoolConfig(
                pool_name=f"{agent_type}_pool",
                agent_type=agent_type,
                min_size=1,
                max_size=5,
                scaling_policy="auto",
                health_check_interval=30,
                load_threshold=0.8,
                response_time_threshold=1000.0,
                scaling_up_cooldown=60,
                scaling_down_cooldown=300
            )
            
            pool = self.load_balancer.create_pool(pool_config)
            
            # 添加初始智能体到池中
            metrics = {
                "agent_id": agent.agent_id,
                "current_load": 0.0,
                "active_connections": 0,
                "response_time": 0.0,
                "success_rate": 1.0,
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "last_heartbeat": datetime.now(),
                "total_requests": 0,
                "failed_requests": 0
            }
            
            pool.add_agent(agent.agent_id, metrics)
            
            # 注册健康检查
            self.health_monitor.register_agent(agent.agent_id, 30)
        
        self.logger.info("负载均衡设置完成")
    
    async def _setup_task_distribution(self):
        """设置任务分发"""
        # 注册智能体能力
        self.task_distributor.register_agent_capability(
            "system-monitor-001", 
            ["cpu_monitoring", "memory_monitoring", "disk_monitoring"]
        )
        
        self.task_distributor.register_agent_capability(
            "security-001",
            ["intrusion_detection", "port_scanning", "vulnerability_assessment"]
        )
        
        self.task_distributor.register_agent_capability(
            "bloodline-001",
            ["lineage_tracking", "impact_analysis", "data_quality_assessment"]
        )
        
        self.task_distributor.register_agent_capability(
            "conversation-001",
            ["conversation_management", "multi_agent_coordination", "message_routing"]
        )
        
        # 设置分发规则
        self.task_distributor.set_distribution_rule("system_monitor", {
            "strategy": "capability_based"
        })
        
        self.task_distributor.set_distribution_rule("network_security", {
            "strategy": "load_balanced"
        })
        
        self.task_distributor.set_distribution_rule("bloodline_analysis", {
            "strategy": "collaboration"
        })
        
        self.task_distributor.set_distribution_rule("conversation_coordinator", {
            "strategy": "round_robin"
        })
        
        self.logger.info("任务分发设置完成")
    
    async def _start_health_monitoring(self):
        """启动健康监控"""
        # 为每个智能体启动健康检查
        for agent_type, agent in self.specialist_agents.items():
            asyncio.create_task(self._monitor_agent_health(agent))
        
        self.logger.info("健康监控启动完成")
    
    async def _monitor_agent_health(self, agent):
        """监控单个智能体健康状态"""
        while True:
            try:
                # 执行健康检查
                health_result = await agent.get_status()
                
                # 记录健康状态
                await self.health_monitor.check_agent_health(
                    agent.agent_id,
                    lambda: health_result
                )
                
                # 记录指标
                if "load" in health_result:
                    self.metrics_collector.record_metric(
                        agent.agent_id, 
                        "load", 
                        health_result["load"]
                    )
                
                await asyncio.sleep(30)  # 30秒检查一次
                
            except Exception as e:
                self.logger.error(f"健康监控错误 [{agent.agent_id}]: {e}")
                await asyncio.sleep(30)
    
    async def demonstrate_collaboration(self):
        """演示智能体协作"""
        self.logger.info("开始演示智能体协作...")
        
        # 创建协作任务
        tasks = [
            TaskNode(
                task_id="task_001",
                agent_id="system-monitor-001",
                task_type="system_monitor",
                priority=1,
                payload={"action": "get_system_status"},
                dependencies=[],
                created_at=datetime.now()
            ),
            TaskNode(
                task_id="task_002", 
                agent_id="security-001",
                task_type="network_security",
                priority=2,
                payload={"action": "get_security_status"},
                dependencies=["task_001"],
                created_at=datetime.now()
            ),
            TaskNode(
                task_id="task_003",
                agent_id="bloodline-001", 
                task_type="bloodline_analysis",
                priority=1,
                payload={"action": "get_data_catalog"},
                dependencies=["task_001"],
                created_at=datetime.now()
            )
        ]
        
        # 创建顺序协作
        collaboration_id = await self.collaboration_manager.create_collaboration(
            mode=CollaborationMode.SEQUENTIAL,
            tasks=tasks,
            participants=list(self.specialist_agents.keys())
        )
        
        self.logger.info(f"创建协作: {collaboration_id}")
        
        # 等待协作完成
        await asyncio.sleep(2)
        
        # 获取协作状态
        status = self.collaboration_manager.get_collaboration_status(collaboration_id)
        self.logger.info(f"协作状态: {status}")
    
    async def demonstrate_task_distribution(self):
        """演示任务分发"""
        self.logger.info("开始演示任务分发...")
        
        # 创建测试任务
        test_tasks = [
            {
                "task_id": "test_001",
                "task_type": "system_monitor",
                "payload": {"action": "get_performance_metrics"}
            },
            {
                "task_id": "test_002", 
                "task_type": "network_security",
                "payload": {"action": "scan_port", "host": "127.0.0.1", "port": 80}
            },
            {
                "task_id": "test_003",
                "task_type": "bloodline_analysis", 
                "payload": {"action": "get_lineage_graph"}
            }
        ]
        
        for task_info in test_tasks:
            # 创建任务节点
            task_node = TaskNode(
                task_id=task_info["task_id"],
                agent_id="",  # 将由分发器决定
                task_type=task_info["task_type"],
                priority=1,
                payload=task_info["payload"],
                dependencies=[],
                created_at=datetime.now()
            )
            
            # 分发任务
            assigned_agent = self.task_distributor.distribute_task(task_node)
            self.logger.info(f"任务 {task_info['task_id']} 分发到 {assigned_agent}")
            
            # 模拟任务执行
            await asyncio.sleep(0.5)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "norma_center": self.norma_center.get_system_status() if self.norma_center else None,
            "load_balancer": self.load_balancer.get_load_balancer_status(),
            "health_monitor": self.health_monitor.get_all_health_status(),
            "task_scheduler": self.task_scheduler.get_scheduler_status(),
            "collaboration": self.collaboration_manager.get_all_collaborations_status(),
            "task_distribution": self.task_distributor.get_distribution_status()
        }
        
        return status
    
    async def shutdown(self):
        """关闭系统"""
        self.logger.info("关闭智能体系统...")
        
        # 关闭所有智能体
        for agent in self.specialist_agents.values():
            try:
                await agent.shutdown()
            except Exception as e:
                self.logger.error(f"关闭智能体失败: {e}")
        
        # 关闭诺玛指挥中枢
        if self.norma_center:
            try:
                await self.norma_center.shutdown()
            except Exception as e:
                self.logger.error(f"关闭诺玛指挥中枢失败: {e}")
        
        self.logger.info("智能体系统已关闭")


async def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建系统编排器
    orchestrator = AgentSystemOrchestrator()
    
    try:
        # 初始化系统
        success = await orchestrator.initialize()
        if not success:
            print("系统初始化失败")
            return
        
        print("🎯 Agent层指挥中枢系统启动成功！")
        print("=" * 50)
        
        # 演示功能
        await orchestrator.demonstrate_collaboration()
        await orchestrator.demonstrate_task_distribution()
        
        # 显示系统状态
        print("\n📊 系统状态:")
        status = await orchestrator.get_system_status()
        print(f"活跃智能体: {len(orchestrator.specialist_agents)}")
        print(f"负载均衡池: {len(orchestrator.load_balancer.pools)}")
        print(f"健康监控: {status['health_monitor']['total_agents']} 个智能体")
        
        # 保持系统运行
        print("\n🔄 系统运行中，按 Ctrl+C 停止...")
        try:
            while True:
                await asyncio.sleep(10)
                # 定期显示系统状态
                current_status = await orchestrator.get_system_status()
                print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - 系统运行正常")
                
        except KeyboardInterrupt:
            print("\n🛑 收到停止信号...")
            
    except Exception as e:
        print(f"❌ 系统运行错误: {e}")
        
    finally:
        # 关闭系统
        await orchestrator.shutdown()
        print("👋 系统已安全关闭")


if __name__ == "__main__":
    asyncio.run(main())