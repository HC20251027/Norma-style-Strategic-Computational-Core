#!/usr/bin/env python3
"""
Agent层指挥中枢系统演示脚本
展示四大专业智能体的核心功能
"""
import asyncio
import sys
import os
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def demo_system_monitoring():
    """演示系统监控功能"""
    print("\n💻 === 系统监控智能体演示 ===")
    
    from src.agents.specialists import SystemMonitorAgent
    from src.agents.core import Task, TaskPriority
    
    # 创建系统监控智能体
    monitor = SystemMonitorAgent(agent_id="demo-monitor")
    await monitor.initialize()
    
    # 获取系统状态
    task = Task(
        id="demo-sys-status",
        agent_type="system_monitor",
        priority=TaskPriority.NORMAL,
        payload={"type": "get_system_status"},
        created_at=datetime.now()
    )
    
    result = await monitor.process_task(task)
    if result["status"] == "success":
        print(f"✅ CPU使用率: {result['cpu']['usage_percent']:.1f}%")
        print(f"✅ 内存使用率: {result['memory']['percent']:.1f}%")
        print(f"✅ 磁盘使用率: {result['disk']['percent']:.1f}%")
        print(f"✅ 运行进程数: {result['processes']}")
    
    await monitor.shutdown()

async def demo_network_security():
    """演示网络安全功能"""
    print("\n🔒 === 网络安全智能体演示 ===")
    
    from src.agents.specialists import NetworkSecurityAgent
    from src.agents.core import Task, TaskPriority
    
    # 创建网络安全智能体
    security = NetworkSecurityAgent(agent_id="demo-security")
    await security.initialize()
    
    # IP声誉检查
    task = Task(
        id="demo-ip-check",
        agent_type="network_security",
        priority=TaskPriority.HIGH,
        payload={"type": "check_ip_reputation", "ip": "8.8.8.8"},
        created_at=datetime.now()
    )
    
    result = await security.process_task(task)
    if result["status"] == "success":
        print(f"✅ IP检查完成: {result['ip']}")
        print(f"✅ 声誉状态: {result['reputation']}")
    
    # 端口扫描
    scan_task = Task(
        id="demo-port-scan",
        agent_type="network_security",
        priority=TaskPriority.NORMAL,
        payload={"type": "scan_port", "host": "127.0.0.1", "port": 80},
        created_at=datetime.now()
    )
    
    scan_result = await security.process_task(scan_task)
    if scan_result["status"] == "success":
        print(f"✅ 端口扫描: 127.0.0.1:80 - {'开放' if scan_result['is_open'] else '关闭'}")
        print(f"✅ 威胁等级: {scan_result['threat_level']}")
    
    await security.shutdown()

async def demo_bloodline_analysis():
    """演示血统分析功能"""
    print("\n🧬 === 血统分析智能体演示 ===")
    
    from src.agents.specialists import BloodlineAnalysisAgent
    from src.agents.core import Task, TaskPriority
    
    # 创建血统分析智能体
    bloodline = BloodlineAnalysisAgent(agent_id="demo-bloodline")
    await bloodline.initialize()
    
    # 跟踪数据血缘
    lineage_task = Task(
        id="demo-lineage",
        agent_type="bloodline_analysis",
        priority=TaskPriority.NORMAL,
        payload={
            "type": "track_lineage",
            "source_table": "users",
            "target_table": "user_stats",
            "transformation": "aggregation",
            "confidence": 0.95
        },
        created_at=datetime.now()
    )
    
    result = await bloodline.process_task(lineage_task)
    if result["status"] == "success":
        print(f"✅ 血缘记录: {result['source_table']} -> {result['target_table']}")
        print(f"✅ 置信度: {result['confidence']}")
    
    # 影响分析
    impact_task = Task(
        id="demo-impact",
        agent_type="bloodline_analysis",
        priority=TaskPriority.HIGH,
        payload={"type": "analyze_impact", "change_target": "users"},
        created_at=datetime.now()
    )
    
    impact_result = await bloodline.process_task(impact_task)
    if impact_result["status"] == "success":
        analysis = impact_result["analysis"]
        print(f"✅ 影响分析: {analysis['total_impacts']} 个影响项")
        print(f"✅ 影响评分: {analysis['impact_score']:.2f}")
        print(f"✅ 风险等级: {analysis['risk_level']}")
    
    await bloodline.shutdown()

async def demo_conversation_coordination():
    """演示对话协调功能"""
    print("\n💬 === 对话协调智能体演示 ===")
    
    from src.agents.specialists import ConversationCoordinatorAgent
    from src.agents.core import Task, TaskPriority
    
    # 创建对话协调智能体
    coordinator = ConversationCoordinatorAgent(agent_id="demo-coordinator")
    await coordinator.initialize()
    
    # 创建对话
    conv_task = Task(
        id="demo-conversation",
        agent_type="conversation_coordinator",
        priority=TaskPriority.NORMAL,
        payload={
            "type": "create_conversation",
            "user_id": "demo-user",
            "participants": ["agent1", "agent2"],
            "context_data": {"topic": "system_demonstration"}
        },
        created_at=datetime.now()
    )
    
    result = await coordinator.process_task(conv_task)
    if result["status"] == "success":
        print(f"✅ 对话创建: {result['conversation_id']}")
        print(f"✅ 参与者: {result['participants']}")
    
    # 智能体协调
    coord_task = Task(
        id="demo-coordination",
        agent_type="conversation_coordinator",
        priority=TaskPriority.HIGH,
        payload={
            "type": "coordinate_agents",
            "coordination_type": "collaborative",
            "agents": ["agent1", "agent2"],
            "task_description": "联合演示任务"
        },
        created_at=datetime.now()
    )
    
    coord_result = await coordinator.process_task(coord_task)
    if coord_result["status"] == "success":
        print(f"✅ 协作创建: {coord_result['coordination']['coordination_id']}")
        print(f"✅ 协作模式: {coord_result['coordination']['coordination_type']}")
    
    await coordinator.shutdown()

async def demo_load_balancing():
    """演示负载均衡功能"""
    print("\n⚖️ === 负载均衡演示 ===")
    
    from src.agents.management import (
        LoadBalancer, PoolConfig, LoadBalancingStrategy,
        AgentMetrics
    )
    from datetime import datetime
    
    # 创建负载均衡器
    lb = LoadBalancer()
    
    # 创建池配置
    pool_config = PoolConfig(
        pool_name="demo_pool",
        agent_type="demo_agent",
        min_size=2,
        max_size=5,
        scaling_policy="auto",
        health_check_interval=30,
        load_threshold=0.8,
        response_time_threshold=1000.0,
        scaling_up_cooldown=60,
        scaling_down_cooldown=300
    )
    
    # 创建池
    pool = lb.create_pool(pool_config)
    
    # 添加智能体到池中
    agents = ["agent-1", "agent-2", "agent-3"]
    for i, agent_id in enumerate(agents):
        metrics = AgentMetrics(
            agent_id=agent_id,
            current_load=0.3 + i * 0.2,  # 不同负载
            active_connections=i,
            response_time=100.0 + i * 50.0,
            success_rate=0.95 - i * 0.05,
            cpu_usage=0.4 + i * 0.1,
            memory_usage=0.5 + i * 0.1,
            last_heartbeat=datetime.now(),
            total_requests=100 + i * 50,
            failed_requests=5 + i
        )
        pool.add_agent(agent_id, metrics)
    
    print(f"✅ 创建智能体池: {pool_config.pool_name}")
    print(f"✅ 添加智能体: {len(agents)} 个")
    
    # 测试不同负载均衡策略
    strategies = [
        LoadBalancingStrategy.ROUND_ROBIN,
        LoadBalancingStrategy.LEAST_CONNECTIONS,
        LoadBalancingStrategy.LEAST_RESPONSE_TIME,
        LoadBalancingStrategy.RESOURCE_BASED
    ]
    
    for strategy in strategies:
        selected = lb.select_agent("demo_pool", strategy)
        print(f"✅ {strategy.value}: 选择 {selected}")
    
    # 显示池状态
    status = pool.get_pool_status()
    print(f"✅ 池状态: 平均负载 {status['avg_load']:.2f}, 平均响应时间 {status['avg_response_time']:.1f}ms")

async def demo_collaboration():
    """演示协作功能"""
    print("\n🤝 === 智能体协作演示 ===")
    
    from src.agents.management import (
        TaskScheduler, CollaborationManager, CollaborationMode,
        TaskNode
    )
    
    # 创建调度器和协作管理器
    scheduler = TaskScheduler()
    collab_manager = CollaborationManager(scheduler)
    
    # 创建任务节点
    tasks = [
        TaskNode(
            task_id="task-1",
            agent_id="worker-1",
            task_type="data_processing",
            priority=1,
            payload={"action": "process_data"},
            dependencies=[],
            created_at=datetime.now()
        ),
        TaskNode(
            task_id="task-2",
            agent_id="worker-2",
            task_type="analysis",
            priority=2,
            payload={"action": "analyze_results"},
            dependencies=["task-1"],
            created_at=datetime.now()
        ),
        TaskNode(
            task_id="task-3",
            agent_id="worker-3",
            task_type="reporting",
            priority=1,
            payload={"action": "generate_report"},
            dependencies=["task-2"],
            created_at=datetime.now()
        )
    ]
    
    # 创建顺序协作
    collab_id = await collab_manager.create_collaboration(
        mode=CollaborationMode.SEQUENTIAL,
        tasks=tasks,
        participants=["worker-1", "worker-2", "worker-3"]
    )
    
    print(f"✅ 创建协作: {collab_id}")
    print(f"✅ 协作模式: {CollaborationMode.SEQUENTIAL.value}")
    print(f"✅ 任务数量: {len(tasks)}")
    
    # 显示调度器状态
    sched_status = scheduler.get_scheduler_status()
    print(f"✅ 调度状态: 待处理 {sched_status['pending_tasks']}, 运行中 {sched_status['running_tasks']}")

async def main():
    """主演示函数"""
    print("🚀 Agent层指挥中枢系统功能演示")
    print("=" * 60)
    print("本演示将展示四大专业智能体的核心功能")
    print("包括系统监控、网络安全、血统分析、对话协调")
    print("以及负载均衡和智能体协作机制")
    print("=" * 60)
    
    try:
        # 演示各个功能模块
        await demo_system_monitoring()
        await demo_network_security()
        await demo_bloodline_analysis()
        await demo_conversation_coordination()
        await demo_load_balancing()
        await demo_collaboration()
        
        print("\n" + "=" * 60)
        print("🎉 演示完成！所有功能模块运行正常。")
        print("\n📚 更多信息:")
        print("   • 查看 README.md 了解详细文档")
        print("   • 运行 python src/agents/main.py 启动完整系统")
        print("   • 运行 python test_agent_system.py 进行系统测试")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())