#!/usr/bin/env python3
"""
多智能体协作系统简化演示程序
展示多智能体系统的核心功能和概念

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import time
import json
import sys
import os
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime


class AgentStatus(Enum):
    """智能体状态枚举"""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Agent:
    """智能体类"""
    id: str
    name: str
    capabilities: List[str]
    status: AgentStatus
    current_load: float = 0.0
    performance_score: float = 1.0


@dataclass
class Task:
    """任务类"""
    id: str
    name: str
    description: str
    priority: int
    dependencies: List[str]
    required_capabilities: List[str]
    estimated_duration: int
    status: TaskStatus
    assigned_agent: Optional[str] = None


class SimpleMultiAgentSystem:
    """简化的多智能体协作系统"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.messages: List[Dict] = []
        self.performance_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_response_time": 0.0
        }
    
    async def register_agent(self, agent: Agent) -> Dict[str, Any]:
        """注册智能体"""
        self.agents[agent.id] = agent
        return {"success": True, "agent_id": agent.id}
    
    async def assign_task(self, task: Task) -> Dict[str, Any]:
        """分配任务"""
        # 查找合适的智能体
        suitable_agents = []
        for agent in self.agents.values():
            if (agent.status == AgentStatus.IDLE and 
                all(cap in agent.capabilities for cap in task.required_capabilities)):
                suitable_agents.append(agent)
        
        if not suitable_agents:
            return {"success": False, "error": "没有合适的智能体"}
        
        # 选择性能最好的智能体
        best_agent = max(suitable_agents, key=lambda a: a.performance_score / (a.current_load + 0.1))
        
        # 分配任务
        task.assigned_agent = best_agent.id
        task.status = TaskStatus.ASSIGNED
        best_agent.status = AgentStatus.BUSY
        best_agent.current_load += 0.2
        
        self.tasks[task.id] = task
        self.performance_metrics["total_tasks"] += 1
        
        return {
            "success": True, 
            "task_id": task.id, 
            "assigned_agent": best_agent.id
        }
    
    async def complete_task(self, task_id: str, success: bool = True) -> Dict[str, Any]:
        """完成任务"""
        if task_id not in self.tasks:
            return {"success": False, "error": "任务不存在"}
        
        task = self.tasks[task_id]
        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        
        # 释放智能体
        if task.assigned_agent and task.assigned_agent in self.agents:
            agent = self.agents[task.assigned_agent]
            agent.status = AgentStatus.IDLE
            agent.current_load = max(0, agent.current_load - 0.2)
        
        # 更新指标
        if success:
            self.performance_metrics["completed_tasks"] += 1
        else:
            self.performance_metrics["failed_tasks"] += 1
        
        return {"success": True, "task_id": task_id, "status": task.status.value}
    
    async def send_message(self, message: Dict) -> Dict[str, Any]:
        """发送消息"""
        message["timestamp"] = time.time()
        self.messages.append(message)
        return {"success": True, "message_id": len(self.messages)}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        total = self.performance_metrics["total_tasks"]
        completed = self.performance_metrics["completed_tasks"]
        success_rate = (completed / total * 100) if total > 0 else 0
        
        return {
            **self.performance_metrics,
            "success_rate": success_rate,
            "agent_count": len(self.agents),
            "active_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS])
        }


async def demo_basic_functionality():
    """演示基础功能"""
    print("\n" + "="*60)
    print("📋 基础功能演示")
    print("="*60)
    
    system = SimpleMultiAgentSystem()
    
    # 1. 智能体注册
    print("\n1️⃣ 智能体注册演示")
    agents = [
        Agent("manager_001", "项目经理", ["management", "coordination"], AgentStatus.IDLE),
        Agent("worker_001", "数据分析师", ["data_analysis", "reporting"], AgentStatus.IDLE),
        Agent("worker_002", "开发工程师", ["programming", "testing"], AgentStatus.IDLE),
        Agent("specialist_001", "AI专家", ["machine_learning", "data_processing"], AgentStatus.IDLE)
    ]
    
    for agent in agents:
        result = await system.register_agent(agent)
        if result["success"]:
            print(f"   ✅ 智能体 {agent.name} 注册成功")
    
    # 2. 任务分配
    print("\n2️⃣ 任务分配演示")
    tasks = [
        Task("task_001", "数据分析", "分析销售数据", 1, [], ["data_analysis"], 30, TaskStatus.PENDING),
        Task("task_002", "代码开发", "开发新功能", 2, [], ["programming"], 60, TaskStatus.PENDING),
        Task("task_003", "模型训练", "训练机器学习模型", 3, [], ["machine_learning"], 120, TaskStatus.PENDING)
    ]
    
    for task in tasks:
        result = await system.assign_task(task)
        if result["success"]:
            print(f"   ✅ 任务 '{task.name}' 分配给 {result['assigned_agent']}")
        else:
            print(f"   ❌ 任务 '{task.name}' 分配失败: {result['error']}")
    
    # 3. 完成任务
    print("\n3️⃣ 任务完成演示")
    for task in tasks:
        await asyncio.sleep(0.1)  # 模拟处理时间
        result = await system.complete_task(task.id, success=True)
        if result["success"]:
            print(f"   ✅ 任务 '{task.name}' 完成")
    
    # 4. 消息通信
    print("\n4️⃣ 消息通信演示")
    message = {
        "type": "notification",
        "content": "所有任务已完成！",
        "sender": "system"
    }
    result = await system.send_message(message)
    if result["success"]:
        print("   ✅ 系统广播消息发送成功")
    
    # 5. 性能指标
    print("\n5️⃣ 性能指标演示")
    metrics = await system.get_performance_metrics()
    print(f"   📊 总任务数: {metrics['total_tasks']}")
    print(f"   ✅ 完成任务: {metrics['completed_tasks']}")
    print(f"   📈 成功率: {metrics['success_rate']:.1f}%")
    print(f"   👥 智能体数: {metrics['agent_count']}")
    
    return system


async def demo_collaboration_patterns():
    """演示协作模式"""
    print("\n" + "="*60)
    print("🤝 协作模式演示")
    print("="*60)
    
    system = SimpleMultiAgentSystem()
    
    # 注册智能体团队
    team_agents = []
    roles = [
        ("team_lead", "团队领导", ["management", "coordination"]),
        ("dev_1", "开发工程师1", ["programming", "testing"]),
        ("dev_2", "开发工程师2", ["programming", "debugging"]),
        ("tester", "测试工程师", ["testing", "quality_assurance"]),
        ("analyst", "数据分析师", ["data_analysis", "reporting"])
    ]
    
    for role_id, name, capabilities in roles:
        agent = Agent(role_id, name, capabilities, AgentStatus.IDLE)
        await system.register_agent(agent)
        team_agents.append(agent)
    
    print(f"\n👥 创建了 {len(team_agents)} 人团队")
    
    # 1. 层次化协作 (团队领导管理项目)
    print("\n1️⃣ 层次化协作演示")
    project_task = Task("project_001", "新产品开发", "开发新产品功能", 1, [], ["management"], 180, TaskStatus.PENDING)
    result = await system.assign_task(project_task)
    if result["success"]:
        print(f"   ✅ 项目任务分配给团队领导: {result['assigned_agent']}")
    
    # 2. 并行协作 (多个开发任务同时进行)
    print("\n2️⃣ 并行协作演示")
    dev_tasks = [
        Task("dev_task_1", "前端开发", "开发用户界面", 2, [], ["programming"], 90, TaskStatus.PENDING),
        Task("dev_task_2", "后端开发", "开发API接口", 2, [], ["programming"], 120, TaskStatus.PENDING),
        Task("dev_task_3", "数据库设计", "设计数据模型", 2, [], ["data_analysis"], 60, TaskStatus.PENDING)
    ]
    
    for task in dev_tasks:
        result = await system.assign_task(task)
        if result["success"]:
            print(f"   ✅ 开发任务 '{task.name}' 分配给: {result['assigned_agent']}")
    
    # 3. 流水线协作 (测试->修复->验证)
    print("\n3️⃣ 流水线协作演示")
    pipeline_tasks = [
        Task("test_001", "功能测试", "测试新功能", 3, [], ["testing"], 45, TaskStatus.PENDING),
        Task("fix_001", "缺陷修复", "修复发现的问题", 3, ["test_001"], ["programming"], 30, TaskStatus.PENDING),
        Task("verify_001", "验证测试", "验证修复效果", 3, ["fix_001"], ["testing"], 20, TaskStatus.PENDING)
    ]
    
    for task in pipeline_tasks:
        result = await system.assign_task(task)
        if result["success"]:
            print(f"   ✅ 流水线任务 '{task.name}' 分配给: {result['assigned_agent']}")
    
    # 模拟任务完成
    print("\n🔄 模拟任务执行...")
    await asyncio.sleep(0.5)
    
    for task_id in ["project_001", "dev_task_1", "dev_task_2", "dev_task_3"]:
        await system.complete_task(task_id, success=True)
    
    # 流水线任务按依赖顺序完成
    await system.complete_task("test_001", success=True)
    await system.complete_task("fix_001", success=True)
    await system.complete_task("verify_001", success=True)
    
    metrics = await system.get_performance_metrics()
    print(f"\n📊 协作模式执行结果:")
    print(f"   总任务: {metrics['total_tasks']}")
    print(f"   完成率: {metrics['success_rate']:.1f}%")
    
    return system


async def demo_load_balancing():
    """演示负载均衡"""
    print("\n" + "="*60)
    print("⚖️ 负载均衡演示")
    print("="*60)
    
    system = SimpleMultiAgentSystem()
    
    # 创建不同性能的智能体
    workers = []
    for i in range(5):
        performance = 0.8 + (i * 0.1)  # 性能递增
        agent = Agent(f"worker_{i:03d}", f"工作智能体{i+1}", ["task_execution"], AgentStatus.IDLE, performance_score=performance)
        await system.register_agent(agent)
        workers.append(agent)
    
    print(f"\n👥 创建了 {len(workers)} 个智能体，性能分数: {[f'{w.performance_score:.1f}' for w in workers]}")
    
    # 创建大量任务测试负载均衡
    print("\n🔄 创建15个任务测试负载均衡...")
    tasks = []
    for i in range(15):
        task = Task(f"load_task_{i:03d}", f"负载测试任务{i+1}", "测试系统负载", 1, [], ["task_execution"], 10, TaskStatus.PENDING)
        tasks.append(task)
        result = await system.assign_task(task)
        if result["success"]:
            print(f"   任务 {i+1:2d} -> {result['assigned_agent']}")
    
    # 分析负载分布
    print(f"\n📊 负载分布分析:")
    agent_loads = {}
    for agent in workers:
        agent_loads[agent.name] = agent.current_load
    
    for name, load in agent_loads.items():
        bar = "█" * int(load * 10)
        print(f"   {name:12s}: {bar:<10s} {load:.1f}")
    
    metrics = await system.get_performance_metrics()
    print(f"\n📈 负载均衡效果:")
    print(f"   平均负载: {sum(agent_loads.values())/len(agent_loads):.2f}")
    print(f"   任务分配: {metrics['total_tasks']} 个")
    
    return system


async def demo_communication_system():
    """演示通信系统"""
    print("\n" + "="*60)
    print("💬 通信系统演示")
    print("="*60)
    
    system = SimpleMultiAgentSystem()
    
    # 注册通信智能体
    comm_agents = []
    for i in range(3):
        agent = Agent(f"comm_agent_{i:03d}", f"通信智能体{i+1}", ["communication"], AgentStatus.IDLE)
        await system.register_agent(agent)
        comm_agents.append(agent)
    
    print(f"\n📡 注册了 {len(comm_agents)} 个通信智能体")
    
    # 1. 广播消息
    print("\n📢 广播消息演示")
    broadcast_msg = {
        "type": "broadcast",
        "content": "系统通知：开始新的协作任务！",
        "sender": "system"
    }
    result = await system.send_message(broadcast_msg)
    if result["success"]:
        print("   ✅ 广播消息发送成功")
    
    # 2. 点对点消息
    print("\n💭 点对点消息演示")
    p2p_msg = {
        "type": "direct_message",
        "content": "你好，我们开始协作吧！",
        "sender": comm_agents[0].id,
        "receiver": comm_agents[1].id
    }
    result = await system.send_message(p2p_msg)
    if result["success"]:
        print(f"   ✅ 消息从 {comm_agents[0].name} 发送到 {comm_agents[1].name}")
    
    # 3. 团队消息
    print("\n👥 团队消息演示")
    team_msg = {
        "type": "team_message",
        "content": "项目更新：所有功能开发完成！",
        "sender": comm_agents[1].id,
        "team": "development_team"
    }
    result = await system.send_message(team_msg)
    if result["success"]:
        print("   ✅ 团队消息发送成功")
    
    # 4. 消息统计
    print(f"\n📊 消息统计:")
    print(f"   总消息数: {len(system.messages)}")
    for i, msg in enumerate(system.messages, 1):
        print(f"   消息 {i}: {msg['type']} - {msg['content']}")
    
    return system


async def run_complete_demo():
    """运行完整演示"""
    print("🎬 多智能体协作系统完整演示")
    print("="*60)
    print("本演示将展示多智能体系统的各种功能和协作模式")
    print("="*60)
    
    demo_results = {}
    
    try:
        # 基础功能演示
        basic_system = await demo_basic_functionality()
        demo_results["基础功能"] = "完成"
        
        # 协作模式演示
        collab_system = await demo_collaboration_patterns()
        demo_results["协作模式"] = "完成"
        
        # 负载均衡演示
        load_system = await demo_load_balancing()
        demo_results["负载均衡"] = "完成"
        
        # 通信系统演示
        comm_system = await demo_communication_system()
        demo_results["通信系统"] = "完成"
        
        # 演示总结
        print("\n" + "="*60)
        print("📋 演示总结")
        print("="*60)
        
        print("\n✅ 演示完成的功能模块:")
        for module, result in demo_results.items():
            print(f"   🔹 {module}: {result}")
        
        # 生成演示报告
        demo_report = {
            "timestamp": time.time(),
            "demo_results": demo_results,
            "system_status": "演示完成",
            "total_modules": len(demo_results)
        }
        
        # 保存演示报告
        report_file = "simple_demo_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(demo_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 演示报告已保存到: {report_file}")
        
        print("\n🎉 多智能体协作系统演示结束")
        print("感谢您的观看！")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """主函数"""
    await run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())