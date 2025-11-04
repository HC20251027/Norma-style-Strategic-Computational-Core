#!/usr/bin/env python3
"""
多智能体协作系统演示程序
展示多智能体系统的各种功能和协作模式
"""

import asyncio
import time
import json
import sys
import os
from typing import Dict, List, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent.main_system import MultiAgentSystem
from multi_agent.utils.logger import Logger
from tests import create_test_agent, create_test_task


class MultiAgentDemo:
    """多智能体协作系统演示类"""
    
    def __init__(self):
        self.logger = Logger()
        self.system = None
        self.demo_results = {}
    
    async def initialize(self):
        """初始化演示系统"""
        print("🚀 初始化多智能体协作系统...")
        
        self.system = MultiAgentSystem()
        await self.system.initialize(logger=self.logger)
        
        print("✅ 系统初始化完成")
        return True
    
    async def cleanup(self):
        """清理演示系统"""
        if self.system:
            await self.system.shutdown()
            print("🧹 系统清理完成")
    
    async def demo_basic_functionality(self):
        """演示基础功能"""
        print("\n" + "="*60)
        print("📋 基础功能演示")
        print("="*60)
        
        # 1. 智能体注册
        print("\n1️⃣ 智能体注册演示")
        agents = [
            create_test_agent("manager_001", ["management", "coordination"]),
            create_test_agent("worker_001", ["task_execution", "data_processing"]),
            create_test_agent("worker_002", ["task_execution", "analysis"]),
            create_test_agent("specialist_001", ["machine_learning", "visualization"])
        ]
        
        for agent in agents:
            result = await self.system.register_agent(agent)
            if result["success"]:
                print(f"   ✅ 智能体 {agent['id']} 注册成功")
            else:
                print(f"   ❌ 智能体 {agent['id']} 注册失败")
        
        agent_count = await self.system.get_agent_count()
        print(f"   📊 当前注册智能体数量: {agent_count}")
        
        # 2. 任务分配
        print("\n2️⃣ 任务分配演示")
        tasks = [
            create_test_task("task_001", priority=1),
            create_test_task("task_002", priority=2),
            create_test_task("task_003", priority=3)
        ]
        
        for task in tasks:
            result = await self.system.assign_task(task)
            if result["success"]:
                assigned_agent = result.get("assigned_agent", "unknown")
                print(f"   ✅ 任务 {task['id']} 分配给智能体 {assigned_agent}")
            else:
                print(f"   ❌ 任务 {task['id']} 分配失败")
        
        # 3. 消息通信
        print("\n3️⃣ 消息通信演示")
        message = {
            "type": "demo_message",
            "content": "这是一条演示消息",
            "sender": "demo_system",
            "timestamp": time.time()
        }
        
        result = await self.system.broadcast_message(message)
        if result["success"]:
            print("   ✅ 消息广播成功")
        else:
            print("   ❌ 消息广播失败")
        
        self.demo_results["basic_functionality"] = {
            "agents_registered": agent_count,
            "tasks_assigned": len(tasks),
            "message_sent": result["success"]
        }
    
    async def demo_collaboration_patterns(self):
        """演示协作模式"""
        print("\n" + "="*60)
        print("🤝 协作模式演示")
        print("="*60)
        
        # 注册更多智能体用于协作
        collaboration_agents = []
        for i in range(8):
            agent = create_test_agent(f"collab_agent_{i:03d}", ["task_execution", "communication"])
            collaboration_agents.append(agent)
            await self.system.register_agent(agent)
        
        # 1. 层次化协作
        print("\n1️⃣ 层次化协作模式演示")
        hierarchical_task = {
            "id": "hierarchical_demo",
            "name": "层次化协作任务",
            "description": "展示层次化协作模式",
            "priority": 1,
            "subtasks": [
                create_test_task("hier_sub_001", priority=1),
                create_test_task("hier_sub_002", priority=2),
                create_test_task("hier_sub_003", priority=3)
            ]
        }
        
        try:
            result = await self.system.execute_collaboration_pattern(
                "hierarchical",
                {"task": hierarchical_task, "agents": collaboration_agents[:4]}
            )
            if result["success"]:
                print("   ✅ 层次化协作执行成功")
            else:
                print("   ❌ 层次化协作执行失败")
        except Exception as e:
            print(f"   ⚠️ 层次化协作演示跳过: {str(e)}")
        
        # 2. 流水线协作
        print("\n2️⃣ 流水线协作模式演示")
        pipeline_task = {
            "id": "pipeline_demo",
            "name": "流水线协作任务",
            "description": "展示流水线协作模式",
            "priority": 1,
            "pipeline_stages": ["input", "process", "output"]
        }
        
        try:
            result = await self.system.execute_collaboration_pattern(
                "pipeline",
                {"task": pipeline_task, "agents": collaboration_agents[4:7]}
            )
            if result["success"]:
                print("   ✅ 流水线协作执行成功")
            else:
                print("   ❌ 流水线协作执行失败")
        except Exception as e:
            print(f"   ⚠️ 流水线协作演示跳过: {str(e)}")
        
        # 3. P2P协作
        print("\n3️⃣ 点对点协作模式演示")
        p2p_task = create_test_task("p2p_demo", priority=2)
        
        try:
            result = await self.system.execute_collaboration_pattern(
                "peer_to_peer",
                {"task": p2p_task, "agents": collaboration_agents[7:]}
            )
            if result["success"]:
                print("   ✅ P2P协作执行成功")
            else:
                print("   ❌ P2P协作执行失败")
        except Exception as e:
            print(f"   ⚠️ P2P协作演示跳过: {str(e)}")
        
        self.demo_results["collaboration_patterns"] = "演示完成"
    
    async def demo_load_balancing(self):
        """演示负载均衡"""
        print("\n" + "="*60)
        print("⚖️ 负载均衡演示")
        print("="*60)
        
        # 注册具有不同性能的智能体
        load_agents = []
        performance_scores = [0.8, 1.0, 1.2, 0.9, 1.1]
        
        for i, score in enumerate(performance_scores):
            agent = create_test_agent(f"load_agent_{i:03d}", ["task_execution"])
            agent["performance_score"] = score
            agent["current_load"] = 0.0
            load_agents.append(agent)
            await self.system.register_agent(agent)
        
        print(f"\n📊 注册了 {len(load_agents)} 个智能体，性能分数: {[a['performance_score'] for a in load_agents]}")
        
        # 创建大量任务测试负载均衡
        print("\n🔄 创建20个任务测试负载均衡...")
        tasks = []
        for i in range(20):
            task = create_test_task(f"load_task_{i:03d}", priority=1)
            tasks.append(task)
            await self.system.assign_task(task)
        
        # 获取负载分布统计
        try:
            load_stats = await self.system.get_load_balance_stats()
            if "distribution" in load_stats:
                print("   ✅ 负载分布统计获取成功")
                print(f"   📈 负载分布: {load_stats['distribution']}")
            else:
                print("   ⚠️ 负载分布统计格式异常")
        except Exception as e:
            print(f"   ⚠️ 负载均衡统计获取失败: {str(e)}")
        
        self.demo_results["load_balancing"] = {
            "agents_count": len(load_agents),
            "tasks_processed": len(tasks)
        }
    
    async def demo_performance_monitoring(self):
        """演示性能监控"""
        print("\n" + "="*60)
        print("📊 性能监控演示")
        print("="*60)
        
        # 创建性能测试工作负载
        print("\n🔄 执行性能测试工作负载...")
        
        perf_agents = []
        for i in range(5):
            agent = create_test_agent(f"perf_agent_{i:03d}", ["task_execution"])
            perf_agents.append(agent)
            await self.system.register_agent(agent)
        
        # 执行多个批次的任务
        for batch in range(3):
            print(f"   📦 执行批次 {batch + 1}...")
            batch_tasks = []
            for i in range(8):
                task = create_test_task(f"perf_batch_{batch}_task_{i:03d}", priority=batch + 1)
                batch_tasks.append(task)
                await self.system.assign_task(task)
            
            # 等待处理
            await asyncio.sleep(0.1)
            
            # 收集性能指标
            try:
                metrics = await self.system.get_performance_metrics()
                if metrics:
                    throughput = metrics.get("throughput", 0)
                    response_time = metrics.get("response_time", 0)
                    print(f"      ✅ 吞吐量: {throughput:.2f}, 响应时间: {response_time:.4f}s")
                else:
                    print("      ⚠️ 性能指标获取失败")
            except Exception as e:
                print(f"      ⚠️ 性能监控错误: {str(e)}")
        
        # 生成性能报告
        print("\n📋 生成性能报告...")
        try:
            report = await self.system.generate_performance_report()
            if report and "summary" in report:
                print("   ✅ 性能报告生成成功")
                summary = report["summary"]
                print(f"   📊 报告摘要: {summary}")
            else:
                print("   ⚠️ 性能报告格式异常")
        except Exception as e:
            print(f"   ⚠️ 性能报告生成失败: {str(e)}")
        
        self.demo_results["performance_monitoring"] = "演示完成"
    
    async def demo_communication_system(self):
        """演示通信系统"""
        print("\n" + "="*60)
        print("💬 通信系统演示")
        print("="*60)
        
        # 注册通信智能体
        comm_agents = []
        for i in range(4):
            agent = create_test_agent(f"comm_agent_{i:03d}", ["communication", "coordination"])
            comm_agents.append(agent)
            await self.system.register_agent(agent)
        
        # 1. 广播消息
        print("\n📢 广播消息演示")
        broadcast_msg = {
            "type": "system_announcement",
            "content": "系统广播：欢迎使用多智能体协作系统！",
            "sender": "demo_system",
            "timestamp": time.time()
        }
        
        result = await self.system.broadcast_message(broadcast_msg)
        if result["success"]:
            print("   ✅ 广播消息发送成功")
        else:
            print("   ❌ 广播消息发送失败")
        
        # 2. 点对点消息
        print("\n💭 点对点消息演示")
        p2p_msg = {
            "type": "direct_message",
            "content": "你好，这是点对点消息演示",
            "sender": comm_agents[0]["id"],
            "receiver": comm_agents[1]["id"],
            "timestamp": time.time()
        }
        
        result = await self.system.send_direct_message(p2p_msg)
        if result["success"]:
            print("   ✅ 点对点消息发送成功")
        else:
            print("   ❌ 点对点消息发送失败")
        
        # 3. 获取消息历史
        print("\n📜 消息历史演示")
        try:
            message_history = await self.system.get_message_history()
            print(f"   📊 消息历史记录数: {len(message_history)}")
        except Exception as e:
            print(f"   ⚠️ 消息历史获取失败: {str(e)}")
        
        self.demo_results["communication"] = "演示完成"
    
    async def demo_system_optimization(self):
        """演示系统优化"""
        print("\n" + "="*60)
        print("🚀 系统优化演示")
        print("="*60)
        
        # 执行系统优化
        print("\n🔧 执行系统优化...")
        try:
            optimization_result = await self.system.optimize_system()
            if optimization_result:
                optimizations = optimization_result.get("optimizations_applied", [])
                print(f"   ✅ 应用了 {len(optimizations)} 项优化")
                for opt in optimizations:
                    print(f"      🔹 {opt}")
            else:
                print("   ⚠️ 优化结果为空")
        except Exception as e:
            print(f"   ⚠️ 系统优化失败: {str(e)}")
        
        # 更新系统配置
        print("\n⚙️ 配置管理演示")
        new_config = {
            "load_balancing_strategy": "weighted",
            "max_concurrent_tasks": 50,
            "performance_threshold": 0.8
        }
        
        try:
            config_result = await self.system.update_configuration(new_config)
            if config_result["success"]:
                print("   ✅ 配置更新成功")
                current_config = await self.system.get_configuration()
                print(f"   📋 当前配置: {current_config}")
            else:
                print("   ❌ 配置更新失败")
        except Exception as e:
            print(f"   ⚠️ 配置管理错误: {str(e)}")
        
        self.demo_results["system_optimization"] = "演示完成"
    
    async def demo_comprehensive_scenario(self):
        """演示综合场景"""
        print("\n" + "="*60)
        print("🎯 综合场景演示")
        print("="*60)
        
        print("\n🏢 模拟一个数据处理公司场景...")
        
        # 创建不同角色的智能体
        roles = {
            "management": ["project_manager", "team_lead"],
            "execution": ["data_analyst", "developer", "tester", "deployer"],
            "specialist": ["ml_engineer", "data_scientist", "ui_designer"]
        }
        
        all_agents = []
        for role_category, role_list in roles.items():
            for role in role_list:
                agent = create_test_agent(f"{role}_{role_category[:4]}", 
                                        self._get_role_capabilities(role))
                all_agents.append(agent)
                await self.system.register_agent(agent)
        
        print(f"   👥 创建了 {len(all_agents)} 个不同角色的智能体")
        
        # 模拟一个完整项目流程
        project_phases = [
            {"name": "需求分析", "tasks": 2, "priority": 1},
            {"name": "数据处理", "tasks": 4, "priority": 2},
            {"name": "模型开发", "tasks": 3, "priority": 3},
            {"name": "测试验证", "tasks": 2, "priority": 2},
            {"name": "部署上线", "tasks": 1, "priority": 1}
        ]
        
        print("\n📋 执行项目流程...")
        total_tasks = 0
        start_time = time.time()
        
        for phase in project_phases:
            print(f"   🔄 {phase['name']} 阶段...")
            
            phase_tasks = []
            for i in range(phase["tasks"]):
                task = create_test_task(
                    f"{phase['name'].lower().replace(' ', '_')}_task_{i:03d}",
                    priority=phase["priority"]
                )
                phase_tasks.append(task)
                await self.system.assign_task(task)
                total_tasks += 1
            
            # 模拟阶段处理时间
            await asyncio.sleep(0.2)
            
            print(f"      ✅ 完成 {len(phase_tasks)} 个任务")
        
        end_time = time.time()
        project_duration = end_time - start_time
        
        print(f"\n📊 项目执行总结:")
        print(f"   ⏱️  总耗时: {project_duration:.2f} 秒")
        print(f"   📋 总任务数: {total_tasks}")
        print(f"   🚀 平均速度: {total_tasks/project_duration:.2f} 任务/秒")
        
        # 获取最终系统状态
        final_metrics = await self.system.get_performance_metrics()
        if final_metrics:
            print(f"   📈 最终吞吐量: {final_metrics.get('throughput', 0):.2f}")
            print(f"   ⏱️  平均响应时间: {final_metrics.get('response_time', 0):.4f}s")
        
        self.demo_results["comprehensive_scenario"] = {
            "total_agents": len(all_agents),
            "total_tasks": total_tasks,
            "duration": project_duration,
            "throughput": total_tasks / project_duration
        }
    
    def _get_role_capabilities(self, role: str) -> List[str]:
        """获取角色对应的能力"""
        capability_map = {
            "project_manager": ["management", "coordination", "planning"],
            "team_lead": ["management", "supervision", "communication"],
            "data_analyst": ["data_analysis", "statistics", "reporting"],
            "developer": ["programming", "debugging", "code_review"],
            "tester": ["testing", "quality_assurance", "validation"],
            "deployer": ["deployment", "devops", "monitoring"],
            "ml_engineer": ["machine_learning", "model_training", "optimization"],
            "data_scientist": ["data_analysis", "machine_learning", "research"],
            "ui_designer": ["design", "user_experience", "visualization"]
        }
        return capability_map.get(role, ["task_execution"])
    
    async def run_complete_demo(self):
        """运行完整演示"""
        print("🎬 多智能体协作系统完整演示")
        print("="*60)
        print("本演示将展示多智能体系统的各种功能和协作模式")
        print("="*60)
        
        try:
            # 初始化系统
            await self.initialize()
            
            # 运行各个演示模块
            await self.demo_basic_functionality()
            await asyncio.sleep(0.5)
            
            await self.demo_collaboration_patterns()
            await asyncio.sleep(0.5)
            
            await self.demo_load_balancing()
            await asyncio.sleep(0.5)
            
            await self.demo_performance_monitoring()
            await asyncio.sleep(0.5)
            
            await self.demo_communication_system()
            await asyncio.sleep(0.5)
            
            await self.demo_system_optimization()
            await asyncio.sleep(0.5)
            
            await self.demo_comprehensive_scenario()
            
            # 演示总结
            print("\n" + "="*60)
            print("📋 演示总结")
            print("="*60)
            
            print("\n✅ 演示完成的功能模块:")
            for module, result in self.demo_results.items():
                print(f"   🔹 {module}: {result}")
            
            # 生成演示报告
            demo_report = {
                "timestamp": time.time(),
                "demo_results": self.demo_results,
                "system_status": "演示完成"
            }
            
            # 保存演示报告
            report_file = "demo_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(demo_report, f, ensure_ascii=False, indent=2)
            
            print(f"\n📄 演示报告已保存到: {report_file}")
            
        except Exception as e:
            print(f"\n❌ 演示过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 清理系统
            await self.cleanup()
        
        print("\n🎉 多智能体协作系统演示结束")
        print("感谢您的观看！")


async def main():
    """主函数"""
    demo = MultiAgentDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())