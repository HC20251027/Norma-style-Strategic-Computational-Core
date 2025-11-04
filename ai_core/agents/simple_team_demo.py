#!/usr/bin/env python3
"""
诺玛专业智能体团队演示脚本
"""

import asyncio
import sys
import os

# 添加当前路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def demo_team():
    """演示团队功能"""
    print("🎯 诺玛专业智能体团队演示")
    print("=" * 50)
    
    try:
        from norma_professional_agents_team import NormaProfessionalTeam
        
        # 初始化团队
        print("🚀 初始化专业智能体团队...")
        team = NormaProfessionalTeam()
        
        # 获取团队状态
        print("\n📊 团队状态:")
        status = team.get_team_status()
        print(f"  团队ID: {status['team_id']}")
        print(f"  团队状态: {status['team_status']}")
        print(f"  Agent数量: {len(status['agents'])}")
        
        # 显示各Agent信息
        print("\n🤖 专业智能体团队:")
        for agent_name, agent_info in status['agents'].items():
            print(f"  • {agent_name}: {agent_info['status']} (能力: {agent_info['capabilities_count']})")
        
        # 演示任务执行
        print("\n📋 执行演示任务...")
        demo_task = {
            "title": "诺玛Agent系统优化咨询",
            "description": "需要对系统进行全面的性能分析和优化建议",
            "complexity": "high"
        }
        
        result = await team.execute_team_task(demo_task)
        
        if result.get('success', False):
            print("✅ 任务执行成功!")
            print(f"  团队效率: {result['team_performance']['coordination_efficiency']:.2%}")
            print(f"  完成率: {result['team_performance']['task_completion_rate']:.2%}")
            print(f"  质量分数: {result['team_performance']['average_quality_score']:.2%}")
        else:
            print(f"❌ 任务执行失败: {result.get('error', 'Unknown error')}")
        
        print("\n🎉 团队演示完成!")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demo_team())
