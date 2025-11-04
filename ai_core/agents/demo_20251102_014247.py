"""
流畅交互系统使用示例
演示如何使用流畅交互管理器实现边做边聊的体验
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from smooth_interaction import (
    SmoothInteractionManager,
    MultiTaskManager,
    ContextManager,
    ProgressTracker,
    InterruptionHandler,
    InteractionFlowOptimizer
)


class SmoothInteractionDemo:
    """流畅交互系统演示"""
    
    def __init__(self):
        self.manager = SmoothInteractionManager()
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """设置回调函数"""
        async def on_state_change(state):
            print(f"状态变化: {state}")
        
        async def on_progress_update(progress_data):
            print(f"进度更新: {progress_data}")
        
        async def on_message(message_data):
            print(f"消息: {message_data}")
        
        self.manager.register_callbacks(
            on_state_change=on_state_change,
            on_progress_update=on_progress_update,
            on_message=on_message
        )
    
    async def demo_basic_interaction(self):
        """演示基本交互"""
        print("\n=== 基本交互演示 ===")
        
        # 开始交互会话
        session_id = await self.manager.start_interaction(
            user_id="user_123",
            initial_context={
                'user_name': '张三',
                'preferences': {
                    'language': 'zh-CN',
                    'notification': True
                }
            }
        )
        print(f"创建会话: {session_id}")
        
        # 处理用户消息
        response = await self.manager.process_message(
            session_id=session_id,
            message="请帮我分析一下销售数据",
            metadata={'source': 'web'}
        )
        print(f"响应: {response}")
        
        # 获取会话状态
        status = await self.manager.get_session_status(session_id)
        print(f"会话状态: {json.dumps(status, ensure_ascii=False, indent=2)}")
        
        # 结束会话
        report = await self.manager.end_interaction(session_id)
        print(f"会话报告: {json.dumps(report, ensure_ascii=False, indent=2)}")
    
    async def demo_task_management(self):
        """演示任务管理"""
        print("\n=== 任务管理演示 ===")
        
        session_id = await self.manager.start_interaction("user_456")
        
        # 创建多个任务
        task1_id = await self.manager.task_manager.create_task(
            session_id=session_id,
            task_type="data_analysis",
            description="分析销售数据",
            priority="high",
            estimated_duration=30
        )
        print(f"创建任务1: {task1_id}")
        
        task2_id = await self.manager.task_manager.create_task(
            session_id=session_id,
            task_type="report_generation",
            description="生成分析报告",
            priority="normal",
            dependencies=[task1_id],
            estimated_duration=20
        )
        print(f"创建任务2: {task2_id}")
        
        # 启动任务
        await self.manager.task_manager.start_task(task1_id)
        await self.manager.task_manager.start_task(task2_id)
        
        # 模拟任务执行
        await asyncio.sleep(1)
        
        # 获取任务进度
        progress1 = await self.manager.get_task_progress(session_id, task1_id)
        progress2 = await self.manager.get_task_progress(session_id, task2_id)
        
        print(f"任务1进度: {progress1}")
        print(f"任务2进度: {progress2}")
        
        # 中断任务1
        success = await self.manager.interrupt_task(session_id, task1_id, "用户要求中断")
        print(f"中断任务1: {success}")
        
        # 恢复任务1
        success = await self.manager.resume_task(session_id, task1_id)
        print(f"恢复任务1: {success}")
        
        await self.manager.end_interaction(session_id)
    
    async def demo_context_management(self):
        """演示上下文管理"""
        print("\n=== 上下文管理演示 ===")
        
        session_id = await self.manager.start_interaction("user_789")
        
        # 更新上下文
        await self.manager.context_manager.update_context(
            session_id=session_id,
            updates={
                'user_preference': {
                    'theme': 'dark',
                    'language': 'zh-CN'
                },
                'current_task': {
                    'type': 'analysis',
                    'status': 'in_progress'
                }
            }
        )
        
        # 添加对话轮次
        await self.manager.context_manager.add_conversation_turn(
            session_id=session_id,
            user_message="请分析这些数据",
            assistant_response="好的，正在分析数据，请稍等..."
        )
        
        # 获取上下文
        context = await self.manager.context_manager.get_context(session_id)
        print(f"当前上下文: {json.dumps(context, ensure_ascii=False, indent=2)}")
        
        # 保存快照
        await self.manager.context_manager.save_context_snapshot(
            session_id=session_id,
            snapshot_name="analysis_session"
        )
        
        # 获取上下文摘要
        summary = await self.manager.context_manager.get_context_summary(session_id)
        print(f"上下文摘要: {json.dumps(summary, ensure_ascii=False, indent=2)}")
        
        await self.manager.end_interaction(session_id)
    
    async def demo_progress_tracking(self):
        """演示进度跟踪"""
        print("\n=== 进度跟踪演示 ===")
        
        session_id = await self.manager.start_interaction("user_101")
        
        # 开始跟踪任务
        task_id = await self.manager.task_manager.create_task(
            session_id=session_id,
            task_type="data_processing",
            description="处理大量数据",
            estimated_duration=10
        )
        
        await self.manager.progress_tracker.start_tracking(
            task_id=task_id,
            session_id=session_id,
            estimated_duration=10
        )
        
        # 模拟进度更新
        for progress in [20, 40, 60, 80, 100]:
            await self.manager.progress_tracker.update_progress(
                task_id=task_id,
                progress=progress,
                message=f"处理进度: {progress}%"
            )
            await asyncio.sleep(0.5)
        
        # 获取进度
        progress = await self.manager.progress_tracker.get_progress(task_id)
        print(f"最终进度: {json.dumps(progress, ensure_ascii=False, indent=2)}")
        
        # 获取历史
        history = await self.manager.progress_tracker.get_progress_history(task_id)
        print(f"进度历史: {len(history)} 条记录")
        
        await self.manager.end_interaction(session_id)
    
    async def demo_interruption_handling(self):
        """演示中断处理"""
        print("\n=== 中断处理演示 ===")
        
        session_id = await self.manager.start_interaction("user_202")
        
        # 创建长时间运行的任务
        task_id = await self.manager.task_manager.create_task(
            session_id=session_id,
            task_type="long_running_task",
            description="执行长时间任务",
            estimated_duration=60
        )
        
        await self.manager.task_manager.start_task(task_id)
        
        # 模拟用户中断请求
        response = await self.manager.process_message(
            session_id=session_id,
            message="停止当前任务",
            metadata={'intent': 'interrupt'}
        )
        print(f"中断响应: {response}")
        
        # 检查中断状态
        interrupt_status = await self.manager.interruption_handler.get_interrupt_status(
            response.get('request_id', '')
        )
        if interrupt_status:
            print(f"中断状态: {json.dumps(interrupt_status, ensure_ascii=False, indent=2)}")
        
        await self.manager.end_interaction(session_id)
    
    async def demo_flow_optimization(self):
        """演示流程优化"""
        print("\n=== 流程优化演示 ===")
        
        session_id = await self.manager.start_interaction("user_303")
        
        # 模拟多次交互
        messages = [
            "请帮我处理数据",
            "这个任务很紧急",
            "能详细说明一下吗",
            "还有其他选择吗",
            "很好，继续"
        ]
        
        for message in messages:
            response = await self.manager.process_message(
                session_id=session_id,
                message=message
            )
            print(f"用户: {message}")
            print(f"助手: {response.get('response', '')}")
            print(f"优化策略: {response.get('optimization_applied', [])}")
            print("---")
            
            await asyncio.sleep(0.1)
        
        # 获取建议
        suggestions = await self.manager.flow_optimizer.get_suggestions(session_id)
        print(f"智能建议: {json.dumps(suggestions, ensure_ascii=False, indent=2)}")
        
        # 获取流程分析
        analytics = await self.manager.flow_optimizer.get_flow_analytics(session_id)
        print(f"流程分析: {json.dumps(analytics, ensure_ascii=False, indent=2)}")
        
        await self.manager.end_interaction(session_id)
    
    async def demo_comprehensive_workflow(self):
        """演示完整工作流"""
        print("\n=== 完整工作流演示 ===")
        
        session_id = await self.manager.start_interaction("user_404")
        
        try:
            # 1. 用户提出需求
            response = await self.manager.process_message(
                session_id=session_id,
                message="我需要分析这个月的销售数据并生成报告",
                metadata={'priority': 'high'}
            )
            print(f"需求响应: {response}")
            
            # 2. 创建分析任务
            analysis_task = await self.manager.task_manager.create_task(
                session_id=session_id,
                task_type="sales_analysis",
                description="分析销售数据",
                priority="high",
                estimated_duration=30
            )
            
            # 3. 开始任务和进度跟踪
            await self.manager.task_manager.start_task(analysis_task)
            await self.manager.progress_tracker.start_tracking(
                task_id=analysis_task,
                session_id=session_id,
                estimated_duration=30
            )
            
            # 4. 模拟任务执行和进度更新
            for i in range(5):
                progress = (i + 1) * 20
                await self.manager.progress_tracker.update_progress(
                    task_id=analysis_task,
                    progress=progress,
                    message=f"正在分析数据... {progress}%"
                )
                await asyncio.sleep(0.3)
            
            # 5. 创建报告生成任务（依赖分析任务）
            report_task = await self.manager.task_manager.create_task(
                session_id=session_id,
                task_type="report_generation",
                description="生成销售报告",
                priority="normal",
                dependencies=[analysis_task],
                estimated_duration=15
            )
            
            await self.manager.task_manager.start_task(report_task)
            
            # 6. 用户中途询问进度
            progress_response = await self.manager.process_message(
                session_id=session_id,
                message="任务进展如何？"
            )
            print(f"进度询问响应: {progress_response}")
            
            # 7. 获取当前所有任务状态
            session_status = await self.manager.get_session_status(session_id)
            print(f"当前任务状态: {len(session_status['active_tasks'])} 个活跃任务")
            
            # 8. 模拟用户中断报告生成任务
            if session_status['active_tasks']:
                report_task_id = session_status['active_tasks'][0]['task_id']
                await self.manager.interrupt_task(
                    session_id=session_id,
                    task_id=report_task_id,
                    reason="用户要求暂停报告生成"
                )
            
            # 9. 用户要求恢复
            await asyncio.sleep(0.5)
            await self.manager.resume_task(session_id, report_task_id)
            
            # 10. 任务完成
            await self.manager.progress_tracker.complete_task(
                task_id=report_task_id,
                final_message="报告生成完成！"
            )
            
            # 11. 获取最终结果
            final_status = await self.manager.get_session_status(session_id)
            print(f"最终状态: 任务完成数 = {len(final_status['active_tasks'])}")
            
        except Exception as e:
            print(f"工作流执行出错: {e}")
        
        finally:
            # 12. 结束会话并获取报告
            report = await self.manager.end_interaction(session_id)
            print(f"会话总结: {json.dumps(report, ensure_ascii=False, indent=2)}")
    
    async def demo_system_monitoring(self):
        """演示系统监控"""
        print("\n=== 系统监控演示 ===")
        
        # 获取系统状态
        system_status = await self.manager.get_system_status()
        print(f"系统状态: {json.dumps(system_status, ensure_ascii=False, indent=2)}")
        
        # 获取各组件统计
        task_stats = await self.manager.task_manager.get_system_load()
        context_stats = await self.manager.context_manager.get_stats()
        progress_stats = await self.manager.progress_tracker.get_stats()
        interrupt_stats = await self.manager.interruption_handler.get_stats()
        flow_stats = await self.manager.flow_optimizer.get_optimization_stats()
        
        print(f"\n任务管理统计: {json.dumps(task_stats, ensure_ascii=False, indent=2)}")
        print(f"\n上下文管理统计: {json.dumps(context_stats, ensure_ascii=False, indent=2)}")
        print(f"\n进度跟踪统计: {json.dumps(progress_stats, ensure_ascii=False, indent=2)}")
        print(f"\n中断处理统计: {json.dumps(interrupt_stats, ensure_ascii=False, indent=2)}")
        print(f"\n流程优化统计: {json.dumps(flow_stats, ensure_ascii=False, indent=2)}")
    
    async def run_all_demos(self):
        """运行所有演示"""
        print("🚀 流畅交互系统演示开始")
        print("=" * 50)
        
        try:
            await self.demo_basic_interaction()
            await self.demo_task_management()
            await self.demo_context_management()
            await self.demo_progress_tracking()
            await self.demo_interruption_handling()
            await self.demo_flow_optimization()
            await self.demo_comprehensive_workflow()
            await self.demo_system_monitoring()
            
        except Exception as e:
            print(f"演示过程中出现错误: {e}")
        
        print("\n" + "=" * 50)
        print("✅ 流畅交互系统演示完成")


async def main():
    """主函数"""
    demo = SmoothInteractionDemo()
    await demo.run_all_demos()


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())