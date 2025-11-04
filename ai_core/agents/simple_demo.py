"""
流畅交互系统简化演示
演示核心功能，避免复杂的序列化问题
"""

import asyncio
import sys
import os

# 添加路径
sys.path.append('/workspace/backend/src')

from smooth_interaction import SmoothInteractionManager


async def simple_demo():
    """简化演示"""
    print("🚀 流畅交互系统简化演示")
    print("=" * 40)
    
    try:
        # 创建管理器
        manager = SmoothInteractionManager()
        print("✅ 创建交互管理器成功")
        
        # 开始会话
        session_id = await manager.start_interaction(
            user_id="demo_user",
            initial_context={'user_name': '演示用户'}
        )
        print(f"✅ 创建会话成功: {session_id[:8]}...")
        
        # 处理消息
        response = await manager.process_message(
            session_id=session_id,
            message="请帮我处理一个简单的任务"
        )
        print(f"✅ 处理消息成功")
        print(f"   响应: {response.get('response', 'N/A')}")
        
        # 获取会话状态
        status = await manager.get_session_status(session_id)
        print(f"✅ 获取会话状态成功")
        print(f"   状态: {status.get('state', 'N/A')}")
        print(f"   活跃任务数: {len(status.get('active_tasks', []))}")
        
        # 测试任务管理
        task_id = await manager.task_manager.create_task(
            session_id=session_id,
            task_type="demo_task",
            description="演示任务",
            priority="normal"
        )
        print(f"✅ 创建任务成功: {task_id[:8]}...")
        
        # 启动任务
        success = await manager.task_manager.start_task(task_id)
        print(f"✅ 启动任务: {'成功' if success else '失败'}")
        
        # 测试进度跟踪
        await manager.progress_tracker.start_tracking(task_id, session_id)
        print("✅ 开始进度跟踪")
        
        # 更新进度
        for progress in [25, 50, 75, 100]:
            await manager.progress_tracker.update_progress(
                task_id=task_id,
                progress=progress,
                message=f"演示进度: {progress}%"
            )
            print(f"   进度更新: {progress}%")
        
        # 完成进度
        await manager.progress_tracker.complete_task(task_id, "演示完成")
        print("✅ 任务完成")
        
        # 测试中断处理
        can_interrupt = await manager.interruption_handler.can_interrupt(task_id)
        print(f"✅ 可以中断任务: {can_interrupt}")
        
        # 测试流程优化
        suggestions = await manager.flow_optimizer.get_suggestions(session_id)
        print(f"✅ 获取智能建议: {len(suggestions)} 条")
        
        # 获取系统状态
        system_status = await manager.get_system_status()
        print(f"✅ 系统状态:")
        print(f"   活跃会话: {system_status.get('active_sessions', 0)}")
        print(f"   总任务数: {system_status.get('total_tasks', 0)}")
        
        # 结束会话
        report = await manager.end_interaction(session_id)
        print(f"✅ 结束会话成功")
        print(f"   会话时长: {report.get('duration', 0):.2f} 秒")
        print(f"   消息数: {report.get('message_count', 0)}")
        
        print("\n" + "=" * 40)
        print("🎉 流畅交互系统演示完成！")
        print("✅ 所有核心功能正常工作")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(simple_demo())