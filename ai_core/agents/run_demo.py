#!/usr/bin/env python3
"""
非阻塞等待系统启动脚本
演示系统的基本功能
"""

import asyncio
import sys
import os

# 添加路径以便导入模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from backend.src.non_blocking import *
from backend.src.non_blocking.examples import NonBlockingSystemDemo


async def demo_basic_functionality():
    """演示基本功能"""
    print("🚀 非阻塞等待系统基本功能演示")
    print("=" * 50)
    
    # 创建配置
    config = NonBlockingConfig()
    print(f"✅ 配置创建完成 - 最大并发任务: {config.max_concurrent_tasks}")
    
    # 创建组件
    task_manager = TaskManager(config)
    progress_pusher = ProgressPusher(task_manager, config)
    result_manager = AsyncResultManager(config)
    status_manager = RealtimeStatusManager(task_manager, config)
    timeout_handler = TimeoutHandler(task_manager, config)
    ux_optimizer = UserExperienceOptimizer(task_manager, config)
    
    print("✅ 所有组件创建完成")
    
    # 启动系统
    print("🔄 启动系统组件...")
    await task_manager.start()
    await progress_pusher.start()
    await result_manager.start()
    await status_manager.start()
    await timeout_handler.start()
    await ux_optimizer.start()
    print("✅ 系统启动完成")
    
    try:
        # 演示1: 基本任务
        print("\n📋 演示1: 基本任务执行")
        async def simple_task():
            await asyncio.sleep(2)
            return "任务完成"
        
        task_id = await task_manager.submit_task(
            name="简单任务",
            func=simple_task,
            correlation_id="demo_user"
        )
        print(f"   任务已提交，ID: {task_id}")
        
        # 等待任务完成
        for i in range(20):
            task = task_manager.get_task(task_id)
            if task.status.value == "completed":
                print(f"   ✅ 任务完成，结果: {task.result}")
                break
            await asyncio.sleep(0.1)
            print(f"   ⏳ 等待中... ({i+1}/20)")
        
        # 演示2: 异步结果
        print("\n📋 演示2: 异步结果获取")
        result_id = await result_manager.create_result("test_task")
        print(f"   异步结果ID: {result_id}")
        
        # 设置结果
        await result_manager.set_result(result_id, {"message": "异步结果数据"})
        
        # 获取结果
        result = result_manager.get_result_sync(result_id)
        print(f"   ✅ 结果获取成功: {result.result}")
        
        # 演示3: 进度跟踪
        print("\n📋 演示3: 进度跟踪")
        async def progress_task():
            tracker = ProgressTracker(task_id, total_steps=5)
            
            for i in range(5):
                await asyncio.sleep(0.5)
                tracker.update(1, f"步骤 {i+1}/5")
            
            tracker.complete("所有步骤完成")
            return "进度任务完成"
        
        progress_task_id = await task_manager.submit_task(
            name="进度跟踪任务",
            func=progress_task,
            correlation_id="demo_user"
        )
        print(f"   进度任务已提交，ID: {progress_task_id}")
        
        # 等待进度任务完成
        for i in range(30):
            task = task_manager.get_task(progress_task_id)
            if task.status.value == "completed":
                print(f"   ✅ 进度任务完成")
                break
            await asyncio.sleep(0.1)
        
        # 演示4: 用户体验优化
        print("\n📋 演示4: 用户体验优化")
        ux_optimizer.set_user_preferences("demo_user", {
            "message_frequency": "high",
            "tone": "encouraging",
            "enable_motivation": True
        })
        print("   ✅ 用户偏好设置完成")
        
        # 发送一些UX消息
        await ux_optimizer.send_progress_update(progress_task_id, 25.0, "进展顺利!", "demo_user")
        await ux_optimizer.send_progress_update(progress_task_id, 50.0, "已完成一半", "demo_user")
        print("   ✅ UX消息发送完成")
        
        # 演示5: 系统统计
        print("\n📋 演示5: 系统统计")
        stats = task_manager.get_stats()
        print(f"   总任务数: {stats['total_tasks']}")
        print(f"   运行中任务: {stats['running_tasks']}")
        print(f"   系统利用率: {stats['utilization']:.2%}")
        
        result_stats = result_manager.get_stats()
        print(f"   结果缓存使用率: {result_stats['cache_utilization']:.2%}")
        
        print("\n🎉 所有演示完成!")
        
    finally:
        # 关闭系统
        print("\n🔄 关闭系统...")
        await ux_optimizer.stop()
        await timeout_handler.stop()
        await status_manager.stop()
        await result_manager.stop()
        await progress_pusher.stop()
        await task_manager.stop()
        print("✅ 系统已关闭")


async def demo_advanced_features():
    """演示高级功能"""
    print("\n🚀 非阻塞等待系统高级功能演示")
    print("=" * 50)
    
    config = NonBlockingConfig()
    task_manager = TaskManager(config)
    progress_pusher = ProgressPusher(task_manager, config)
    result_manager = AsyncResultManager(config)
    status_manager = RealtimeStatusManager(task_manager, config)
    timeout_handler = TimeoutHandler(task_manager, config)
    ux_optimizer = UserExperienceOptimizer(task_manager, config)
    
    # 启动系统
    await asyncio.gather(
        task_manager.start(),
        progress_pusher.start(),
        result_manager.start(),
        status_manager.start(),
        timeout_handler.start(),
        ux_optimizer.start()
    )
    
    try:
        # 演示1: 熔断器
        print("\n📋 演示1: 熔断器")
        circuit_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=2)
        
        async def unstable_service():
            import random
            if random.random() < 0.8:  # 80%概率失败
                raise Exception("服务不可用")
            return "服务正常"
        
        for i in range(5):
            try:
                result = await circuit_breaker.call(unstable_service)
                print(f"   调用 {i+1}: {result}")
            except Exception as e:
                print(f"   调用 {i+1}: 失败 - {str(e)[:30]}...")
        
        # 演示2: 限流器
        print("\n📋 演示2: 限流器")
        rate_limiter = RateLimiter(max_calls=3, time_window=5)
        
        async def limited_service():
            await asyncio.sleep(0.1)
            return "服务调用成功"
        
        for i in range(6):
            if await rate_limiter.acquire():
                result = await limited_service()
                print(f"   调用 {i+1}: {result}")
            else:
                print(f"   调用 {i+1}: 被限流")
            await asyncio.sleep(0.2)
        
        # 演示3: 批处理
        print("\n📋 演示3: 批处理")
        batch_processor = BatchProcessor(task_manager, batch_size=3)
        await batch_processor.start()
        
        async def batch_task(item):
            await asyncio.sleep(0.5)
            return f"处理完成: {item}"
        
        # 添加任务到批处理
        for i in range(8):
            await batch_processor.add_task(batch_task, f"项目{i+1}")
        
        print(f"   已添加8个任务到批处理")
        print(f"   待处理任务数: {batch_processor.get_pending_count()}")
        
        # 等待批处理
        await asyncio.sleep(3)
        print(f"   批处理后待处理任务数: {batch_processor.get_pending_count()}")
        
        await batch_processor.stop()
        
        # 演示4: 健康检查
        print("\n📋 演示4: 健康检查")
        health_checker = HealthChecker(task_manager)
        
        def custom_check():
            return task_manager.config.max_concurrent_tasks > 0
        
        health_checker.add_check("config_check", custom_check)
        health_status = await health_checker.run_checks()
        
        for check_name, result in health_status.items():
            status_icon = "✅" if result["status"] == "healthy" else "❌"
            print(f"   {status_icon} {check_name}: {result['status']}")
        
        print("\n🎉 高级功能演示完成!")
        
    finally:
        await asyncio.gather(
            ux_optimizer.stop(),
            timeout_handler.stop(),
            status_manager.stop(),
            result_manager.stop(),
            progress_pusher.stop(),
            task_manager.stop()
        )


async def run_full_demo():
    """运行完整演示"""
    try:
        await demo_basic_functionality()
        await demo_advanced_features()
        
        print("\n" + "=" * 50)
        print("🎊 非阻塞等待系统演示全部完成!")
        print("=" * 50)
        print("\n📚 系统包含以下功能:")
        print("   ✅ 异步任务管理")
        print("   ✅ 进度推送和预测")
        print("   ✅ 异步结果获取")
        print("   ✅ 实时状态更新")
        print("   ✅ 超时和失败处理")
        print("   ✅ 用户体验优化")
        print("   ✅ 熔断器和限流器")
        print("   ✅ 批处理支持")
        print("   ✅ 健康检查")
        print("   ✅ 完整的API接口")
        print("   ✅ WebSocket支持")
        print("   ✅ 详细的监控统计")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("非阻塞等待系统演示程序")
    print("=" * 50)
    
    # 运行演示
    asyncio.run(run_full_demo())