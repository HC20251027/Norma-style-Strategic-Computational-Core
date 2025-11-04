"""
异步任务机制使用示例

演示如何使用异步任务管理器的各种功能
"""

import asyncio
import time
import random
from typing import List, Dict, Any

# 导入异步任务机制组件
from .task_manager import AsyncTaskManager, create_task_manager
from .task_models import TaskPriority, TaskStatus
from .scheduler import SchedulingStrategy
from .cache_manager import CacheStrategy
from .config import AsyncTaskConfig, ConfigTemplates
from .utils import (
    async_task, retry_on_failure, timeout, progress_tracker,
    measure_time, batch_process, rate_limit, memoize
)


# 示例任务函数
async def simple_async_task(name: str, duration: float = 1.0):
    """简单的异步任务"""
    print(f"开始执行任务: {name}")
    await asyncio.sleep(duration)
    print(f"任务完成: {name}")
    return f"结果来自 {name}"


def simple_sync_task(name: str, duration: float = 1.0):
    """简单的同步任务"""
    print(f"开始执行同步任务: {name}")
    time.sleep(duration)
    print(f"同步任务完成: {name}")
    return f"同步结果来自 {name}"


@retry_on_failure(max_retries=3, delay=1.0)
async def unreliable_task(name: str, fail_probability: float = 0.5):
    """不可靠的任务（用于演示重试机制）"""
    print(f"执行不可靠任务: {name}")
    
    if random.random() < fail_probability:
        raise Exception(f"任务 {name} 随机失败")
    
    await asyncio.sleep(0.5)
    return f"不可靠任务 {name} 成功完成"


@timeout(seconds=2.0)
async def timed_task(name: str, duration: float = 3.0):
    """超时任务"""
    print(f"执行超时任务: {name}")
    await asyncio.sleep(duration)
    return f"超时任务 {name} 完成"


@progress_tracker
async def data_processing_task(data: List[int], name: str = "数据处理"):
    """带进度跟踪的数据处理任务"""
    total = len(data)
    processed = []
    
    for i, item in enumerate(data):
        # 模拟处理过程
        await asyncio.sleep(0.1)
        processed.append(item * 2)
        
        # 更新进度
        progress = (i + 1) / total * 100
        print(f"处理进度: {progress:.1f}%")
    
    return {
        "task_name": name,
        "input_count": total,
        "output_count": len(processed),
        "result": processed
    }


@measure_time
async def performance_task(name: str, iterations: int = 1000):
    """性能测试任务"""
    result = 0
    for i in range(iterations):
        result += i ** 2
        await asyncio.sleep(0.001)  # 模拟一些工作
    
    return f"性能任务 {name} 完成，计算结果: {result}"


@rate_limit(calls_per_second=2.0)
async def rate_limited_task(name: str):
    """速率限制任务"""
    print(f"执行速率限制任务: {name}")
    await asyncio.sleep(0.1)
    return f"速率限制任务 {name} 完成"


@memoize(ttl=5.0)
async def expensive_computation(x: int, y: int) -> int:
    """昂贵的计算任务（带缓存）"""
    print(f"执行昂贵计算: {x} + {y}")
    await asyncio.sleep(1.0)  # 模拟昂贵操作
    return x + y


async def batch_data_processing(data_list: List[Dict[str, Any]]):
    """批量数据处理任务"""
    @batch_process(batch_size=5, max_workers=3)
    async def process_item(item: Dict[str, Any]):
        await asyncio.sleep(0.2)  # 模拟处理时间
        return {
            "id": item["id"],
            "processed": True,
            "value": item["value"] * 2
        }
    
    results = await process_item(data_list)
    return results


async def dependency_task_chain():
    """依赖任务链示例"""
    manager = create_task_manager()
    await manager.start()
    
    try:
        # 创建依赖任务链
        task1_id = await manager.submit_task(
            simple_async_task,
            "任务1",
            name="准备数据",
            priority=TaskPriority.HIGH
        )
        
        task2_id = await manager.submit_task(
            simple_async_task,
            "任务2",
            name="处理数据",
            priority=TaskPriority.NORMAL,
            dependencies=[task1_id]
        )
        
        task3_id = await manager.submit_task(
            simple_async_task,
            "任务3",
            name="保存结果",
            priority=TaskPriority.HIGH,
            dependencies=[task2_id]
        )
        
        print(f"提交了依赖任务链: {task1_id} -> {task2_id} -> {task3_id}")
        
        # 等待所有任务完成
        results = []
        for task_id in [task1_id, task2_id, task3_id]:
            result = await manager.wait_for_task(task_id, timeout=30)
            if result:
                results.append(result.result)
        
        return results
        
    finally:
        await manager.stop()


async def advanced_features_demo():
    """高级功能演示"""
    # 创建配置
    config = ConfigTemplates.production()
    config.max_concurrent = 5
    config.cache_enabled = True
    config.monitoring_enabled = True
    
    # 创建任务管理器
    manager = AsyncTaskManager(
        max_concurrent=config.max_concurrent,
        cache_enabled=config.cache_enabled,
        cache_size=config.cache_size,
        storage_enabled=True,
        storage_type="sqlite",
        monitoring_enabled=config.monitoring_enabled
    )
    
    await manager.start()
    
    try:
        # 提交各种类型的任务
        task_ids = []
        
        # 1. 简单异步任务
        task_id = await manager.submit_task(
            simple_async_task,
            "简单任务",
            duration=1.0,
            name="简单异步任务",
            priority=TaskPriority.NORMAL
        )
        task_ids.append(task_id)
        
        # 2. 同步任务
        task_id = await manager.submit_task(
            simple_sync_task,
            "同步任务",
            duration=0.5,
            name="同步任务示例",
            priority=TaskPriority.LOW
        )
        task_ids.append(task_id)
        
        # 3. 带重试的任务
        task_id = await manager.submit_task(
            unreliable_task,
            "重试任务",
            fail_probability=0.7,
            name="重试机制演示",
            retry_count=5,
            priority=TaskPriority.HIGH
        )
        task_ids.append(task_id)
        
        # 4. 数据处理任务
        data = list(range(20))
        task_id = await manager.submit_task(
            data_processing_task,
            data,
            name="数据处理",
            metadata={"batch_size": len(data)}
        )
        task_ids.append(task_id)
        
        # 5. 性能测试任务
        task_id = await manager.submit_task(
            performance_task,
            "性能测试",
            iterations=500,
            name="性能测试任务",
            timeout=30
        )
        task_ids.append(task_id)
        
        # 6. 速率限制任务
        for i in range(5):
            task_id = await manager.submit_task(
                rate_limited_task,
                f"速率限制任务{i+1}",
                name=f"速率限制{i+1}"
            )
            task_ids.append(task_id)
        
        # 7. 缓存演示任务
        for i in range(3):
            task_id = await manager.submit_task(
                expensive_computation,
                10, 20,
                name=f"昂贵计算{i+1}"
            )
            task_ids.append(task_id)
        
        print(f"提交了 {len(task_ids)} 个任务")
        
        # 等待所有任务完成并收集结果
        results = []
        for task_id in task_ids:
            result = await manager.wait_for_task(task_id, timeout=60)
            if result:
                results.append({
                    "task_id": task_id,
                    "status": result.status.value,
                    "result": result.result
                })
        
        # 获取队列状态
        status = await manager.get_queue_status()
        print(f"队列状态: {status}")
        
        # 获取性能统计
        if manager.get_monitor():
            stats = await manager.get_monitor().get_performance_stats()
            print(f"性能统计: {stats}")
        
        return results
        
    finally:
        await manager.stop()


async def monitoring_demo():
    """监控功能演示"""
    manager = create_task_manager(max_concurrent=3, monitoring_enabled=True)
    await manager.start()
    
    try:
        # 提交一些任务
        task_ids = []
        for i in range(10):
            task_id = await manager.submit_task(
                simple_async_task,
                f"监控任务{i+1}",
                duration=random.uniform(0.5, 2.0),
                name=f"监控演示{i+1}"
            )
            task_ids.append(task_id)
        
        # 监控任务执行
        monitor = manager.get_monitor()
        if monitor:
            # 添加告警回调
            async def alert_handler(alert):
                print(f"告警: {alert.level} - {alert.message}")
            
            monitor.add_alert_callback(alert_handler)
            
            # 等待一段时间观察监控数据
            for _ in range(20):
                await asyncio.sleep(1)
                
                # 获取系统指标
                system_metrics = await monitor.get_system_metrics()
                if system_metrics:
                    print(f"CPU: {system_metrics.cpu_percent:.1f}%, "
                          f"内存: {system_metrics.memory_percent:.1f}%")
                
                # 检查是否有活跃告警
                active_alerts = await monitor.get_active_alerts()
                if active_alerts:
                    print(f"活跃告警数量: {len(active_alerts)}")
        
        # 等待所有任务完成
        for task_id in task_ids:
            await manager.wait_for_task(task_id, timeout=30)
        
        # 导出监控数据
        if monitor:
            await monitor.export_metrics("monitoring_data.json")
            print("监控数据已导出到 monitoring_data.json")
        
    finally:
        await manager.stop()


async def cache_demo():
    """缓存功能演示"""
    manager = create_task_manager(cache_enabled=True, cache_size=100)
    await manager.start()
    
    try:
        # 提交相同的计算任务（应该从缓存获取结果）
        task_ids = []
        for i in range(5):
            task_id = await manager.submit_task(
                expensive_computation,
                5, 15,
                name=f"缓存测试{i+1}"
            )
            task_ids.append(task_id)
        
        # 等待结果
        results = []
        for task_id in task_ids:
            result = await manager.wait_for_task(task_id, timeout=30)
            if result:
                results.append(result.result)
        
        # 检查缓存统计
        cache_manager = manager.get_cache_manager()
        if cache_manager:
            stats = await cache_manager.get_stats()
            print(f"缓存统计: {stats}")
        
        print(f"所有结果: {results}")
        
    finally:
        await manager.stop()


async def storage_demo():
    """存储功能演示"""
    manager = create_task_manager(storage_enabled=True, storage_path="demo_tasks.db")
    await manager.start()
    
    try:
        # 提交一些任务
        task_ids = []
        for i in range(5):
            task_id = await manager.submit_task(
                simple_async_task,
                f"存储测试{i+1}",
                duration=1.0,
                name=f"存储任务{i+1}"
            )
            task_ids.append(task_id)
        
        # 等待完成
        for task_id in task_ids:
            await manager.wait_for_task(task_id, timeout=30)
        
        # 导出任务
        await manager.export_tasks("exported_tasks.json")
        print("任务已导出到 exported_tasks.json")
        
        # 创建备份
        backup_path = await manager.create_backup()
        print(f"备份已创建: {backup_path}")
        
        # 列出所有任务
        all_tasks = await manager.list_tasks()
        print(f"总任务数: {len(all_tasks)}")
        
        # 按状态统计
        for status in TaskStatus:
            tasks = await manager.list_tasks(status=status)
            print(f"{status.value}: {len(tasks)} 个任务")
        
    finally:
        await manager.stop()


async def main():
    """主函数 - 运行所有示例"""
    print("=== 异步任务机制示例演示 ===\n")
    
    examples = [
        ("简单任务示例", simple_async_task_example),
        ("依赖任务链", dependency_task_chain),
        ("高级功能演示", advanced_features_demo),
        ("监控功能演示", monitoring_demo),
        ("缓存功能演示", cache_demo),
        ("存储功能演示", storage_demo)
    ]
    
    for name, func in examples:
        print(f"\n{'='*50}")
        print(f"运行示例: {name}")
        print('='*50)
        
        try:
            await func()
            print(f"示例 '{name}' 完成")
        except Exception as e:
            print(f"示例 '{name}' 失败: {e}")
        
        print(f"示例 '{name}' 结束\n")


async def simple_async_task_example():
    """简单的异步任务示例"""
    manager = create_task_manager(max_concurrent=3)
    await manager.start()
    
    try:
        # 提交任务
        task_id = await manager.submit_task(
            simple_async_task,
            "示例任务",
            duration=2.0,
            name="简单任务示例",
            priority=TaskPriority.NORMAL
        )
        
        print(f"任务已提交: {task_id}")
        
        # 等待完成
        result = await manager.wait_for_task(task_id, timeout=30)
        
        if result:
            print(f"任务完成，结果: {result.result}")
        else:
            print("任务超时或失败")
        
        # 获取队列状态
        status = await manager.get_queue_status()
        print(f"队列状态: {status}")
        
    finally:
        await manager.stop()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())