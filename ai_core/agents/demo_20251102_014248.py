#!/usr/bin/env python3
"""
语音流水线演示脚本

快速演示语音-文本-工具流水线的核心功能
"""

import asyncio
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入流水线组件
from voice_pipeline import VoicePipelineManager, PipelineRequest
from voice_pipeline.config import get_test_config


async def demo_basic_pipeline():
    """演示基本流水线功能"""
    print("=" * 60)
    print("🎤 语音-文本-工具流水线演示")
    print("=" * 60)
    
    # 1. 初始化流水线
    print("\n📋 步骤 1: 初始化流水线...")
    config = get_test_config()
    pipeline = VoicePipelineManager(config)
    await pipeline.start()
    print("✅ 流水线初始化完成")
    
    try:
        # 2. 演示不同类型的请求
        test_cases = [
            {
                "name": "天气查询",
                "audio_data": b"北京今天天气怎么样",
                "expected_tools": ["weather"]
            },
            {
                "name": "时间查询",
                "audio_data": b"现在几点了",
                "expected_tools": ["time"]
            },
            {
                "name": "计算请求",
                "audio_data": b"计算一下 2 加 3 等于多少",
                "expected_tools": ["calculator"]
            },
            {
                "name": "系统信息",
                "audio_data": b"查看系统状态",
                "expected_tools": ["system_info"]
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔄 步骤 {i + 1}: 处理 {test_case['name']}...")
            
            # 创建请求
            request = PipelineRequest(
                audio_data=test_case["audio_data"],
                language="zh-CN",
                metadata={
                    "test_case": test_case['name'],
                    "demo": True
                }
            )
            
            # 处理请求
            print(f"   正在处理 {test_case['name']} 请求...")
            response = await pipeline.process_request(request)
            
            # 显示结果
            print(f"   ✅ 处理完成!")
            print(f"   📝 文本响应: {response.text_response}")
            print(f"   ⏱️  处理时间: {response.processing_time:.2f}秒")
            print(f"   🎵 音频长度: {len(response.audio_response) if response.audio_response else 0} 字节")
            
            if response.error_message:
                print(f"   ⚠️  错误信息: {response.error_message}")
        
        # 3. 显示性能指标
        print(f"\n📊 步骤 {len(test_cases) + 2}: 性能指标...")
        metrics = await pipeline.get_metrics()
        
        print(f"   📈 总请求数: {metrics['pipeline_metrics']['total_requests']}")
        print(f"   ✅ 成功请求: {metrics['pipeline_metrics']['successful_requests']}")
        print(f"   ❌ 失败请求: {metrics['pipeline_metrics']['failed_requests']}")
        print(f"   ⏱️  平均处理时间: {metrics['pipeline_metrics']['average_processing_time']:.2f}秒")
        print(f"   🔄 活跃请求: {metrics['active_requests']}")
        
        # 4. 显示可用工具
        print(f"\n🛠️  步骤 {len(test_cases) + 3}: 可用工具...")
        tools = await pipeline.get_available_tools()
        print(f"   📦 总工具数: {len(tools)}")
        for tool_name, tool_def in tools.items():
            print(f"   • {tool_name}: {tool_def['description']}")
        
        # 5. 演示状态监控
        print(f"\n📋 步骤 {len(test_cases) + 4}: 状态监控...")
        if metrics['pipeline_metrics']['total_requests'] > 0:
            # 获取最后一个请求的状态
            last_request_id = None
            # 这里简化处理，实际中应该跟踪请求ID
            
            if last_request_id:
                status = await pipeline.get_request_status(last_request_id)
                if status:
                    print(f"   📊 请求状态:")
                    print(f"      ID: {status['request_id']}")
                    print(f"      状态: {status['status']}")
                    print(f"      阶段: {status['current_stage']}")
                    print(f"      进度: {status['progress']:.1%}")
        
        print(f"\n🎉 演示完成!")
        
    finally:
        # 停止流水线
        print("\n🛑 停止流水线...")
        await pipeline.stop()
        print("✅ 流水线已停止")


async def demo_error_handling():
    """演示错误处理"""
    print("\n" + "=" * 60)
    print("🛡️ 错误处理演示")
    print("=" * 60)
    
    # 配置错误处理
    from voice_pipeline.config import PipelineConfig
    config = PipelineConfig(
        error_handling={
            "max_retries": 2,
            "retry_delay": 0.5,
            "fallback_enabled": True
        }
    )
    
    pipeline = VoicePipelineManager(config)
    await pipeline.start()
    
    try:
        print("\n🔄 演示错误处理和重试机制...")
        
        # 创建可能导致错误的请求
        error_request = PipelineRequest(
            audio_data=b"invalid_audio_data_that_will_cause_error",
            language="invalid_language",
            metadata={"test_error": True}
        )
        
        print("   发送可能导致错误的请求...")
        response = await pipeline.process_request(error_request)
        
        print(f"   📝 响应: {response.text_response}")
        print(f"   ✅ 状态: {response.status.value}")
        print(f"   ⏱️  处理时间: {response.processing_time:.2f}秒")
        
        if response.error_message:
            print(f"   ⚠️  错误信息: {response.error_message}")
        
        # 显示错误统计
        metrics = await pipeline.get_metrics()
        error_stats = metrics.get('error_statistics', {})
        
        if error_stats['total_errors'] > 0:
            print(f"\n📊 错误统计:")
            for error_type, count in error_stats['error_breakdown'].items():
                print(f"   • {error_type}: {count} 次")
        
        print("\n✅ 错误处理演示完成")
        
    finally:
        await pipeline.stop()


async def demo_tool_management():
    """演示工具管理"""
    print("\n" + "=" * 60)
    print("🛠️ 工具管理演示")
    print("=" * 60)
    
    config = get_test_config()
    pipeline = VoicePipelineManager(config)
    await pipeline.start()
    
    try:
        # 1. 显示当前工具
        print("\n📋 当前可用工具:")
        tools = await pipeline.get_available_tools()
        for tool_name, tool_def in tools.items():
            print(f"   • {tool_name}: {tool_def['description']}")
        
        # 2. 添加自定义工具
        print("\n➕ 添加自定义工具...")
        custom_tool = {
            "description": "获取随机笑话",
            "parameters": {
                "category": {"type": "str", "required": False}
            },
            "keywords": ["笑话", "幽默", "搞笑", "娱乐"]
        }
        
        await pipeline.add_custom_tool("joke", custom_tool)
        print("   ✅ 已添加工具: joke")
        
        # 3. 验证工具已添加
        print("\n🔍 验证工具添加:")
        tools = await pipeline.get_available_tools()
        if "joke" in tools:
            print("   ✅ 工具 'joke' 已成功添加")
            print(f"   📝 描述: {tools['joke']['description']}")
        
        # 4. 测试自定义工具
        print("\n🧪 测试自定义工具...")
        joke_request = PipelineRequest(
            audio_data=b"给我讲个笑话",
            language="zh-CN",
            metadata={"test_custom_tool": True}
        )
        
        response = await pipeline.process_request(joke_request)
        print(f"   📝 响应: {response.text_response}")
        
        # 5. 移除自定义工具
        print("\n➖ 移除自定义工具...")
        removed = await pipeline.remove_tool("joke")
        if removed:
            print("   ✅ 工具 'joke' 已成功移除")
        
        print("\n✅ 工具管理演示完成")
        
    finally:
        await pipeline.stop()


async def demo_performance_monitoring():
    """演示性能监控"""
    print("\n" + "=" * 60)
    print("📊 性能监控演示")
    print("=" * 60)
    
    config = get_test_config()
    pipeline = VoicePipelineManager(config)
    await pipeline.start()
    
    try:
        # 1. 批量处理请求
        print("\n🔄 批量处理请求...")
        batch_size = 3
        
        for i in range(batch_size):
            request = PipelineRequest(
                audio_data=f"batch_test_{i}".encode(),
                language="zh-CN",
                metadata={
                    "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "batch_index": i
                }
            )
            
            print(f"   处理请求 {i + 1}/{batch_size}...")
            response = await pipeline.process_request(request)
            print(f"   ✅ 完成: {response.success}, 时间: {response.processing_time:.2f}秒")
        
        # 2. 获取详细指标
        print("\n📈 获取性能指标...")
        metrics = await pipeline.get_metrics()
        
        print(f"   📊 流水线指标:")
        print(f"      总请求数: {metrics['pipeline_metrics']['total_requests']}")
        print(f"      成功请求: {metrics['pipeline_metrics']['successful_requests']}")
        print(f"      失败请求: {metrics['pipeline_metrics']['failed_requests']}")
        print(f"      平均处理时间: {metrics['pipeline_metrics']['average_processing_time']:.2f}秒")
        print(f"      活跃请求: {metrics['active_requests']}")
        print(f"      运行状态: {'运行中' if metrics['is_running'] else '已停止'}")
        
        # 3. 显示错误统计
        error_stats = metrics.get('error_statistics', {})
        if error_stats['total_errors'] > 0:
            print(f"\n⚠️ 错误统计:")
            for error_type, count in error_stats['error_breakdown'].items():
                print(f"      {error_type}: {count} 次")
        
        # 4. 状态管理
        print(f"\n📋 状态管理:")
        active_requests = await pipeline.state_manager.get_active_requests()
        print(f"      活跃请求数: {len(active_requests)}")
        
        # 清理旧状态
        cleaned = await pipeline.state_manager.cleanup_old_states(max_age_hours=0.1)
        print(f"      清理状态数: {cleaned}")
        
        print("\n✅ 性能监控演示完成")
        
    finally:
        await pipeline.stop()


async def main():
    """主演示函数"""
    print("🎯 语音-文本-工具流水线完整演示")
    print("本演示将展示流水线的所有核心功能")
    
    try:
        # 1. 基本流水线演示
        await demo_basic_pipeline()
        
        # 2. 错误处理演示
        await demo_error_handling()
        
        # 3. 工具管理演示
        await demo_tool_management()
        
        # 4. 性能监控演示
        await demo_performance_monitoring()
        
        print("\n" + "=" * 60)
        print("🎉 所有演示完成!")
        print("=" * 60)
        print("\n📚 更多信息请查看:")
        print("   • README.md - 完整文档")
        print("   • examples.py - 更多示例")
        print("   • test_pipeline.py - 测试用例")
        print("\n🚀 开始使用语音流水线吧!")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())