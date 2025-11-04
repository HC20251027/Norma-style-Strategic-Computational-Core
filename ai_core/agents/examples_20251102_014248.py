"""
语音流水线使用示例

演示如何使用语音-文本-工具流水线
"""

import asyncio
import logging
from datetime import datetime

from .pipeline_manager import VoicePipelineManager
from .models import PipelineRequest, PipelineConfig
from .config import get_default_config, get_development_config


async def basic_usage_example():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    
    # 1. 初始化流水线管理器
    config = get_default_config()
    pipeline = VoicePipelineManager(config)
    
    # 2. 启动流水线
    await pipeline.start()
    
    try:
        # 3. 创建请求
        request = PipelineRequest(
            audio_data=b"mock_audio_data",  # 模拟音频数据
            language="zh-CN",
            metadata={"user_id": "user123"}
        )
        
        # 4. 处理请求
        response = await pipeline.process_request(request)
        
        # 5. 输出结果
        print(f"请求ID: {response.request_id}")
        print(f"成功: {response.success}")
        print(f"状态: {response.status.value}")
        print(f"处理时间: {response.processing_time:.2f}秒")
        print(f"文本响应: {response.text_response}")
        print(f"音频响应长度: {len(response.audio_response) if response.audio_response else 0}字节")
        
    finally:
        # 6. 停止流水线
        await pipeline.stop()


async def advanced_usage_example():
    """高级使用示例"""
    print("\n=== 高级使用示例 ===")
    
    # 1. 创建自定义配置
    config = PipelineConfig(
        speech_recognition={
            "engine": "mock",
            "language": "zh-CN",
            "confidence_threshold": 0.8
        },
        text_to_speech={
            "engine": "mock",
            "voice": "zh",
            "rate": 180,
            "volume": 0.9
        },
        performance={
            "max_concurrent_requests": 5,
            "cache_results": True
        }
    )
    
    pipeline = VoicePipelineManager(config)
    await pipeline.start()
    
    try:
        # 2. 批量处理请求
        requests = []
        for i in range(3):
            request = PipelineRequest(
                audio_data=f"audio_data_{i}".encode(),
                language="zh-CN",
                metadata={"batch_id": "batch_001", "request_index": i}
            )
            requests.append(request)
        
        # 3. 并行处理
        tasks = [pipeline.process_request(req) for req in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 4. 处理结果
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                print(f"请求 {i} 处理失败: {response}")
            else:
                print(f"请求 {i} 处理成功: {response.text_response}")
        
        # 5. 获取性能指标
        metrics = await pipeline.get_metrics()
        print(f"\n性能指标:")
        print(f"总请求数: {metrics['pipeline_metrics']['total_requests']}")
        print(f"成功请求数: {metrics['pipeline_metrics']['successful_requests']}")
        print(f"失败请求数: {metrics['pipeline_metrics']['failed_requests']}")
        print(f"平均处理时间: {metrics['pipeline_metrics']['average_processing_time']:.2f}秒")
        
    finally:
        await pipeline.stop()


async def tool_management_example():
    """工具管理示例"""
    print("\n=== 工具管理示例 ===")
    
    config = get_development_config()
    pipeline = VoicePipelineManager(config)
    await pipeline.start()
    
    try:
        # 1. 获取可用工具
        available_tools = await pipeline.get_available_tools()
        print(f"可用工具数量: {len(available_tools)}")
        for tool_name, tool_def in available_tools.items():
            print(f"- {tool_name}: {tool_def['description']}")
        
        # 2. 添加自定义工具
        custom_tool = {
            "description": "获取笑话",
            "parameters": {
                "category": {"type": "str", "required": False}
            },
            "keywords": ["笑话", "幽默", "搞笑"]
        }
        
        await pipeline.add_custom_tool("joke", custom_tool)
        print("\n已添加自定义工具: joke")
        
        # 3. 测试自定义工具
        request = PipelineRequest(
            audio_data=b"tell_me_a_joke",
            language="zh-CN"
        )
        
        response = await pipeline.process_request(request)
        print(f"自定义工具测试结果: {response.text_response}")
        
        # 4. 移除工具
        removed = await pipeline.remove_tool("joke")
        print(f"移除工具结果: {removed}")
        
    finally:
        await pipeline.stop()


async def error_handling_example():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    config = get_development_config()
    # 设置较短的超时时间以触发错误
    config.tool_mapping["timeout_seconds"] = 1
    config.error_handling["max_retries"] = 2
    
    pipeline = VoicePipelineManager(config)
    await pipeline.start()
    
    try:
        # 1. 创建可能失败的请求
        request = PipelineRequest(
            audio_data=b"invalid_audio_data",
            language="invalid_language"  # 无效语言
        )
        
        response = await pipeline.process_request(request)
        
        print(f"错误处理结果:")
        print(f"成功: {response.success}")
        print(f"状态: {response.status.value}")
        print(f"错误信息: {response.error_message}")
        print(f"文本响应: {response.text_response}")
        
        # 2. 获取错误统计
        metrics = await pipeline.get_metrics()
        error_stats = metrics.get('error_statistics', {})
        print(f"\n错误统计:")
        print(f"总错误数: {error_stats.get('total_errors', 0)}")
        print(f"错误分布: {error_stats.get('error_breakdown', {})}")
        
    finally:
        await pipeline.stop()


async def monitoring_example():
    """监控示例"""
    print("\n=== 监控示例 ===")
    
    config = get_default_config()
    pipeline = VoicePipelineManager(config)
    await pipeline.start()
    
    try:
        # 1. 处理多个请求以生成监控数据
        for i in range(5):
            request = PipelineRequest(
                audio_data=f"monitoring_test_{i}".encode(),
                language="zh-CN"
            )
            
            response = await pipeline.process_request(request)
            print(f"请求 {i} 处理完成: {response.success}")
        
        # 2. 获取详细状态
        state = await pipeline.get_request_status(response.request_id)
        if state:
            print(f"\n最后请求的状态:")
            print(f"请求ID: {state['request_id']}")
            print(f"状态: {state['status']}")
            print(f"当前阶段: {state['current_stage']}")
            print(f"进度: {state['progress']:.1%}")
            print(f"处理时间: {state['processing_times']}")
        
        # 3. 获取所有活跃请求
        active_requests = await pipeline.state_manager.get_active_requests()
        print(f"\n活跃请求数量: {len(active_requests)}")
        
        # 4. 获取完整性能指标
        metrics = await pipeline.get_metrics()
        print(f"\n完整性能指标:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
    finally:
        await pipeline.stop()


async def configuration_example():
    """配置示例"""
    print("\n=== 配置示例 ===")
    
    # 1. 获取不同环境的配置
    dev_config = get_development_config()
    prod_config = get_production_config()
    test_config = get_test_config()
    
    print("开发环境配置:")
    print(f"- 语音引擎: {dev_config.speech_recognition['engine']}")
    print(f"- TTS引擎: {dev_config.text_to_speech['engine']}")
    print(f"- 并发数: {dev_config.performance['max_concurrent_requests']}")
    
    print("\n生产环境配置:")
    print(f"- 语音引擎: {prod_config.speech_recognition['engine']}")
    print(f"- TTS引擎: {prod_config.text_to_speech['engine']}")
    print(f"- 并发数: {prod_config.performance['max_concurrent_requests']}")
    
    # 2. 动态更新配置
    pipeline = VoicePipelineManager(dev_config)
    await pipeline.start()
    
    try:
        # 更新配置
        new_config = get_production_config()
        new_config.speech_recognition["confidence_threshold"] = 0.9
        await pipeline.update_config(new_config)
        
        print(f"\n配置已更新，置信度阈值: {new_config.speech_recognition['confidence_threshold']}")
        
    finally:
        await pipeline.stop()


async def pipeline_test_example():
    """流水线测试示例"""
    print("\n=== 流水线测试示例 ===")
    
    config = get_test_config()
    pipeline = VoicePipelineManager(config)
    await pipeline.start()
    
    try:
        # 1. 运行完整测试
        test_results = await pipeline.test_pipeline()
        
        print("流水线测试结果:")
        for component, result in test_results.items():
            status = "✓" if result else "✗"
            print(f"{status} {component}: {'通过' if result else '失败'}")
        
        overall_status = "✓ 通过" if test_results['overall'] else "✗ 失败"
        print(f"\n整体测试结果: {overall_status}")
        
    finally:
        await pipeline.stop()


async def main():
    """主函数"""
    print("语音-文本-工具流水线示例")
    print("=" * 50)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 运行所有示例
        await basic_usage_example()
        await advanced_usage_example()
        await tool_management_example()
        await error_handling_example()
        await monitoring_example()
        await configuration_example()
        await pipeline_test_example()
        
        print("\n" + "=" * 50)
        print("所有示例运行完成！")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())