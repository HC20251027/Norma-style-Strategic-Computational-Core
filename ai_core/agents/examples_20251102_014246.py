"""
LLM多模态核心系统使用示例
演示如何使用LLM编排器、多模态处理、流式生成等功能
"""

import asyncio
import base64
import json
from typing import Dict, Any

from llm import (
    LLMOrchestrator,
    LLMManager,
    LLMInterface,
    ResponseCache,
    ModelRouter,
    LoadBalancer,
    TextProcessor,
    ImageProcessor,
    AudioProcessor,
    VideoProcessor,
    StreamManager
)
from llm.config import ModelType, Provider
from llm.core.router import RoutingStrategy
from llm.core.load_balancer import LoadBalanceAlgorithm

async def basic_text_generation_example():
    """基础文本生成示例"""
    print("=== 基础文本生成示例 ===")
    
    # 创建LLM管理器
    manager = LLMManager()
    await manager.start()
    
    try:
        # 生成文本
        result = await manager.generate_text(
            prompt="请介绍一下人工智能的发展历史",
            model="gpt-3.5-turbo"
        )
        
        if result['success']:
            print(f"生成的文本: {result['response'].content}")
            print(f"使用的模型: {result['response'].model_used}")
            print(f"响应时间: {result['metrics']['response_time']:.2f}秒")
        else:
            print(f"生成失败: {result['error']}")
    
    finally:
        await manager.shutdown()

async def multimodal_processing_example():
    """多模态处理示例"""
    print("\n=== 多模态处理示例 ===")
    
    # 创建文本处理器
    text_processor = TextProcessor()
    
    # 处理文本
    text_result = await text_processor.process_text(
        "这是一个测试文本，包含中文和English混合内容。"
    )
    
    print("文本处理结果:")
    print(f"- 处理后文本: {text_result['processed_text']}")
    print(f"- 检测语言: {text_result['detected_language']}")
    print(f"- Token数量: {text_result['token_count']}")
    print(f"- 质量评分: {text_result['quality_score']:.2f}")
    
    # 提取关键词
    keywords = await text_processor.extract_keywords(
        text_result['processed_text'],
        language=text_result['detected_language']
    )
    
    print(f"- 关键词: {[kw['word'] for kw in keywords[:5]]}")

async def image_processing_example():
    """图像处理示例"""
    print("\n=== 图像处理示例 ===")
    
    # 创建图像处理器
    image_processor = ImageProcessor()
    
    # 模拟图像数据（实际使用时应传入真实的图像文件）
    mock_image_data = b"mock_image_data"
    
    try:
        # 处理图像
        result = await image_processor.process_image(mock_image_data)
        
        print("图像处理结果:")
        print(f"- 原始大小: {result['original_size']} 字节")
        print(f"- 处理后大小: {result['processed_size']} 字节")
        print(f"- 压缩比: {result['compression_ratio']:.2f}")
        print(f"- 图像哈希: {result['hash'][:16]}...")
        
        metadata = result['metadata']
        print(f"- 图像格式: {metadata.format}")
        print(f"- 图像尺寸: {metadata.width}x{metadata.height}")
        print(f"- 颜色模式: {metadata.mode}")
        
    except Exception as e:
        print(f"图像处理失败: {e}")

async def streaming_example():
    """流式生成示例"""
    print("\n=== 流式生成示例 ===")
    
    # 创建流管理器
    stream_manager = StreamManager()
    
    try:
        # 创建流连接
        connection_id = await stream_manager.create_connection("test_request")
        print(f"创建流连接: {connection_id}")
        
        # 流式发送文本
        test_text = "这是一个流式文本生成的示例，文本会逐个字符发送。"
        await stream_manager.stream_text(connection_id, test_text)
        
        print("流式文本已发送")
        
        # 获取连接统计
        stats = stream_manager.get_stats()
        print(f"流统计: {stats['active_connections']} 个活跃连接")
        
    finally:
        await stream_manager.shutdown()

async def model_routing_example():
    """模型路由示例"""
    print("\n=== 模型路由示例 ===")
    
    # 创建模型路由器
    router = ModelRouter()
    
    # 测试不同类型的路由
    test_cases = [
        (ModelType.TEXT, "生成一段文本"),
        (ModelType.MULTIMODAL, "分析这张图片"),
    ]
    
    for model_type, prompt in test_cases:
        # 路由请求
        selected_model = await router.route_request(
            request_type=model_type,
            content={"prompt": prompt}
        )
        
        print(f"类型: {model_type.value}")
        print(f"选择的模型: {selected_model}")
        print(f"提示词: {prompt}")
        print("---")
    
    # 获取模型状态
    model_status = router.get_model_status()
    print("\n模型状态:")
    for model_name, status in model_status.items():
        print(f"- {model_name}: {status['available']} (错误率: {status['error_rate']:.2%})")

async def cache_example():
    """缓存示例"""
    print("\n=== 缓存示例 ===")
    
    # 创建缓存
    cache = ResponseCache()
    
    # 设置缓存
    test_data = {
        "content": "这是缓存的测试数据",
        "model": "gpt-3.5-turbo",
        "timestamp": "2024-01-01"
    }
    
    await cache.set("test_key", test_data, ttl=60)
    print("数据已缓存")
    
    # 获取缓存
    cached_data = await cache.get("test_key")
    if cached_data:
        print(f"缓存命中: {cached_data['content']}")
    else:
        print("缓存未命中")
    
    # 获取缓存统计
    stats = cache.get_stats()
    print(f"缓存统计: 命中率 {stats['hit_rate']:.2%}")

async def load_balancer_example():
    """负载均衡示例"""
    print("\n=== 负载均衡示例 ===")
    
    # 创建负载均衡器
    load_balancer = LoadBalancer()
    
    # 添加节点
    from llm.core.load_balancer import ServerNode
    
    nodes = [
        ServerNode("node1", "192.168.1.10", 8080, weight=2),
        ServerNode("node2", "192.168.1.11", 8080, weight=1),
        ServerNode("node3", "192.168.1.12", 8080, weight=3),
    ]
    
    for node in nodes:
        load_balancer.add_node(node)
    
    print("已添加3个节点")
    
    # 测试负载均衡算法
    algorithms = [
        LoadBalanceAlgorithm.ROUND_ROBIN,
        LoadBalanceAlgorithm.WEIGHTED_ROUND_ROBIN,
        LoadBalanceAlgorithm.LEAST_CONNECTIONS
    ]
    
    for algorithm in algorithms:
        load_balancer.set_algorithm(algorithm)
        
        # 选择节点
        selected_nodes = []
        for i in range(5):
            node = await load_balancer.select_node()
            if node:
                selected_nodes.append(node.id)
        
        print(f"{algorithm.value}: {selected_nodes}")
    
    # 获取负载均衡统计
    stats = load_balancer.get_load_balance_stats()
    print(f"\n负载均衡统计:")
    print(f"- 总节点数: {stats['total_nodes']}")
    print(f"- 健康节点数: {stats['healthy_nodes']}")
    print(f"- 总请求数: {stats['total_requests']}")

async def batch_processing_example():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")
    
    # 创建LLM接口
    llm_interface = LLMInterface()
    
    # 创建批量请求
    requests = [
        {
            "type": "text",
            "prompt": "解释什么是机器学习",
            "id": "batch_1"
        },
        {
            "type": "text", 
            "prompt": "介绍深度学习的基本概念",
            "id": "batch_2"
        },
        {
            "type": "text",
            "prompt": "说明神经网络的工作原理",
            "id": "batch_3"
        }
    ]
    
    # 批量处理
    results = await llm_interface.batch_process(requests, max_concurrent=2)
    
    print(f"批量处理完成，处理了 {len(results)} 个请求:")
    for i, result in enumerate(results):
        if result.error:
            print(f"- 请求 {i+1}: 失败 - {result.error}")
        else:
            print(f"- 请求 {i+1}: 成功 - 模型 {result.model_used}")

async def system_monitoring_example():
    """系统监控示例"""
    print("\n=== 系统监控示例 ===")
    
    # 创建LLM管理器
    manager = LLMManager()
    await manager.start()
    
    try:
        # 发送一些请求以生成指标
        for i in range(5):
            await manager.generate_text(f"测试请求 {i+1}")
            await asyncio.sleep(0.1)
        
        # 获取系统状态
        status = manager.get_service_status()
        
        print("系统状态:")
        print(f"- 服务状态: {status['status']}")
        print(f"- 运行时间: {status['uptime']:.2f} 秒")
        print(f"- 总请求数: {status['metrics']['total_requests']}")
        print(f"- 成功请求数: {status['metrics']['successful_requests']}")
        print(f"- 平均响应时间: {status['metrics']['avg_response_time']:.2f} 秒")
        
        # 获取详细指标
        detailed_metrics = manager.get_detailed_metrics()
        print(f"\n模型性能:")
        for model_name, perf in detailed_metrics['model_performance'].items():
            print(f"- {model_name}: {perf['total_requests']} 请求, "
                  f"成功率 {perf['success_rate']:.2%}")
    
    finally:
        await manager.shutdown()

async def configuration_management_example():
    """配置管理示例"""
    print("\n=== 配置管理示例 ===")
    
    # 创建LLM管理器
    manager = LLMManager()
    
    # 查看当前配置
    print("当前配置:")
    config_data = manager.interface_config.__dict__
    for key, value in config_data.items():
        print(f"- {key}: {value}")
    
    # 更新配置
    manager.update_configuration(
        default_model="gpt-4",
        enable_streaming=True,
        max_retries=5
    )
    
    print("\n配置已更新")
    
    # 设置路由策略
    manager.set_routing_strategy(RoutingStrategy.PRIORITY_BASED)
    
    # 设置负载均衡算法
    manager.set_load_balance_algorithm(LoadBalanceAlgorithm.WEIGHTED_ROUND_ROBIN)
    
    print("路由和负载均衡策略已设置")

async def main():
    """主函数 - 运行所有示例"""
    print("LLM多模态核心系统示例")
    print("=" * 50)
    
    examples = [
        basic_text_generation_example,
        multimodal_processing_example,
        image_processing_example,
        streaming_example,
        model_routing_example,
        cache_example,
        load_balancer_example,
        batch_processing_example,
        system_monitoring_example,
        configuration_management_example
    ]
    
    for example in examples:
        try:
            await example()
            await asyncio.sleep(1)  # 间隔1秒
        except Exception as e:
            print(f"示例 {example.__name__} 执行失败: {e}")
    
    print("\n所有示例执行完成!")

if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())