#!/usr/bin/env python3
"""
语音流水线启动脚本

提供命令行接口来启动和使用语音流水线
"""

import argparse
import asyncio
import sys
import json
import logging
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from pipeline_manager import VoicePipelineManager
from models import PipelineRequest, PipelineConfig
from config import get_default_config, get_development_config, get_production_config
from examples import main as run_examples


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_config(config_type: str = "default") -> PipelineConfig:
    """加载配置"""
    config_map = {
        "default": get_default_config,
        "development": get_development_config,
        "production": get_production_config
    }
    
    if config_type not in config_map:
        print(f"Unknown config type: {config_type}")
        print(f"Available types: {', '.join(config_map.keys())}")
        sys.exit(1)
    
    return config_map[config_type]()


async def start_pipeline(config_type: str = "default"):
    """启动流水线服务"""
    setup_logging()
    
    print(f"启动语音流水线 (配置: {config_type})...")
    
    config = load_config(config_type)
    pipeline = VoicePipelineManager(config)
    
    await pipeline.start()
    
    try:
        print("流水线已启动，按 Ctrl+C 停止...")
        # 保持运行
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止流水线...")
    finally:
        await pipeline.stop()
        print("流水线已停止")


async def test_pipeline(config_type: str = "default"):
    """测试流水线"""
    setup_logging("DEBUG")
    
    print(f"测试语音流水线 (配置: {config_type})...")
    
    config = load_config(config_type)
    pipeline = VoicePipelineManager(config)
    
    await pipeline.start()
    
    try:
        # 运行测试
        test_results = await pipeline.test_pipeline()
        
        print("\n测试结果:")
        for component, result in test_results.items():
            status = "✓ 通过" if result else "✗ 失败"
            print(f"  {component}: {status}")
        
        overall_status = "✓ 通过" if test_results['overall'] else "✗ 失败"
        print(f"\n整体测试: {overall_status}")
        
        # 获取指标
        metrics = await pipeline.get_metrics()
        print(f"\n性能指标:")
        print(f"  总请求数: {metrics['pipeline_metrics']['total_requests']}")
        print(f"  成功请求数: {metrics['pipeline_metrics']['successful_requests']}")
        print(f"  失败请求数: {metrics['pipeline_metrics']['failed_requests']}")
        
    finally:
        await pipeline.stop()


async def process_audio_file(audio_file: str, config_type: str = "default"):
    """处理音频文件"""
    setup_logging()
    
    if not Path(audio_file).exists():
        print(f"音频文件不存在: {audio_file}")
        sys.exit(1)
    
    print(f"处理音频文件: {audio_file}")
    
    # 读取音频文件
    with open(audio_file, 'rb') as f:
        audio_data = f.read()
    
    config = load_config(config_type)
    pipeline = VoicePipelineManager(config)
    
    await pipeline.start()
    
    try:
        # 创建请求
        request = PipelineRequest(
            audio_data=audio_data,
            language="zh-CN",
            metadata={"source_file": audio_file}
        )
        
        # 处理请求
        print("正在处理...")
        response = await pipeline.process_request(request)
        
        # 输出结果
        print(f"\n处理结果:")
        print(f"  请求ID: {response.request_id}")
        print(f"  成功: {response.success}")
        print(f"  状态: {response.status.value}")
        print(f"  处理时间: {response.processing_time:.2f}秒")
        print(f"  文本响应: {response.text_response}")
        print(f"  音频响应长度: {len(response.audio_response) if response.audio_response else 0}字节")
        
        if response.error_message:
            print(f"  错误信息: {response.error_message}")
        
        # 保存输出音频
        if response.audio_response:
            output_file = f"output_{response.request_id}.wav"
            with open(output_file, 'wb') as f:
                f.write(response.audio_response)
            print(f"  输出音频已保存: {output_file}")
        
    finally:
        await pipeline.stop()


async def run_examples_demo():
    """运行示例演示"""
    setup_logging()
    print("运行语音流水线示例演示...")
    await run_examples()


async def show_status(config_type: str = "default"):
    """显示流水线状态"""
    setup_logging()
    
    config = load_config(config_type)
    pipeline = VoicePipelineManager(config)
    
    await pipeline.start()
    
    try:
        # 获取指标
        metrics = await pipeline.get_metrics()
        
        print("流水线状态:")
        print(f"  运行状态: {'运行中' if pipeline.is_running else '已停止'}")
        print(f"  活跃请求: {metrics['active_requests']}")
        print(f"  总请求数: {metrics['pipeline_metrics']['total_requests']}")
        print(f"  成功请求: {metrics['pipeline_metrics']['successful_requests']}")
        print(f"  失败请求: {metrics['pipeline_metrics']['failed_requests']}")
        print(f"  平均处理时间: {metrics['pipeline_metrics']['average_processing_time']:.2f}秒")
        
        # 显示错误统计
        error_stats = metrics['error_statistics']
        if error_stats['total_errors'] > 0:
            print(f"\n错误统计:")
            for error_type, count in error_stats['error_breakdown'].items():
                print(f"  {error_type}: {count}")
        
        # 显示可用工具
        tools = await pipeline.get_available_tools()
        print(f"\n可用工具 ({len(tools)} 个):")
        for tool_name, tool_def in tools.items():
            print(f"  {tool_name}: {tool_def['description']}")
        
    finally:
        await pipeline.stop()


async def interactive_mode(config_type: str = "default"):
    """交互模式"""
    setup_logging()
    
    config = load_config(config_type)
    pipeline = VoicePipelineManager(config)
    
    await pipeline.start()
    
    try:
        print("语音流水线交互模式")
        print("输入 'help' 查看帮助，输入 'quit' 退出")
        
        while True:
            try:
                # 获取用户输入
                text = input("\n请输入文本 (或输入 'quit' 退出): ").strip()
                
                if text.lower() == 'quit':
                    break
                elif text.lower() == 'help':
                    print("\n可用命令:")
                    print("  help - 显示帮助")
                    print("  quit - 退出程序")
                    print("  status - 显示状态")
                    print("  tools - 显示可用工具")
                    print("  其他文本 - 处理语音请求")
                    continue
                elif text.lower() == 'status':
                    await show_status(config_type)
                    continue
                elif text.lower() == 'tools':
                    tools = await pipeline.get_available_tools()
                    print(f"\n可用工具 ({len(tools)} 个):")
                    for tool_name, tool_def in tools.items():
                        print(f"  {tool_name}: {tool_def['description']}")
                    continue
                
                if not text:
                    print("请输入有效文本")
                    continue
                
                # 创建请求（使用模拟音频数据）
                request = PipelineRequest(
                    audio_data=b"interactive_mode_audio",
                    language="zh-CN",
                    metadata={"mode": "interactive"}
                )
                
                # 模拟语音转文本结果
                request.metadata["simulated_text"] = text
                
                # 处理请求
                print("正在处理...")
                response = await pipeline.process_request(request)
                
                # 显示结果
                print(f"\n处理结果:")
                print(f"  文本: {response.text_response}")
                print(f"  状态: {response.status.value}")
                print(f"  处理时间: {response.processing_time:.2f}秒")
                
                if response.error_message:
                    print(f"  错误: {response.error_message}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理错误: {e}")
        
        print("\n退出交互模式")
        
    finally:
        await pipeline.stop()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="语音流水线命令行工具")
    parser.add_argument(
        "command",
        choices=["start", "test", "process", "examples", "status", "interactive"],
        help="要执行的命令"
    )
    parser.add_argument(
        "--config",
        choices=["default", "development", "production"],
        default="default",
        help="配置类型"
    )
    parser.add_argument(
        "--audio-file",
        help="音频文件路径 (用于 process 命令)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 执行命令
    if args.command == "start":
        asyncio.run(start_pipeline(args.config))
    elif args.command == "test":
        asyncio.run(test_pipeline(args.config))
    elif args.command == "process":
        if not args.audio_file:
            print("错误: process 命令需要 --audio-file 参数")
            sys.exit(1)
        asyncio.run(process_audio_file(args.audio_file, args.config))
    elif args.command == "examples":
        asyncio.run(run_examples_demo())
    elif args.command == "status":
        asyncio.run(show_status(args.config))
    elif args.command == "interactive":
        asyncio.run(interactive_mode(args.config))


if __name__ == "__main__":
    main()