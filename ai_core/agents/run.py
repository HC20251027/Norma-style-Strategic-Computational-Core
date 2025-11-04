#!/usr/bin/env python3
"""
语音网关服务启动脚本

快速启动语音网关服务的命令行工具
"""

import asyncio
import argparse
import logging
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice_gateway import VoiceGateway, VoiceGatewayConfig
from voice_gateway.demo import VoiceGatewayDemo

def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

async def run_server(host: str = "0.0.0.0", port: int = 8765, debug: bool = False):
    """运行语音网关服务器"""
    setup_logging("DEBUG" if debug else "INFO")
    logger = logging.getLogger(__name__)
    
    try:
        # 创建配置
        config = VoiceGatewayConfig()
        config.WEBSOCKET_CONFIG["host"] = host
        config.WEBSOCKET_CONFIG["port"] = port
        
        # 创建语音网关
        gateway = VoiceGateway(config)
        
        logger.info("正在初始化语音网关服务...")
        
        # 初始化
        if not await gateway.initialize():
            logger.error("语音网关初始化失败")
            return False
        
        logger.info("正在启动语音网关服务...")
        
        # 启动
        if not await gateway.start():
            logger.error("语音网关启动失败")
            return False
        
        logger.info("=" * 50)
        logger.info("语音网关服务启动成功！")
        logger.info(f"WebSocket地址: ws://{host}:{port}")
        logger.info(f"按 Ctrl+C 停止服务")
        logger.info("=" * 50)
        
        # 保持服务运行
        try:
            while gateway.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到停止信号")
        
        logger.info("正在停止服务...")
        await gateway.stop()
        logger.info("服务已停止")
        
        return True
        
    except Exception as e:
        logger.error(f"服务运行失败: {e}")
        return False

async def run_demo():
    """运行演示"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        demo = VoiceGatewayDemo()
        await demo.run_demo()
    except Exception as e:
        logger.error(f"演示运行失败: {e}")

def run_test():
    """运行测试"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        from test_voice_gateway import run_tests
        asyncio.run(run_tests())
    except Exception as e:
        logger.error(f"测试运行失败: {e}")

def list_devices():
    """列出音频设备"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        from voice_gateway import AudioRecorder, RecordingConfig
        
        config = RecordingConfig()
        recorder = AudioRecorder(config)
        
        if recorder.initialize():
            devices = recorder.list_audio_devices()
            
            logger.info("可用音频输入设备:")
            for i, device in enumerate(devices):
                logger.info(f"  {i}: {device['name']} (通道数: {device['channels']}, 采样率: {device['sample_rate']})")
            
            recorder.terminate()
        else:
            logger.error("无法初始化音频录制器")
            
    except Exception as e:
        logger.error(f"列出设备失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="语音网关服务")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 服务器命令
    server_parser = subparsers.add_parser("server", help="启动语音网关服务器")
    server_parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    server_parser.add_argument("--port", type=int, default=8765, help="服务器端口")
    server_parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    # 演示命令
    demo_parser = subparsers.add_parser("demo", help="运行交互式演示")
    
    # 测试命令
    test_parser = subparsers.add_parser("test", help="运行组件测试")
    
    # 设备命令
    devices_parser = subparsers.add_parser("devices", help="列出音频设备")
    
    args = parser.parse_args()
    
    if args.command == "server":
        asyncio.run(run_server(args.host, args.port, args.debug))
    elif args.command == "demo":
        asyncio.run(run_demo())
    elif args.command == "test":
        run_test()
    elif args.command == "devices":
        list_devices()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()