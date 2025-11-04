#!/usr/bin/env python3
"""
AG-UI使用示例
展示如何使用诺玛AI系统的AG-UI端点

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
import aiohttp
from datetime import datetime

async def example_health_check():
    """示例：健康检查"""
    print("=== 健康检查示例 ===")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8002/status') as response:
                result = await response.json()
                print(f"服务状态: {result['status']}")
                print(f"AG-UI启用: {result['agui_enabled']}")
                print(f"版本: {result['version']}")
    except Exception as e:
        print(f"健康检查失败: {e}")

async def example_agui_chat():
    """示例：AG-UI聊天交互"""
    print("\n=== AG-UI聊天示例 ===")
    
    # 构建AG-UI请求数据
    request_data = {
        "thread_id": "example_thread_001",
        "run_id": f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "messages": [
            {
                "role": "user",
                "content": "你好，诺玛，请介绍一下卡塞尔学院",
                "id": "msg_001"
            }
        ],
        "tools": [
            {
                "name": "search_knowledge",
                "description": "搜索知识库",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            }
        ],
        "context": [
            {
                "description": "当前时间",
                "value": datetime.now().isoformat()
            }
        ]
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8002/agui',
                json=request_data,
                headers={'Content-Type': 'application/json'}
            ) as response:
                print("收到AG-UI事件流:")
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        try:
                            event_data = json.loads(line[6:])  # 移除 'data: ' 前缀
                            event_type = event_data.get('type')
                            print(f"  📡 事件: {event_type}")
                            
                            # 显示特定事件的详细内容
                            if event_type == 'text_message_content':
                                content = event_data.get('content', '')
                                print(f"     内容: {content[:100]}{'...' if len(content) > 100 else ''}")
                            elif event_type == 'run_finished':
                                result = event_data.get('result', {})
                                print(f"     结果: {result}")
                                
                        except json.JSONDecodeError:
                            print(f"     原始数据: {line}")
                            
    except Exception as e:
        print(f"AG-UI请求失败: {e}")

async def example_conversation():
    """示例：多轮对话"""
    print("\n=== 多轮对话示例 ===")
    
    conversation_history = [
        {
            "role": "user",
            "content": "我想了解龙族血统分析",
            "id": "msg_001"
        },
        {
            "role": "assistant", 
            "content": "龙族血统分析是卡塞尔学院的重要功能之一。我们可以分析学生的血统纯度和能力。",
            "id": "msg_002"
        },
        {
            "role": "user",
            "content": "那么我的血统分析结果如何？",
            "id": "msg_003"
        }
    ]
    
    request_data = {
        "thread_id": "conversation_001",
        "run_id": f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "messages": conversation_history,
        "tools": [
            {
                "name": "dragon_blood_analysis",
                "description": "龙族血统分析",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "student_name": {"type": "string"}
                    }
                }
            }
        ]
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:8002/agui',
                json=request_data
            ) as response:
                print("多轮对话事件流:")
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        try:
                            event_data = json.loads(line[6:])
                            event_type = event_data.get('type')
                            print(f"  📡 {event_type}")
                            
                            if event_type == 'tool_call_start':
                                tool_name = event_data.get('tool_name')
                                print(f"     🔧 开始调用工具: {tool_name}")
                                
                        except json.JSONDecodeError:
                            continue
                            
    except Exception as e:
        print(f"多轮对话失败: {e}")

async def main():
    """主函数"""
    print("诺玛AI系统 AG-UI 使用示例")
    print("=" * 50)
    print("注意：请确保后端服务正在运行 (python /workspace/backend/main_agui.py)")
    print()
    
    # 检查服务是否运行
    try:
        await example_health_check()
        print("\n✅ 服务连接成功，开始演示...")
        
        await asyncio.sleep(1)
        await example_agui_chat()
        
        await asyncio.sleep(2)
        await example_conversation()
        
        print("\n🎉 示例演示完成！")
        
    except Exception as e:
        print(f"❌ 服务连接失败: {e}")
        print("请确保后端服务正在运行在 http://localhost:8002")

if __name__ == "__main__":
    asyncio.run(main())