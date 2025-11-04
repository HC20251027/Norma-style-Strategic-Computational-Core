#!/usr/bin/env python3
"""
非阻塞等待系统API服务器
提供REST API和WebSocket接口
"""

import asyncio
import uvicorn
import sys
import os

# 添加路径以便导入模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from backend.src.non_blocking import *
from backend.src.non_blocking.api import NonBlockingAPI


class NonBlockingServer:
    """非阻塞等待系统服务器"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.config = NonBlockingConfig()
        
        # 创建组件
        self.task_manager = TaskManager(self.config)
        self.progress_pusher = ProgressPusher(self.task_manager, self.config)
        self.result_manager = AsyncResultManager(self.config)
        self.status_manager = RealtimeStatusManager(self.task_manager, self.config)
        self.timeout_handler = TimeoutHandler(self.task_manager, self.config)
        self.ux_optimizer = UserExperienceOptimizer(self.task_manager, self.config)
        
        # 创建API
        self.api = NonBlockingAPI(
            task_manager=self.task_manager,
            progress_pusher=self.progress_pusher,
            result_manager=self.result_manager,
            status_manager=self.status_manager,
            timeout_handler=self.timeout_handler,
            ux_optimizer=self.ux_optimizer,
            config=self.config
        )
        
        self.app = self.api.get_app()
        self.running = False
    
    async def start(self):
        """启动服务器"""
        if self.running:
            print("服务器已在运行")
            return
        
        print(f"🚀 启动非阻塞等待系统服务器...")
        print(f"   主机: {self.host}")
        print(f"   端口: {self.port}")
        
        # 启动所有组件
        print("🔄 启动系统组件...")
        await self.task_manager.start()
        await self.progress_pusher.start()
        await self.result_manager.start()
        await self.status_manager.start()
        await self.timeout_handler.start()
        await self.ux_optimizer.start()
        
        self.running = True
        print("✅ 系统组件启动完成")
        
        # 显示API信息
        self._show_api_info()
        
        # 启动Uvicorn服务器
        print(f"\n🌐 启动Web服务器...")
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=True
        )
        server = uvicorn.Server(config)
        
        try:
            await server.serve()
        except KeyboardInterrupt:
            print("\n🛑 收到停止信号")
        finally:
            await self.stop()
    
    async def stop(self):
        """停止服务器"""
        if not self.running:
            return
        
        print("🔄 正在停止服务器...")
        
        # 停止所有组件
        await self.ux_optimizer.stop()
        await self.timeout_handler.stop()
        await self.status_manager.stop()
        await self.result_manager.stop()
        await self.progress_pusher.stop()
        await self.task_manager.stop()
        
        self.running = False
        print("✅ 服务器已停止")
    
    def _show_api_info(self):
        """显示API信息"""
        print("\n📡 API接口信息:")
        print("   REST API:")
        print("     GET  /health                    - 健康检查")
        print("     POST /tasks                     - 创建任务")
        print("     GET  /tasks/{task_id}           - 获取任务信息")
        print("     GET  /tasks/{task_id}/progress  - 获取任务进度")
        print("     GET  /results/{result_id}       - 获取结果")
        print("     GET  /results/{result_id}/wait  - 等待结果")
        print("     GET  /stats                     - 获取系统统计")
        print("     DELETE /tasks/{task_id}         - 取消任务")
        
        print("\n   WebSocket API:")
        print("     WS  /ws/{connection_id}         - WebSocket连接")
        
        print("\n📖 使用示例:")
        print("   # 创建任务")
        print('   curl -X POST "http://localhost:8000/tasks" \\')
        print('        -H "Content-Type: application/json" \\')
        print('        -d \'{"name": "测试任务", "function_path": "time.sleep", "args": [2]}\'')
        
        print("\n   # 获取任务信息")
        print('   curl "http://localhost:8000/tasks/{task_id}"')
        
        print("\n   # 等待结果")
        print('   curl "http://localhost:8000/results/{result_id}/wait?timeout=30"')


async def demo_api_usage():
    """演示API使用"""
    print("🎯 非阻塞等待系统API使用演示")
    print("=" * 50)
    
    # 创建服务器实例
    server = NonBlockingServer(host="127.0.0.1", port=8001)
    
    # 在后台启动服务器
    server_task = asyncio.create_task(server.start())
    
    # 等待服务器启动
    await asyncio.sleep(2)
    
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            print("\n📡 测试API接口:")
            
            # 1. 健康检查
            print("1. 健康检查...")
            async with session.get("http://127.0.0.1:8001/health") as resp:
                if resp.status == 200:
                    health_data = await resp.json()
                    print(f"   ✅ 健康状态: {health_data['status']}")
                else:
                    print(f"   ❌ 健康检查失败: {resp.status}")
            
            # 2. 创建任务
            print("2. 创建任务...")
            task_data = {
                "name": "API测试任务",
                "function_path": "asyncio.sleep",
                "args": [2],
                "correlation_id": "api_demo_user"
            }
            
            async with session.post(
                "http://127.0.0.1:8001/tasks",
                json=task_data
            ) as resp:
                if resp.status == 200:
                    task_response = await resp.json()
                    task_id = task_response["task_id"]
                    result_id = task_response["result_id"]
                    print(f"   ✅ 任务创建成功")
                    print(f"      任务ID: {task_id}")
                    print(f"      结果ID: {result_id}")
                    
                    # 3. 获取任务信息
                    print("3. 获取任务信息...")
                    await asyncio.sleep(0.5)
                    
                    async with session.get(f"http://127.0.0.1:8001/tasks/{task_id}") as resp:
                        if resp.status == 200:
                            task_info = await resp.json()
                            print(f"   ✅ 任务状态: {task_info['status']}")
                            print(f"      进度: {task_info['progress']}%")
                        else:
                            print(f"   ❌ 获取任务信息失败: {resp.status}")
                    
                    # 4. 等待结果
                    print("4. 等待结果...")
                    async with session.get(
                        f"http://127.0.0.1:8001/results/{result_id}/wait?timeout=10"
                    ) as resp:
                        if resp.status == 200:
                            result_data = await resp.json()
                            print(f"   ✅ 结果获取成功: {result_data['result']}")
                        else:
                            print(f"   ❌ 等待结果失败: {resp.status}")
                
                else:
                    print(f"   ❌ 创建任务失败: {resp.status}")
            
            # 5. 获取系统统计
            print("5. 获取系统统计...")
            async with session.get("http://127.0.0.1:8001/stats") as resp:
                if resp.status == 200:
                    stats_data = await resp.json()
                    print(f"   ✅ 系统统计获取成功")
                    print(f"      任务统计: {stats_data['task_manager']['total_tasks']} 个任务")
                    print(f"      结果统计: {stats_data['result_manager']['total_results']} 个结果")
                else:
                    print(f"   ❌ 获取系统统计失败: {resp.status}")
        
        print("\n🎉 API演示完成!")
        
    except ImportError:
        print("   ⚠️  aiohttp未安装，跳过API演示")
    except Exception as e:
        print(f"   ❌ API演示出错: {e}")
    
    finally:
        # 停止服务器
        server_task.cancel()
        try:
            await server.stop()
        except:
            pass


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="非阻塞等待系统API服务器")
    parser.add_argument("--host", default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    parser.add_argument("--demo", action="store_true", help="运行API演示")
    parser.add_argument("--reload", action="store_true", help="启用热重载")
    
    args = parser.parse_args()
    
    if args.demo:
        # 运行API演示
        asyncio.run(demo_api_usage())
    else:
        # 启动服务器
        server = NonBlockingServer(host=args.host, port=args.port)
        
        if args.reload:
            # 开发模式，使用热重载
            config = uvicorn.Config(
                app=server.app,
                host=args.host,
                port=args.port,
                reload=True,
                log_level="info"
            )
            server_instance = uvicorn.Server(config)
            asyncio.run(server_instance.serve())
        else:
            # 生产模式
            asyncio.run(server.start())


if __name__ == "__main__":
    main()