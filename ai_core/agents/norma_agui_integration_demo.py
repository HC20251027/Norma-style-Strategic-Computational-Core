#!/usr/bin/env python3
"""
诺玛AI系统 - AG-UI事件系统集成示例
展示如何在诺玛系统中集成和使用AG-UI事件系统

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import sys
import json
from datetime import datetime
from typing import Dict, Any

# 添加代码路径
sys.path.append('/workspace/code')

from agui_event_system import (
    AGUIEventSystem,
    EventEncoder,
    EventType,
    EventPriority,
    EventStatus,
    AGUIEvent,
    event_system,
    publish_system_event,
    publish_user_message,
    publish_ai_response,
    publish_blood_analysis,
    publish_security_event
)

class NormaEventIntegration:
    """诺玛AI系统事件集成类"""
    
    def __init__(self):
        self.event_system = event_system
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.is_running = False
    
    async def initialize(self):
        """初始化事件系统"""
        print("正在初始化诺玛AI系统事件集成...")
        
        # 启动事件系统
        await self.event_system.start()
        
        # 设置事件处理器
        self._setup_event_handlers()
        
        self.is_running = True
        print("诺玛AI系统事件集成已启动")
    
    def _setup_event_handlers(self):
        """设置事件处理器"""
        # 用户消息处理器
        self.event_system.add_event_handler(
            EventType.USER_MESSAGE,
            self._handle_user_message
        )
        
        # 系统状态处理器
        self.event_system.add_event_handler(
            EventType.SYSTEM_STATUS,
            self._handle_system_status
        )
        
        # 安全事件处理器
        self.event_system.add_event_handler(
            EventType.SECURITY_ALERT,
            self._handle_security_alert
        )
        
        # 血统分析处理器
        self.event_system.add_event_handler(
            EventType.BLOOD_RESULT,
            self._handle_blood_result
        )
    
    async def _handle_user_message(self, event: AGUIEvent):
        """处理用户消息事件"""
        user_id = event.data.get("user_id", "unknown")
        message = event.data.get("message", "")
        
        print(f"📨 收到用户消息 [{user_id}]: {message}")
        
        # 创建用户会话
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                "session_start": datetime.now(),
                "message_count": 0,
                "last_activity": datetime.now()
            }
        
        session = self.user_sessions[user_id]
        session["message_count"] += 1
        session["last_activity"] = datetime.now()
        
        # 模拟AI处理
        await asyncio.sleep(0.1)  # 模拟处理时间
        
        # 生成AI响应
        ai_response = await self._generate_ai_response(message, user_id)
        
        # 发布AI响应事件
        await publish_ai_response(ai_response, message)
    
    async def _handle_system_status(self, event: AGUIEvent):
        """处理系统状态事件"""
        status_data = event.data
        print(f"📊 系统状态更新: {status_data}")
        
        # 可以在这里添加系统状态监控逻辑
        if "cpu_usage" in status_data:
            cpu_usage = float(status_data["cpu_usage"].replace("%", ""))
            if cpu_usage > 80:
                await publish_security_event(
                    EventType.SECURITY_ALERT,
                    {
                        "alert_type": "high_cpu_usage",
                        "value": cpu_usage,
                        "threshold": 80,
                        "timestamp": datetime.now().isoformat()
                    }
                )
    
    async def _handle_security_alert(self, event: AGUIEvent):
        """处理安全警报事件"""
        alert_data = event.data
        print(f"🚨 安全警报: {alert_data}")
        
        # 可以在这里添加安全响应逻辑
        alert_type = alert_data.get("alert_type", "unknown")
        if alert_type == "high_cpu_usage":
            print("⚠️  检测到高CPU使用率，建议检查系统负载")
    
    async def _handle_blood_result(self, event: AGUIEvent):
        """处理血统分析结果事件"""
        result_data = event.data
        student_name = result_data.get("student_name", "未知")
        analysis_result = result_data.get("analysis_result", {})
        
        print(f"🩸 血统分析完成 - 学生: {student_name}")
        print(f"   血统类型: {analysis_result.get('bloodline_type', '未知')}")
        print(f"   纯度: {analysis_result.get('purity_level', '未知')}")
        print(f"   能力: {analysis_result.get('abilities', '未知')}")
    
    async def _generate_ai_response(self, message: str, user_id: str) -> str:
        """生成AI响应"""
        # 简单的响应逻辑
        if "血统" in message or "blood" in message.lower():
            return "正在为您查询血统信息，请稍候..."
        elif "安全" in message or "security" in message.lower():
            return "正在进行安全状态检查，请稍候..."
        elif "状态" in message or "status" in message.lower():
            return "系统当前运行正常，各项指标良好。"
        else:
            return f"您好！我是诺玛·劳恩斯，已收到您的消息: {message[:20]}..."
    
    async def simulate_user_interaction(self):
        """模拟用户交互"""
        print("\n=== 模拟用户交互 ===")
        
        users = ["student001", "student002", "admin001"]
        
        for user_id in users:
            # 发送用户消息
            messages = [
                "你好，诺玛！",
                "我想查询我的血统信息",
                "系统状态怎么样？",
                "安全检查正常吗？"
            ]
            
            for message in messages:
                await publish_user_message(message, user_id)
                await asyncio.sleep(0.5)  # 模拟用户输入间隔
        
        # 显示会话统计
        print(f"\n会话统计:")
        for user_id, session in self.user_sessions.items():
            print(f"  用户 {user_id}: {session['message_count']} 条消息")
    
    async def simulate_system_events(self):
        """模拟系统事件"""
        print("\n=== 模拟系统事件 ===")
        
        # 系统状态事件
        await publish_system_event(EventType.SYSTEM_STATUS, {
            "cpu_usage": "15.3%",
            "memory_usage": "42.7%",
            "disk_usage": "67.2%",
            "network_status": "活跃",
            "active_connections": 15,
            "timestamp": datetime.now().isoformat()
        })
        
        await asyncio.sleep(0.5)
        
        # 血统分析事件
        students = [
            ("路明非", {"bloodline_type": "S级混血种", "purity_level": "95.2%", "abilities": "黄金瞳、言灵·君焰"}),
            ("楚子航", {"bloodline_type": "A级混血种", "purity_level": "87.3%", "abilities": "黄金瞳、言灵·君焰"}),
            ("凯撒", {"bloodline_type": "A级混血种", "purity_level": "89.1%", "abilities": "黄金瞳、言灵·镰鼬"})
        ]
        
        for student_name, result in students:
            await publish_blood_analysis(student_name, result)
            await asyncio.sleep(0.3)
        
        # 安全事件
        await publish_security_event(EventType.SECURITY_STATUS, {
            "firewall_status": "正常",
            "antivirus_status": "运行中",
            "intrusion_detection": "监控中",
            "threat_level": "低",
            "last_scan": datetime.now().isoformat()
        })
        
        await asyncio.sleep(0.5)
        
        # 模拟高CPU使用率警报
        await publish_system_event(EventType.SYSTEM_STATUS, {
            "cpu_usage": "85.7%",  # 高CPU使用率
            "memory_usage": "78.2%",
            "timestamp": datetime.now().isoformat()
        })
    
    async def show_event_statistics(self):
        """显示事件统计"""
        print("\n=== 事件统计信息 ===")
        
        # 获取流统计
        stats = self.event_system.get_stream_stats()
        print("事件流统计:")
        for stream_name, stream_stats in stats.items():
            print(f"  {stream_name}流: {stream_stats['event_count']} 个事件")
        
        # 获取最近事件
        recent_events = self.event_system.get_event_history(10)
        print(f"\n最近 {len(recent_events)} 个事件:")
        for i, event in enumerate(recent_events, 1):
            print(f"  {i}. {event.timestamp} - {event.type.value} - {event.source}")
        
        # 用户会话统计
        print(f"\n用户会话统计:")
        for user_id, session in self.user_sessions.items():
            duration = datetime.now() - session["session_start"]
            print(f"  用户 {user_id}: {session['message_count']} 条消息, "
                  f"会话时长: {duration.total_seconds():.1f}秒")
    
    async def cleanup(self):
        """清理资源"""
        print("\n正在清理资源...")
        
        # 停止事件系统
        await self.event_system.stop()
        
        self.is_running = False
        print("诺玛AI系统事件集成已停止")

async def main():
    """主函数"""
    print("诺玛AI系统 - AG-UI事件系统集成演示")
    print("=" * 60)
    
    # 创建集成实例
    integration = NormaEventIntegration()
    
    try:
        # 初始化
        await integration.initialize()
        
        # 模拟用户交互
        await integration.simulate_user_interaction()
        
        # 模拟系统事件
        await integration.simulate_system_events()
        
        # 等待事件处理完成
        await asyncio.sleep(1)
        
        # 显示统计信息
        await integration.show_event_statistics()
        
    except Exception as e:
        print(f"演示过程中发生错误: {e}")
    
    finally:
        # 清理资源
        await integration.cleanup()
    
    print("\n=== 演示完成 ===")
    print("AG-UI事件系统已成功集成到诺玛AI系统中！")

if __name__ == "__main__":
    asyncio.run(main())