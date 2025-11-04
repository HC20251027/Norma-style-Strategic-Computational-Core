#!/usr/bin/env python3
"""
聊天界面组件
提供用户友好的对话交互界面

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import dataclass, asdict

from ..utils.logger import NormaLogger

@dataclass
class ChatMessage:
    """聊天消息"""
    id: str
    content: str
    sender: str  # "user" or "assistant"
    timestamp: datetime
    message_type: str = "text"  # text, image, audio, video, file
    metadata: Dict[str, Any] = None
    is_streaming: bool = False
    streaming_content: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "sender": self.sender,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type,
            "metadata": self.metadata or {},
            "is_streaming": self.is_streaming,
            "streaming_content": self.streaming_content
        }

class ChatInterface:
    """聊天界面类"""
    
    def __init__(self, agent):
        """初始化聊天界面
        
        Args:
            agent: 诺玛Agent实例
        """
        self.agent = agent
        self.logger = NormaLogger("chat_interface")
        
        # 聊天会话管理
        self.chat_sessions: Dict[str, List[ChatMessage]] = {}
        self.current_session_id: Optional[str] = None
        
        # 界面状态
        self.is_active = False
        self.is_streaming = False
        
        # 配置
        self.config = {
            "max_messages_per_session": 100,
            "auto_save": True,
            "streaming_enabled": True,
            "show_timestamps": True,
            "show_sender_avatar": True
        }
        
        # 消息处理器
        self.message_handlers: Dict[str, callable] = {}
    
    async def create_chat_session(self, session_id: Optional[str] = None) -> str:
        """创建聊天会话"""
        
        if not session_id:
            session_id = f"chat_{int(datetime.now().timestamp())}"
        
        self.chat_sessions[session_id] = []
        self.current_session_id = session_id
        
        # 创建欢迎消息
        welcome_message = ChatMessage(
            id=f"welcome_{session_id}",
            content="你好！我是诺玛·劳恩斯，卡塞尔学院的主控计算机AI系统。有什么可以帮助您的吗？",
            sender="assistant",
            timestamp=datetime.now(),
            metadata={"type": "system_welcome"}
        )
        
        self.chat_sessions[session_id].append(welcome_message)
        
        self.logger.info(f"创建聊天会话: {session_id}")
        return session_id
    
    async def send_message(
        self,
        content: str,
        message_type: str = "text",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """发送消息并获取回复"""
        
        # 确保有活跃会话
        if not session_id:
            if not self.current_session_id:
                session_id = await self.create_chat_session()
            else:
                session_id = self.current_session_id
        elif session_id not in self.chat_sessions:
            await self.create_chat_session(session_id)
        
        # 创建用户消息
        user_message = ChatMessage(
            id=f"user_{int(datetime.now().timestamp())}",
            content=content,
            sender="user",
            timestamp=datetime.now(),
            message_type=message_type,
            metadata=metadata or {}
        )
        
        # 添加用户消息到会话
        self.chat_sessions[session_id].append(user_message)
        
        # 限制消息数量
        await self._limit_messages(session_id)
        
        # 创建助手消息（用于流式更新）
        assistant_message = ChatMessage(
            id=f"assistant_{int(datetime.now().timestamp())}",
            content="",
            sender="assistant",
            timestamp=datetime.now(),
            message_type="text",
            is_streaming=True,
            streaming_content=""
        )
        
        self.chat_sessions[session_id].append(assistant_message)
        
        self.is_streaming = True
        
        try:
            # 通过Agent处理消息
            async for response_chunk in self.agent.process_message(
                message=content,
                session_id=session_id,
                message_type=message_type,
                metadata=metadata
            ):
                # 更新流式消息
                assistant_message.streaming_content += response_chunk
                assistant_message.content = assistant_message.streaming_content
                
                # 发送增量更新
                yield json.dumps({
                    "type": "message_chunk",
                    "message_id": assistant_message.id,
                    "chunk": response_chunk,
                    "timestamp": datetime.now().isoformat()
                }, ensure_ascii=False)
            
            # 完成流式更新
            assistant_message.is_streaming = False
            
            # 发送完成信号
            yield json.dumps({
                "type": "message_complete",
                "message_id": assistant_message.id,
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"消息处理失败: {e}")
            assistant_message.content = f"抱歉，处理您的消息时发生了错误: {str(e)}"
            assistant_message.is_streaming = False
            
            yield json.dumps({
                "type": "message_error",
                "message_id": assistant_message.id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False)
        
        finally:
            self.is_streaming = False
    
    async def _limit_messages(self, session_id: str) -> None:
        """限制会话中的消息数量"""
        
        max_messages = self.config["max_messages_per_session"]
        messages = self.chat_sessions[session_id]
        
        if len(messages) > max_messages:
            # 保留系统消息和最近的对话
            system_messages = [msg for msg in messages if msg.metadata.get("type") == "system_welcome"]
            recent_messages = messages[-max_messages+len(system_messages):]
            
            self.chat_sessions[session_id] = system_messages + recent_messages
    
    def get_chat_session(self, session_id: str) -> Optional[List[ChatMessage]]:
        """获取聊天会话"""
        return self.chat_sessions.get(session_id)
    
    def get_current_session(self) -> Optional[List[ChatMessage]]:
        """获取当前会话"""
        if self.current_session_id:
            return self.chat_sessions.get(self.current_session_id)
        return None
    
    def get_session_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取会话消息"""
        
        messages = self.chat_sessions.get(session_id, [])
        
        # 限制返回数量
        if limit > 0:
            messages = messages[-limit:]
        
        return [msg.to_dict() for msg in messages]
    
    async def delete_message(self, session_id: str, message_id: str) -> bool:
        """删除消息"""
        
        if session_id not in self.chat_sessions:
            return False
        
        messages = self.chat_sessions[session_id]
        for i, msg in enumerate(messages):
            if msg.id == message_id:
                del messages[i]
                return True
        
        return False
    
    async def clear_session(self, session_id: str) -> bool:
        """清空会话"""
        
        if session_id in self.chat_sessions:
            # 保留欢迎消息
            welcome_messages = [
                msg for msg in self.chat_sessions[session_id]
                if msg.metadata.get("type") == "system_welcome"
            ]
            
            self.chat_sessions[session_id] = welcome_messages
            return True
        
        return False
    
    async def switch_session(self, session_id: str) -> bool:
        """切换会话"""
        
        if session_id in self.chat_sessions:
            self.current_session_id = session_id
            self.logger.info(f"切换到会话: {session_id}")
            return True
        
        return False
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """获取所有会话信息"""
        
        sessions_info = []
        
        for session_id, messages in self.chat_sessions.items():
            # 计算会话统计
            user_messages = [msg for msg in messages if msg.sender == "user"]
            assistant_messages = [msg for msg in messages if msg.sender == "assistant"]
            
            last_message = messages[-1] if messages else None
            
            session_info = {
                "session_id": session_id,
                "message_count": len(messages),
                "user_message_count": len(user_messages),
                "assistant_message_count": len(assistant_messages),
                "is_current": session_id == self.current_session_id,
                "last_activity": last_message.timestamp.isoformat() if last_message else None,
                "created_at": messages[0].timestamp.isoformat() if messages else None
            }
            
            sessions_info.append(session_info)
        
        return sorted(sessions_info, key=lambda x: x["last_activity"] or "", reverse=True)
    
    def export_session(self, session_id: str, format: str = "json") -> Optional[str]:
        """导出会话"""
        
        if session_id not in self.chat_sessions:
            return None
        
        messages = self.chat_sessions[session_id]
        
        if format == "json":
            export_data = {
                "session_id": session_id,
                "exported_at": datetime.now().isoformat(),
                "message_count": len(messages),
                "messages": [msg.to_dict() for msg in messages]
            }
            return json.dumps(export_data, ensure_ascii=False, indent=2)
        
        elif format == "text":
            # 导出为纯文本格式
            lines = [f"聊天会话导出 - {session_id}"]
            lines.append(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"消息数量: {len(messages)}")
            lines.append("-" * 50)
            
            for msg in messages:
                timestamp = msg.timestamp.strftime("%H:%M:%S")
                sender_name = "用户" if msg.sender == "user" else "诺玛"
                lines.append(f"[{timestamp}] {sender_name}: {msg.content}")
            
            return "\n".join(lines)
        
        return None
    
    def get_chat_stats(self) -> Dict[str, Any]:
        """获取聊天统计"""
        
        total_sessions = len(self.chat_sessions)
        total_messages = sum(len(messages) for messages in self.chat_sessions.values())
        
        # 按会话统计
        session_stats = []
        for session_id, messages in self.chat_sessions.items():
            user_count = len([msg for msg in messages if msg.sender == "user"])
            assistant_count = len([msg for msg in messages if msg.sender == "assistant"])
            
            session_stats.append({
                "session_id": session_id,
                "total_messages": len(messages),
                "user_messages": user_count,
                "assistant_messages": assistant_count
            })
        
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "current_session": self.current_session_id,
            "is_streaming": self.is_streaming,
            "session_stats": session_stats,
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """更新配置"""
        self.config.update(new_config)
        self.logger.info(f"更新聊天界面配置: {new_config}")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        
        return {
            "component": "chat_interface",
            "status": "active" if self.is_active else "inactive",
            "total_sessions": len(self.chat_sessions),
            "current_session": self.current_session_id,
            "is_streaming": self.is_streaming,
            "config": self.config
        }

class ChatUIManager:
    """聊天UI管理器"""
    
    def __init__(self, chat_interface: ChatInterface):
        """初始化聊天UI管理器
        
        Args:
            chat_interface: 聊天界面实例
        """
        self.chat_interface = chat_interface
        self.logger = NormaLogger("chat_ui_manager")
        
        # UI状态
        self.active_connections: Dict[str, Any] = {}
        self.ui_components: Dict[str, Any] = {}
        
        # 主题和样式
        self.theme_config = {
            "primary_color": "#3b82f6",
            "secondary_color": "#6b7280",
            "background_color": "#ffffff",
            "text_color": "#1f2937",
            "border_color": "#e5e7eb",
            "font_family": "system-ui, -apple-system, sans-serif"
        }
        
        # 快捷命令
        self.quick_commands = {
            "/help": "显示帮助信息",
            "/status": "查看系统状态",
            "/clear": "清空当前对话",
            "/export": "导出对话记录",
            "/blood": "血统分析",
            "/security": "安全检查",
            "/multimodal": "多模态感知"
        }
    
    def render_message(self, message: ChatMessage) -> Dict[str, Any]:
        """渲染消息"""
        
        return {
            "id": message.id,
            "content": message.content,
            "sender": message.sender,
            "sender_display": "用户" if message.sender == "user" else "诺玛·劳恩斯",
            "timestamp": message.timestamp.isoformat(),
            "timestamp_display": message.timestamp.strftime("%H:%M"),
            "message_type": message.message_type,
            "metadata": message.metadata,
            "is_streaming": message.is_streaming,
            "streaming_content": message.streaming_content,
            "avatar_url": "/avatars/user.png" if message.sender == "user" else "/avatars/norma.png",
            "bg_color": "#f3f4f6" if message.sender == "user" else "#dbeafe",
            "text_color": "#1f2937"
        }
    
    def render_chat_header(self, session_id: str) -> Dict[str, Any]:
        """渲染聊天头部"""
        
        messages = self.chat_interface.get_chat_session(session_id)
        if not messages:
            return {}
        
        last_message = messages[-1]
        
        return {
            "session_id": session_id,
            "title": "与诺玛·劳恩斯的对话",
            "subtitle": f"最后活动: {last_message.timestamp.strftime('%Y-%m-%d %H:%M')}",
            "status": "在线",
            "avatar_url": "/avatars/norma.png",
            "agent_name": "诺玛·劳恩斯",
            "agent_title": "卡塞尔学院主控计算机AI系统"
        }
    
    def render_input_area(self) -> Dict[str, Any]:
        """渲染输入区域"""
        
        return {
            "placeholder": "输入消息... (使用 /help 查看快捷命令)",
            "send_button_text": "发送",
            "multimodal_buttons": [
                {"type": "image", "icon": "📷", "label": "图片"},
                {"type": "audio", "icon": "🎤", "label": "语音"},
                {"type": "file", "icon": "📎", "label": "文件"}
            ],
            "quick_commands": list(self.quick_commands.keys()),
            "streaming_indicator": self.chat_interface.is_streaming
        }
    
    def render_sidebar(self) -> Dict[str, Any]:
        """渲染侧边栏"""
        
        sessions = self.chat_interface.get_all_sessions()
        
        return {
            "sessions": sessions,
            "current_session": self.chat_interface.current_session_id,
            "new_chat_button": {
                "text": "新建对话",
                "action": "create_session"
            },
            "stats": {
                "total_sessions": len(sessions),
                "total_messages": sum(s["message_count"] for s in sessions)
            }
        }
    
    def render_settings_panel(self) -> Dict[str, Any]:
        """渲染设置面板"""
        
        return {
            "theme": self.theme_config,
            "chat_config": self.chat_interface.config,
            "quick_commands": self.quick_commands,
            "export_formats": ["json", "text"],
            "available_actions": [
                {"id": "clear_session", "text": "清空当前对话", "icon": "🗑️"},
                {"id": "export_session", "text": "导出对话记录", "icon": "💾"},
                {"id": "delete_session", "text": "删除对话", "icon": "❌"},
                {"id": "view_stats", "text": "查看统计", "icon": "📊"}
            ]
        }
    
    def get_ui_state(self) -> Dict[str, Any]:
        """获取完整UI状态"""
        
        current_session = self.chat_interface.current_session_id
        
        return {
            "chat": {
                "current_session": current_session,
                "messages": self.chat_interface.get_session_messages(current_session) if current_session else [],
                "header": self.render_chat_header(current_session) if current_session else {},
                "input_area": self.render_input_area(),
                "is_streaming": self.chat_interface.is_streaming
            },
            "sidebar": self.render_sidebar(),
            "settings": self.render_settings_panel(),
            "theme": self.theme_config,
            "timestamp": datetime.now().isoformat()
        }
    
    async def handle_ui_action(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理UI动作"""
        
        try:
            if action == "create_session":
                session_id = await self.chat_interface.create_chat_session()
                return {"success": True, "session_id": session_id}
            
            elif action == "switch_session":
                session_id = data.get("session_id")
                success = await self.chat_interface.switch_session(session_id)
                return {"success": success, "session_id": session_id}
            
            elif action == "clear_session":
                session_id = self.chat_interface.current_session_id
                if session_id:
                    success = await self.chat_interface.clear_session(session_id)
                    return {"success": success}
            
            elif action == "export_session":
                session_id = self.chat_interface.current_session_id
                format_type = data.get("format", "json")
                if session_id:
                    exported_data = self.chat_interface.export_session(session_id, format_type)
                    return {"success": True, "data": exported_data, "format": format_type}
            
            elif action == "delete_session":
                session_id = data.get("session_id")
                if session_id in self.chat_interface.chat_sessions:
                    del self.chat_interface.chat_sessions[session_id]
                    return {"success": True, "session_id": session_id}
            
            elif action == "update_config":
                config_updates = data.get("config", {})
                self.chat_interface.update_config(config_updates)
                return {"success": True, "config": config_updates}
            
            else:
                return {"success": False, "error": f"未知动作: {action}"}
        
        except Exception as e:
            self.logger.error(f"UI动作处理失败 {action}: {e}")
            return {"success": False, "error": str(e)}
    
    def update_theme(self, new_theme: Dict[str, Any]) -> None:
        """更新主题"""
        self.theme_config.update(new_theme)
        self.logger.info(f"更新UI主题: {new_theme}")
    
    def get_quick_command_help(self) -> str:
        """获取快捷命令帮助"""
        
        help_text = "快捷命令:\n"
        for cmd, desc in self.quick_commands.items():
            help_text += f"{cmd} - {desc}\n"
        
        return help_text