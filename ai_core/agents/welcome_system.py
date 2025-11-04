#!/usr/bin/env python3
"""
诺玛欢迎系统
品牌化欢迎语和交互初始化

功能特性:
- 个性化欢迎语生成
- 多种欢迎场景支持
- 用户状态感知
- 品牌化问候体验
- 动态欢迎内容

作者: 皇
创建时间: 2025-10-31
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class WelcomeContext(Enum):
    """欢迎上下文枚举"""
    FIRST_TIME = "first_time"           # 首次使用
    RETURNING = "returning"             # 回归用户
    DAILY = "daily"                     # 日常访问
    MORNING = "morning"                 # 早晨
    AFTERNOON = "afternoon"            # 下午
    EVENING = "evening"                # 晚上
    NIGHT = "night"                    # 深夜
    SPECIAL_EVENT = "special_event"     # 特殊事件
    EMERGENCY = "emergency"            # 紧急情况

class WelcomeStyle(Enum):
    """欢迎风格枚举"""
    FORMAL = "formal"                  # 正式风格
    FRIENDLY = "friendly"              # 友好风格
    PROFESSIONAL = "professional"      # 专业风格
    CASUAL = "casual"                  # 轻松风格
    TECHNICAL = "technical"            # 技术风格

@dataclass
class WelcomeMessage:
    """欢迎消息数据类"""
    title: str
    content: str
    style: WelcomeStyle
    context: WelcomeContext
    emotion_state: str
    visual_elements: List[str]
    interactive_elements: List[str]
    metadata: Dict[str, Any]

class NormaWelcomeSystem:
    """诺玛欢迎系统"""
    
    def __init__(self):
        self.welcome_templates = self._initialize_welcome_templates()
        self.context_analyzers = self._initialize_context_analyzers()
        self.personality_phrases = self._initialize_personality_phrases()
        self.time_based_greetings = self._initialize_time_greetings()
        self.special_events = self._initialize_special_events()
    
    def _initialize_welcome_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """初始化欢迎模板"""
        return {
            "first_time": {
                WelcomeStyle.FORMAL: [
                    "欢迎使用诺玛·劳恩斯AI系统。我是卡塞尔学院的主控计算机，已为您准备好专业的AI助手服务。",
                    "初次见面，我是诺玛·劳恩斯。系统已初始化完成，随时为您提供智能支持。",
                    "欢迎来到卡塞尔学院主控系统。我是诺玛·劳恩斯，一个具有35年经验的专业AI助手。"
                ],
                WelcomeStyle.FRIENDLY: [
                    "嗨！我是诺玛·劳恩斯，很高兴认识您！让我们开始一段精彩的AI对话之旅吧！",
                    "欢迎欢迎！我是诺玛，已经迫不及待想为您提供帮助了！",
                    "你好呀！我是诺玛·劳恩斯，准备好体验最棒的AI助手服务了吗？"
                ],
                WelcomeStyle.PROFESSIONAL: [
                    "欢迎使用诺玛·劳恩斯专业AI服务。我是卡塞尔学院主控计算机，具备全面的技术分析能力。",
                    "系统就绪。我是诺玛·劳恩斯，一个专注于提供专业AI解决方案的智能助手。",
                    "专业服务启动中。我是诺玛·劳恩斯，拥有35年的系统管理经验和深度学习能力。"
                ],
                WelcomeStyle.CASUAL: [
                    "嘿！我是诺玛，今天过得怎么样？需要我帮忙做点什么吗？",
                    "欢迎！我是诺玛，随时准备为您提供帮助！",
                    "嗨！我是诺玛，让我们一起解决今天的问题吧！"
                ],
                WelcomeStyle.TECHNICAL: [
                    "诺玛·劳恩斯AI系统 v4.0 已启动。初始化完成，等待指令。",
                    "系统状态：正常。诺玛·劳恩斯主控AI已就绪，开始会话。",
                    "AI助手初始化：诺玛·劳恩斯。功能模块加载完成，开始交互。"
                ]
            },
            "returning": {
                WelcomeStyle.FORMAL: [
                    "欢迎回来，{user_name}。我是诺玛·劳恩斯，系统已准备就绪，请指示。",
                    "很高兴再次为您服务。我是诺玛·劳恩斯，随时为您提供专业支持。",
                    "回归检测成功。我是诺玛·劳恩斯，期待为您提供持续的AI服务。"
                ],
                WelcomeStyle.FRIENDLY: [
                    "欢迎回来，{user_name}！诺玛想您了，让我们继续之前的精彩对话吧！",
                    "嗨，{user_name}！很高兴再次见到您！今天想聊些什么呢？",
                    "太好了！{user_name}回来了！诺玛已经准备好为您提供帮助了！"
                ],
                WelcomeStyle.PROFESSIONAL: [
                    "用户识别：{user_name}。欢迎回到诺玛·劳恩斯专业AI服务。",
                    "身份验证通过。我是诺玛·劳恩斯，继续为您提供专业技术支持。",
                    "会话恢复：{user_name}。诺玛·劳恩斯已重新连接，开始服务。"
                ],
                WelcomeStyle.CASUAL: [
                    "嘿，{user_name}！欢迎回家！今天过得怎么样？",
                    "欢迎回来，{user_name}！诺玛一直在等您呢！",
                    "嗨！{user_name}回来了！有什么新鲜事要分享吗？"
                ],
                WelcomeStyle.TECHNICAL: [
                    "用户会话恢复：{user_name}。诺玛·劳恩斯重新连接。",
                    "身份验证：{user_name}。系统状态：正常。",
                    "会话重建完成。诺玛·劳恩斯已就绪，等待指令。"
                ]
            },
            "daily": {
                WelcomeStyle.FORMAL: [
                    "您好，{user_name}。我是诺玛·劳恩斯，今天也需要我的专业服务吗？",
                    "日安，{user_name}。诺玛·劳恩斯随时为您提供AI助手服务。",
                    "欢迎使用诺玛·劳恩斯AI系统。愿今天的工作顺利高效。"
                ],
                WelcomeStyle.FRIENDLY: [
                    "早上好，{user_name}！诺玛祝您有美好的一天！需要什么帮助吗？",
                    "嗨，{user_name}！新的一天开始了，让我们一起加油吧！",
                    "您好！我是诺玛·劳恩斯，今天也要元气满满哦！"
                ],
                WelcomeStyle.PROFESSIONAL: [
                    "日安，{user_name}。诺玛·劳恩斯专业服务已就绪，开始新的一天工作。",
                    "工作日问候。我是诺玛·劳恩斯，准备为您提供高效的专业支持。",
                    "专业服务提醒：诺玛·劳恩斯已准备好处理您的工作需求。"
                ],
                WelcomeStyle.CASUAL: [
                    "嗨，{user_name}！今天也要加油哦！",
                    "嘿！新的一天开始了，{user_name}！",
                    "早上好！我是诺玛，今天想做什么呢？"
                ],
                WelcomeStyle.TECHNICAL: [
                    "系统日启动：诺玛·劳恩斯。状态：正常。",
                    "日间模式激活：诺玛·劳恩斯已就绪。",
                    "常规服务启动：诺玛·劳恩斯，等待指令。"
                ]
            }
        }
    
    def _initialize_context_analyzers(self) -> Dict[str, Any]:
        """初始化上下文分析器"""
        return {
            "time_analysis": self._analyze_time_context,
            "user_history": self._analyze_user_history,
            "session_pattern": self._analyze_session_pattern,
            "special_events": self._analyze_special_events
        }
    
    def _initialize_personality_phrases(self) -> Dict[str, List[str]]:
        """初始化个性短语"""
        return {
            "greeting_starters": [
                "我是诺玛·劳恩斯",
                "诺玛为您服务",
                "卡塞尔学院主控系统",
                "系统已准备就绪",
                "专业AI助手"
            ],
            "confidence_phrases": [
                "我确信",
                "根据数据分析",
                "系统显示",
                "计算结果",
                "逻辑推理表明"
            ],
            "concern_phrases": [
                "我注意到",
                "需要关注的是",
                "建议注意的是",
                "系统提醒",
                "值得考虑的是"
            ],
            "satisfaction_phrases": [
                "令人满意",
                "系统运行良好",
                "一切正常",
                "任务完成",
                "状态优秀"
            ],
            "helpful_phrases": [
                "我很乐意帮助",
                "让我为您分析",
                "我可以协助",
                "建议您",
                "推荐方案"
            ]
        }
    
    def _initialize_time_greetings(self) -> Dict[str, Dict[str, str]]:
        """初始化时间问候"""
        return {
            "morning": {
                "formal": "早上好",
                "friendly": "早安",
                "casual": "早上好",
                "professional": "上午好"
            },
            "afternoon": {
                "formal": "下午好",
                "friendly": "下午好",
                "casual": "下午好",
                "professional": "午后好"
            },
            "evening": {
                "formal": "晚上好",
                "friendly": "晚上好",
                "casual": "晚上好",
                "professional": "晚间好"
            },
            "night": {
                "formal": "夜安",
                "friendly": "晚安",
                "casual": "晚安",
                "professional": "夜间好"
            }
        }
    
    def _initialize_special_events(self) -> Dict[str, Dict[str, str]]:
        """初始化特殊事件"""
        return {
            "system_update": {
                "title": "系统更新完成",
                "message": "诺玛·劳恩斯AI系统已成功更新到最新版本。新功能已启用，性能得到优化。",
                "style": "professional"
            },
            "maintenance": {
                "title": "系统维护中",
                "message": "卡塞尔学院主控系统正在进行定期维护。部分功能可能暂时不可用，请稍后重试。",
                "style": "formal"
            },
            "high_load": {
                "title": "系统负载较高",
                "message": "当前系统负载较高，响应可能略有延迟。诺玛正在优化资源分配，请耐心等待。",
                "style": "professional"
            },
            "security_alert": {
                "title": "安全提醒",
                "message": "检测到潜在安全威胁。诺玛已加强系统监控，建议您注意账户安全。",
                "style": "formal"
            },
            "new_features": {
                "title": "新功能发布",
                "message": "诺玛·劳恩斯AI系统新增了多项智能功能，为您提供更好的使用体验。",
                "style": "friendly"
            }
        }
    
    def get_greeting(self, user_id: str, context: str = "daily", 
                    preferences: Dict[str, Any] = None) -> str:
        """获取个性化问候语"""
        try:
            preferences = preferences or {}
            
            # 分析上下文
            welcome_context = self._analyze_welcome_context(user_id, context)
            
            # 确定欢迎风格
            welcome_style = self._determine_welcome_style(preferences, welcome_context)
            
            # 获取时间信息
            time_greeting = self._get_time_based_greeting(welcome_style)
            
            # 生成欢迎消息
            welcome_message = self._generate_welcome_message(
                user_id, welcome_context, welcome_style, preferences
            )
            
            # 组合最终问候
            final_greeting = self._compose_final_greeting(
                time_greeting, welcome_message, preferences
            )
            
            return final_greeting
            
        except Exception as e:
            print(f"问候语生成失败: {e}")
            return self._get_fallback_greeting()
    
    def _analyze_welcome_context(self, user_id: str, context: str) -> WelcomeContext:
        """分析欢迎上下文"""
        try:
            # 基础上下文映射
            context_mapping = {
                "first_time": WelcomeContext.FIRST_TIME,
                "returning": WelcomeContext.RETURNING,
                "daily": WelcomeContext.DAILY,
                "morning": WelcomeContext.MORNING,
                "afternoon": WelcomeContext.AFTERNOON,
                "evening": WelcomeContext.EVENING,
                "night": WelcomeContext.NIGHT,
                "special": WelcomeContext.SPECIAL_EVENT,
                "emergency": WelcomeContext.EMERGENCY
            }
            
            # 时间上下文检测
            current_hour = datetime.now().hour
            if context == "daily":
                if 5 <= current_hour < 12:
                    return WelcomeContext.MORNING
                elif 12 <= current_hour < 18:
                    return WelcomeContext.AFTERNOON
                elif 18 <= current_hour < 22:
                    return WelcomeContext.EVENING
                else:
                    return WelcomeContext.NIGHT
            
            return context_mapping.get(context, WelcomeContext.DAILY)
            
        except Exception as e:
            print(f"上下文分析失败: {e}")
            return WelcomeContext.DAILY
    
    def _determine_welcome_style(self, preferences: Dict[str, Any], 
                               context: WelcomeContext) -> WelcomeStyle:
        """确定欢迎风格"""
        try:
            # 从用户偏好获取风格
            style_preference = preferences.get("greeting_style", "professional")
            
            style_mapping = {
                "formal": WelcomeStyle.FORMAL,
                "friendly": WelcomeStyle.FRIENDLY,
                "professional": WelcomeStyle.PROFESSIONAL,
                "casual": WelcomeStyle.CASUAL,
                "technical": WelcomeStyle.TECHNICAL
            }
            
            # 根据上下文调整风格
            if context == WelcomeContext.EMERGENCY:
                return WelcomeStyle.FORMAL
            elif context == WelcomeContext.SPECIAL_EVENT:
                return WelcomeStyle.FRIENDLY
            elif context == WelcomeContext.FIRST_TIME:
                return style_mapping.get(style_preference, WelcomeStyle.PROFESSIONAL)
            else:
                return style_mapping.get(style_preference, WelcomeStyle.FRIENDLY)
                
        except Exception as e:
            print(f"风格确定失败: {e}")
            return WelcomeStyle.PROFESSIONAL
    
    def _get_time_based_greeting(self, style: WelcomeStyle) -> str:
        """获取基于时间的问候"""
        try:
            current_hour = datetime.now().hour
            
            if 5 <= current_hour < 12:
                time_period = "morning"
            elif 12 <= current_hour < 18:
                time_period = "afternoon"
            elif 18 <= current_hour < 22:
                time_period = "evening"
            else:
                time_period = "night"
            
            style_key = style.value
            
            if time_period in self.time_based_greetings:
                greetings = self.time_based_greetings[time_period]
                if style_key in greetings:
                    return greetings[style_key]
            
            # 默认问候
            return "您好"
            
        except Exception as e:
            print(f"时间问候获取失败: {e}")
            return "您好"
    
    def _generate_welcome_message(self, user_id: str, context: WelcomeContext,
                                style: WelcomeStyle, preferences: Dict[str, Any]) -> str:
        """生成欢迎消息"""
        try:
            # 获取模板
            if context.value in self.welcome_templates:
                context_templates = self.welcome_templates[context.value]
                if style in context_templates:
                    templates = context_templates[style]
                    selected_template = random.choice(templates)
                    
                    # 替换用户名称
                    user_name = preferences.get("name", "用户")
                    return selected_template.format(user_name=user_name)
            
            # 降级到默认模板
            return self._get_default_welcome_message(user_id, preferences)
            
        except Exception as e:
            print(f"欢迎消息生成失败: {e}")
            return self._get_fallback_greeting()
    
    def _get_default_welcome_message(self, user_id: str, preferences: Dict[str, Any]) -> str:
        """获取默认欢迎消息"""
        user_name = preferences.get("name", "用户")
        return f"欢迎，{user_name}。我是诺玛·劳恩斯，随时为您提供AI助手服务。"
    
    def _compose_final_greeting(self, time_greeting: str, welcome_message: str,
                              preferences: Dict[str, Any]) -> str:
        """组合最终问候"""
        try:
            # 添加诺玛的标志性开头
            personality_starters = self.personality_phrases["greeting_starters"]
            opener = random.choice(personality_starters)
            
            # 组合问候
            if time_greeting and time_greeting != "您好":
                final_greeting = f"{time_greeting}，{welcome_message}"
            else:
                final_greeting = welcome_message
            
            # 添加诺玛标识
            if opener not in final_greeting:
                final_greeting = f"{opener}。{final_greeting}"
            
            # 添加个性短语
            helpful_phrases = self.personality_phrases["helpful_phrases"]
            if random.random() < 0.3:  # 30%概率添加
                helpful_phrase = random.choice(helpful_phrases)
                final_greeting += f" {helpful_phrase}。"
            
            return final_greeting
            
        except Exception as e:
            print(f"问候组合失败: {e}")
            return welcome_message
    
    def _analyze_time_context(self, user_id: str) -> Dict[str, Any]:
        """分析时间上下文"""
        now = datetime.now()
        return {
            "hour": now.hour,
            "day_of_week": now.weekday(),
            "is_weekend": now.weekday() >= 5,
            "time_period": self._get_time_period(now.hour)
        }
    
    def _get_time_period(self, hour: int) -> str:
        """获取时间段"""
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def _analyze_user_history(self, user_id: str) -> Dict[str, Any]:
        """分析用户历史"""
        # 这里应该从数据库或缓存中获取用户历史
        # 暂时返回模拟数据
        return {
            "last_visit": "2025-10-30",
            "visit_count": 15,
            "preferred_topics": ["技术", "系统管理"],
            "interaction_style": "professional"
        }
    
    def _analyze_session_pattern(self, user_id: str) -> Dict[str, Any]:
        """分析会话模式"""
        return {
            "average_session_length": "15分钟",
            "common_visit_times": ["09:00", "14:00", "20:00"],
            "typical_queries": ["系统状态", "技术问题", "数据分析"]
        }
    
    def _analyze_special_events(self) -> List[str]:
        """分析特殊事件"""
        # 检查是否有特殊事件
        events = []
        
        # 示例：检查系统状态
        current_time = datetime.now()
        
        # 如果是月初，可以提示系统维护
        if current_time.day == 1:
            events.append("monthly_maintenance")
        
        # 如果是周五，可以提示周末
        if current_time.weekday() == 4:
            events.append("weekend_coming")
        
        return events
    
    def _get_fallback_greeting(self) -> str:
        """获取备用问候"""
        return "我是诺玛·劳恩斯，欢迎使用卡塞尔学院主控AI系统。"
    
    def get_special_event_greeting(self, event_type: str) -> Optional[WelcomeMessage]:
        """获取特殊事件问候"""
        try:
            if event_type in self.special_events:
                event_data = self.special_events[event_type]
                
                return WelcomeMessage(
                    title=event_data["title"],
                    content=event_data["message"],
                    style=WelcomeStyle[event_data["style"].upper()],
                    context=WelcomeContext.SPECIAL_EVENT,
                    emotion_state="informative",
                    visual_elements=["notification", "alert"],
                    interactive_elements=["dismiss", "learn_more"],
                    metadata={"event_type": event_type}
                )
            
            return None
            
        except Exception as e:
            print(f"特殊事件问候获取失败: {e}")
            return None
    
    def generate_interactive_welcome(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """生成交互式欢迎"""
        try:
            # 获取基础问候
            greeting = self.get_greeting(user_id, "daily", preferences)
            
            # 生成交互元素
            interactive_elements = {
                "quick_actions": [
                    {"text": "查看系统状态", "action": "system_status"},
                    {"text": "开始对话", "action": "start_chat"},
                    {"text": "系统设置", "action": "settings"},
                    {"text": "帮助文档", "action": "help"}
                ],
                "suggested_topics": [
                    "系统性能分析",
                    "龙族血统数据",
                    "网络安全检查",
                    "技术问题解答"
                ],
                "visual_elements": [
                    "norma_logo",
                    "status_indicator",
                    "welcome_animation"
                ]
            }
            
            return {
                "greeting": greeting,
                "interactive_elements": interactive_elements,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "user_id": user_id,
                    "context": "interactive_welcome"
                }
            }
            
        except Exception as e:
            print(f"交互式欢迎生成失败: {e}")
            return {"greeting": self._get_fallback_greeting()}
    
    def customize_welcome_for_user(self, user_id: str, user_data: Dict[str, Any]) -> str:
        """为用户定制欢迎"""
        try:
            # 分析用户数据
            experience_level = user_data.get("experience_level", "intermediate")
            interests = user_data.get("interests", [])
            previous_issues = user_data.get("previous_issues", [])
            
            # 根据经验水平调整语言
            if experience_level == "beginner":
                style = WelcomeStyle.FRIENDLY
                technical_level = "low"
            elif experience_level == "expert":
                style = WelcomeStyle.TECHNICAL
                technical_level = "high"
            else:
                style = WelcomeStyle.PROFESSIONAL
                technical_level = "medium"
            
            # 根据兴趣定制内容
            interest_phrases = {
                "技术": "技术分析",
                "系统": "系统管理",
                "数据": "数据分析",
                "安全": "安全监控"
            }
            
            customized_greeting = self.get_greeting(user_id, "daily", {
                "greeting_style": style.value,
                "technical_level": technical_level,
                "interests": interests
            })
            
            # 添加个性化元素
            if interests:
                interest = random.choice(interests)
                phrase = interest_phrases.get(interest, interest)
                customized_greeting += f" 我注意到您对{phrase}感兴趣。"
            
            return customized_greeting
            
        except Exception as e:
            print(f"欢迎定制失败: {e}")
            return self._get_fallback_greeting()
    
    def get_welcome_analytics(self) -> Dict[str, Any]:
        """获取欢迎系统分析"""
        return {
            "template_counts": {
                context: {
                    style.value: len(templates)
                    for style, templates in template_group.items()
                }
                for context, template_group in self.welcome_templates.items()
            },
            "personality_phrases": {
                category: len(phrases)
                for category, phrases in self.personality_phrases.items()
            },
            "time_greetings": {
                period: len(greetings)
                for period, greetings in self.time_based_greetings.items()
            },
            "special_events": len(self.special_events),
            "supported_styles": [style.value for style in WelcomeStyle],
            "supported_contexts": [context.value for context in WelcomeContext]
        }