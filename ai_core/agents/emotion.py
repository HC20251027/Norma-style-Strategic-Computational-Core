"""
语音情感和语调控制器
提供对语音情感、语调、语速等参数的精确控制
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
import json

logger = logging.getLogger(__name__)


class EmotionType(Enum):
    """情感类型"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    CALM = "calm"
    SURPRISED = "surprised"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"


class VoiceStyle(Enum):
    """语音风格"""
    NARRATOR = "narrator"
    CONVERSATION = "conversation"
    NEWS = "news"
    STORYTELLING = "storytelling"
    INSTRUCTIONAL = "instructional"
    ENTERTAINMENT = "entertainment"


@dataclass
class EmotionConfig:
    """情感配置"""
    emotion: EmotionType
    intensity: float  # 0.0-1.0
    pitch_shift: float  # 半音偏移
    speed_factor: float  # 语速因子
    volume_factor: float  # 音量因子
    breath_pause: float  # 呼吸停顿时间(秒)
    emphasis_words: List[str]  # 需要强调的词汇


@dataclass
class VoiceStyleConfig:
    """语音风格配置"""
    style: VoiceStyle
    pitch_range: Tuple[float, float]  # 音高范围(半音)
    speed_range: Tuple[float, float]  # 语速范围
    volume_range: Tuple[float, float]  # 音量范围
    pause_patterns: List[Tuple[str, float]]  # 停顿模式 (正则, 停顿时间)
    ssml_tags: List[str]  # SSML标签


class EmotionController:
    """语音情感和语调控制器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.emotion_configs = self._load_emotion_configs()
        self.style_configs = self._load_style_configs()
        self.custom_patterns = self._load_custom_patterns()
        
    def _load_emotion_configs(self) -> Dict[EmotionType, EmotionConfig]:
        """加载情感配置"""
        default_configs = {
            EmotionType.HAPPY: EmotionConfig(
                emotion=EmotionType.HAPPY,
                intensity=0.8,
                pitch_shift=2.0,
                speed_factor=1.1,
                volume_factor=1.2,
                breath_pause=0.1,
                emphasis_words=["太好了", "非常", "真的", "太棒了"]
            ),
            EmotionType.SAD: EmotionConfig(
                emotion=EmotionType.SAD,
                intensity=0.7,
                pitch_shift=-3.0,
                speed_factor=0.8,
                volume_factor=0.7,
                breath_pause=0.3,
                emphasis_words=["遗憾", "可惜", "难过"]
            ),
            EmotionType.ANGRY: EmotionConfig(
                emotion=EmotionType.ANGRY,
                intensity=0.9,
                pitch_shift=1.0,
                speed_factor=1.3,
                volume_factor=1.5,
                breath_pause=0.05,
                emphasis_words=["绝对", "必须", "一定", "坚决"]
            ),
            EmotionType.NEUTRAL: EmotionConfig(
                emotion=EmotionType.NEUTRAL,
                intensity=0.5,
                pitch_shift=0.0,
                speed_factor=1.0,
                volume_factor=1.0,
                breath_pause=0.2,
                emphasis_words=[]
            ),
            EmotionType.EXCITED: EmotionConfig(
                emotion=EmotionType.EXCITED,
                intensity=1.0,
                pitch_shift=4.0,
                speed_factor=1.2,
                volume_factor=1.3,
                breath_pause=0.05,
                emphasis_words=["太", "非常", "超级", "特别"]
            ),
            EmotionType.CALM: EmotionConfig(
                emotion=EmotionType.CALM,
                intensity=0.3,
                pitch_shift=-1.0,
                speed_factor=0.9,
                volume_factor=0.9,
                breath_pause=0.4,
                emphasis_words=["请", "建议", "可以", "也许"]
            ),
            EmotionType.SURPRISED: EmotionConfig(
                emotion=EmotionType.SURPRISED,
                intensity=0.9,
                pitch_shift=3.0,
                speed_factor=1.4,
                volume_factor=1.1,
                breath_pause=0.2,
                emphasis_words=["竟然", "居然", "没想到", "竟然"]
            ),
            EmotionType.FEARFUL: EmotionConfig(
                emotion=EmotionType.FEARFUL,
                intensity=0.8,
                pitch_shift=1.0,
                speed_factor=1.1,
                volume_factor=0.8,
                breath_pause=0.3,
                emphasis_words=["小心", "注意", "危险", "害怕"]
            ),
            EmotionType.DISGUSTED: EmotionConfig(
                emotion=EmotionType.DISGUSTED,
                intensity=0.7,
                pitch_shift=-2.0,
                speed_factor=1.0,
                volume_factor=1.1,
                breath_pause=0.25,
                emphasis_words=["恶心", "讨厌", "反感", "不屑"]
            )
        }
        
        # 加载自定义配置
        custom_configs = self.config.get("emotion_configs", {})
        for emotion_str, config_dict in custom_configs.items():
            try:
                emotion = EmotionType(emotion_str)
                default_configs[emotion] = EmotionConfig(**config_dict)
            except (ValueError, TypeError) as e:
                logger.warning(f"加载情感配置失败 {emotion_str}: {e}")
        
        return default_configs
    
    def _load_style_configs(self) -> Dict[VoiceStyle, VoiceStyleConfig]:
        """加载语音风格配置"""
        return {
            VoiceStyle.NARRATOR: VoiceStyleConfig(
                style=VoiceStyle.NARRATOR,
                pitch_range=(-2.0, 2.0),
                speed_range=(0.9, 1.1),
                volume_range=(0.9, 1.1),
                pause_patterns=[
                    (r'[。！？]', 0.5),
                    (r'[，；]', 0.2),
                    (r'[:：]', 0.3)
                ],
                ssml_tags=["<break time='0.5s'/>", "<prosody rate='medium'/>"]
            ),
            VoiceStyle.CONVERSATION: VoiceStyleConfig(
                style=VoiceStyle.CONVERSATION,
                pitch_range=(-1.0, 1.0),
                speed_range=(1.0, 1.2),
                volume_range=(1.0, 1.2),
                pause_patterns=[
                    (r'[。！？]', 0.3),
                    (r'[，]', 0.15),
                    (r'[嗯啊]', 0.1)
                ],
                ssml_tags=["<prosody rate='fast'/>", "<break time='0.3s'/>"]
            ),
            VoiceStyle.NEWS: VoiceStyleConfig(
                style=VoiceStyle.NEWS,
                pitch_range=(-1.0, 0.0),
                speed_range=(1.0, 1.1),
                volume_range=(1.0, 1.1),
                pause_patterns=[
                    (r'[。！？]', 0.4),
                    (r'[，]', 0.15),
                    (r'[：]', 0.25)
                ],
                ssml_tags=["<prosody rate='medium'/>", "<break time='0.4s'/>"]
            ),
            VoiceStyle.STORYTELLING: VoiceStyleConfig(
                style=VoiceStyle.STORYTELLING,
                pitch_range=(-3.0, 3.0),
                speed_range=(0.8, 1.0),
                volume_range=(0.9, 1.3),
                pause_patterns=[
                    (r'[。！？]', 0.8),
                    (r'[，]', 0.3),
                    (r'[…]', 1.0)
                ],
                ssml_tags=["<prosody rate='slow'/>", "<break time='0.8s'/>"]
            ),
            VoiceStyle.INSTRUCTIONAL: VoiceStyleConfig(
                style=VoiceStyle.INSTRUCTIONAL,
                pitch_range=(-1.0, 1.0),
                speed_range=(0.9, 1.0),
                volume_range=(1.0, 1.1),
                pause_patterns=[
                    (r'[。！？]', 0.6),
                    (r'[，]', 0.25),
                    (r'[：]', 0.4)
                ],
                ssml_tags=["<prosody rate='medium'/>", "<break time='0.6s'/>"]
            ),
            VoiceStyle.ENTERTAINMENT: VoiceStyleConfig(
                style=VoiceStyle.ENTERTAINMENT,
                pitch_range=(-2.0, 4.0),
                speed_range=(1.1, 1.4),
                volume_range=(1.1, 1.4),
                pause_patterns=[
                    (r'[。！？]', 0.2),
                    (r'[，]', 0.1),
                    (r'[哈哈]', 0.1)
                ],
                ssml_tags=["<prosody rate='fast'/>", "<break time='0.2s'/>"]
            )
        }
    
    def _load_custom_patterns(self) -> Dict[str, Any]:
        """加载自定义模式"""
        return self.config.get("custom_patterns", {})
    
    def get_emotion_config(self, emotion: EmotionType) -> EmotionConfig:
        """获取情感配置"""
        return self.emotion_configs.get(emotion, self.emotion_configs[EmotionType.NEUTRAL])
    
    def get_style_config(self, style: VoiceStyle) -> VoiceStyleConfig:
        """获取风格配置"""
        return self.style_configs.get(style, self.style_configs[VoiceStyle.CONVERSATION])
    
    def analyze_text_emotion(self, text: str) -> EmotionType:
        """分析文本情感"""
        # 简单的关键词匹配分析
        emotion_keywords = {
            EmotionType.HAPPY: ["开心", "高兴", "快乐", "愉快", "兴奋", "太好了", "棒"],
            EmotionType.SAD: ["难过", "伤心", "失望", "遗憾", "可惜", "沮丧"],
            EmotionType.ANGRY: ["生气", "愤怒", "气愤", "恼火", "愤怒", "讨厌"],
            EmotionType.EXCITED: ["激动", "兴奋", "狂热", "热情", "激动"],
            EmotionType.SURPRISED: ["惊讶", "吃惊", "意外", "没想到", "居然"],
            EmotionType.FEARFUL: ["害怕", "恐惧", "担心", "紧张", "忧虑"],
            EmotionType.DISGUSTED: ["恶心", "讨厌", "反感", "厌恶", "不屑"]
        }
        
        emotion_scores = {emotion: 0 for emotion in emotion_keywords}
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                emotion_scores[emotion] += text.count(keyword)
        
        # 返回得分最高的情感
        max_emotion = max(emotion_scores, key=emotion_scores.get)
        if emotion_scores[max_emotion] > 0:
            return max_emotion
        else:
            return EmotionType.NEUTRAL
    
    def apply_emotion_to_ssml(
        self,
        text: str,
        emotion: EmotionType,
        style: Optional[VoiceStyle] = None,
        custom_config: Optional[EmotionConfig] = None
    ) -> str:
        """将情感应用到SSML文本"""
        if custom_config:
            emotion_config = custom_config
        else:
            emotion_config = self.get_emotion_config(emotion)
        
        # 计算SSML参数
        pitch = emotion_config.pitch_shift
        rate = emotion_config.speed_factor
        volume = emotion_config.volume_factor
        
        # 转换为SSML格式
        rate_str = f"{int((rate - 1.0) * 100)}%"
        volume_str = f"{int((volume - 1.0) * 100)}%"
        pitch_str = f"{int(pitch)}st"
        
        # 处理强调词汇
        emphasized_text = self._apply_emphasis(text, emotion_config.emphasis_words)
        
        # 添加停顿
        emphasized_text = self._apply_pauses(emphasized_text, emotion_config.breath_pause)
        
        # 应用风格
        if style:
            emphasized_text = self._apply_style(emphasized_text, style)
        
        # 构建SSML
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis">
            <voice>
                <prosody rate="{rate_str}" volume="{volume_str}" pitch="{pitch_str}">
                    {emphasized_text}
                </prosody>
            </voice>
        </speak>
        """
        
        return ssml.strip()
    
    def _apply_emphasis(self, text: str, emphasis_words: List[str]) -> str:
        """应用词汇强调"""
        emphasized_text = text
        
        for word in emphasis_words:
            if word in text:
                # 使用SSML emphasis标签
                emphasized_text = emphasized_text.replace(
                    word, f'<emphasis level="moderate">{word}</emphasis>'
                )
        
        return emphasized_text
    
    def _apply_pauses(self, text: str, breath_pause: float) -> str:
        """应用呼吸停顿"""
        # 在句号、问号、感叹号后添加停顿
        pause_tag = f'<break time="{breath_pause}s"/>'
        
        # 替换标点符号
        text = re.sub(r'[。！？]', lambda m: m.group() + pause_tag, text)
        text = re.sub(r'[，；]', lambda m: m.group() + '<break time="0.2s"/>', text)
        
        return text
    
    def _apply_style(self, text: str, style: VoiceStyle) -> str:
        """应用语音风格"""
        style_config = self.get_style_config(style)
        
        # 应用停顿模式
        for pattern, pause_time in style_config.pause_patterns:
            text = re.sub(pattern, lambda m: m.group() + f'<break time="{pause_time}s"/>', text)
        
        return text
    
    def create_emotion_mix(
        self,
        primary_emotion: EmotionType,
        secondary_emotion: Optional[EmotionType] = None,
        mix_ratio: float = 0.7
    ) -> EmotionConfig:
        """创建混合情感配置"""
        primary_config = self.get_emotion_config(primary_emotion)
        
        if secondary_emotion is None:
            return primary_config
        
        secondary_config = self.get_emotion_config(secondary_emotion)
        
        # 混合参数
        return EmotionConfig(
            emotion=primary_emotion,
            intensity=primary_config.intensity * mix_ratio + secondary_config.intensity * (1 - mix_ratio),
            pitch_shift=primary_config.pitch_shift * mix_ratio + secondary_config.pitch_shift * (1 - mix_ratio),
            speed_factor=primary_config.speed_factor * mix_ratio + secondary_config.speed_factor * (1 - mix_ratio),
            volume_factor=primary_config.volume_factor * mix_ratio + secondary_config.volume_factor * (1 - mix_ratio),
            breath_pause=primary_config.breath_pause * mix_ratio + secondary_config.breath_pause * (1 - mix_ratio),
            emphasis_words=list(set(primary_config.emphasis_words + secondary_config.emphasis_words))
        )
    
    def generate_emotion_progression(
        self,
        text: str,
        emotions: List[EmotionType],
        segments: Optional[List[int]] = None
    ) -> List[Tuple[str, EmotionType]]:
        """生成情感渐进序列"""
        if segments is None:
            # 平均分段
            segment_length = len(text) // len(emotions)
            segments = [i * segment_length for i in range(len(emotions) + 1)]
            segments[-1] = len(text)
        
        result = []
        for i, emotion in enumerate(emotions):
            start = segments[i]
            end = segments[i + 1] if i + 1 < len(segments) else len(text)
            segment_text = text[start:end].strip()
            if segment_text:
                result.append((segment_text, emotion))
        
        return result
    
    def optimize_for_context(
        self,
        text: str,
        context: Dict[str, Any]
    ) -> Tuple[EmotionType, VoiceStyle, EmotionConfig]:
        """根据上下文优化情感和风格"""
        # 分析上下文
        text_type = context.get("type", "conversation")  # conversation, narration, news, etc.
        audience = context.get("audience", "general")  # general, children, professional
        setting = context.get("setting", "casual")  # formal, casual, entertainment
        
        # 选择合适的风格
        style_mapping = {
            "narration": VoiceStyle.STORYTELLING,
            "news": VoiceStyle.NEWS,
            "instruction": VoiceStyle.INSTRUCTIONAL,
            "entertainment": VoiceStyle.ENTERTAINMENT,
            "conversation": VoiceStyle.CONVERSATION
        }
        
        style = style_mapping.get(text_type, VoiceStyle.CONVERSATION)
        
        # 分析情感
        emotion = self.analyze_text_emotion(text)
        
        # 根据受众调整
        if audience == "children":
            # 儿童内容更活泼
            emotion = EmotionType.EXCITED if emotion == EmotionType.NEUTRAL else emotion
        elif audience == "professional":
            # 专业内容更正式
            emotion = EmotionType.CALM if emotion in [EmotionType.EXCITED, EmotionType.HAPPY] else emotion
        
        # 根据场合调整
        if setting == "formal":
            style = VoiceStyle.NARRATOR if style == VoiceStyle.CONVERSATION else style
        
        # 创建优化的配置
        emotion_config = self.get_emotion_config(emotion)
        
        return emotion, style, emotion_config
    
    def validate_emotion_config(self, config: EmotionConfig) -> Tuple[bool, List[str]]:
        """验证情感配置"""
        errors = []
        
        # 验证强度范围
        if not 0.0 <= config.intensity <= 1.0:
            errors.append(f"强度必须在0.0-1.0范围内，当前值: {config.intensity}")
        
        # 验证音高偏移范围
        if not -12.0 <= config.pitch_shift <= 12.0:
            errors.append(f"音高偏移必须在-12到12半音范围内，当前值: {config.pitch_shift}")
        
        # 验证语速因子范围
        if not 0.5 <= config.speed_factor <= 2.0:
            errors.append(f"语速因子必须在0.5-2.0范围内，当前值: {config.speed_factor}")
        
        # 验证音量因子范围
        if not 0.5 <= config.volume_factor <= 2.0:
            errors.append(f"音量因子必须在0.5-2.0范围内，当前值: {config.volume_factor}")
        
        # 验证停顿时间范围
        if not 0.0 <= config.breath_pause <= 2.0:
            errors.append(f"停顿时间必须在0.0-2.0秒范围内，当前值: {config.breath_pause}")
        
        return len(errors) == 0, errors