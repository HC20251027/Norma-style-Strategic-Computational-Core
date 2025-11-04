"""
文本处理模块
支持文本预处理、验证、格式化和优化
"""

import re
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

from ..utils import calculate_tokens_estimate

logger = logging.getLogger(__name__)

@dataclass
class TextProcessingConfig:
    """文本处理配置"""
    max_length: int = 4096
    min_length: int = 1
    remove_special_chars: bool = False
    normalize_whitespace: bool = True
    convert_to_lowercase: bool = False
    enable_spelling_check: bool = False
    enable_grammar_check: bool = False
    language: str = "auto"  # auto, zh, en, ja, ko
    encoding: str = "utf-8"

class TextProcessor:
    """文本处理器"""
    
    def __init__(self, config: TextProcessingConfig = None):
        self.config = config or TextProcessingConfig()
        self.supported_languages = ['zh', 'en', 'ja', 'ko', 'auto']
        
        # 特殊字符映射
        self.special_char_map = {
            '“': '"', '”': '"', '‘': "'", '’': "'",
            '—': '-', '–': '-', '…': '...',
            '《': '<', '》': '>', '【': '[', '】': ']',
            '、': ',', '。': '.', '！': '!', '？': '?'
        }
        
        # 停用词列表（简化版）
        self.stopwords = {
            'zh': {'的', '了', '是', '在', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'},
            'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        }
    
    async def process_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        处理文本
        
        Args:
            text: 输入文本
            **kwargs: 额外参数
            
        Returns:
            处理后的文本和元数据
        """
        if not isinstance(text, str):
            raise ValueError("文本必须是字符串类型")
        
        original_text = text
        processing_steps = []
        
        # 1. 基本验证
        if not text.strip():
            raise ValueError("文本不能为空")
        
        # 2. 长度检查
        if len(text) < self.config.min_length:
            raise ValueError(f"文本长度不能少于{self.config.min_length}个字符")
        
        if len(text) > self.config.max_length:
            text = await self._truncate_text(text)
            processing_steps.append(f"截断到{self.config.max_length}字符")
        
        # 3. 标准化空白字符
        if self.config.normalize_whitespace:
            text = await self._normalize_whitespace(text)
            processing_steps.append("标准化空白字符")
        
        # 4. 移除特殊字符
        if self.config.remove_special_chars:
            text = await self._remove_special_chars(text)
            processing_steps.append("移除特殊字符")
        
        # 5. 转换大小写
        if self.config.convert_to_lowercase:
            text = text.lower()
            processing_steps.append("转换为小写")
        
        # 6. 语言检测
        detected_language = await self._detect_language(text)
        
        # 7. Token估算
        token_count = calculate_tokens_estimate(text)
        
        # 8. 质量评估
        quality_score = await self._assess_text_quality(text, detected_language)
        
        return {
            'processed_text': text,
            'original_text': original_text,
            'processing_steps': processing_steps,
            'detected_language': detected_language,
            'token_count': token_count,
            'character_count': len(text),
            'quality_score': quality_score,
            'config_used': self.config.__dict__.copy()
        }
    
    async def _truncate_text(self, text: str) -> str:
        """截断文本"""
        if len(text) <= self.config.max_length:
            return text
        
        # 尝试在句子边界截断
        truncated = text[:self.config.max_length]
        
        # 查找最后一个句号、问号或感叹号
        last_boundary = max(
            truncated.rfind('。'),
            truncated.rfind('.'),
            truncated.rfind('?'),
            truncated.rfind('!'),
            truncated.rfind('\n')
        )
        
        if last_boundary > self.config.max_length * 0.8:  # 如果句边界不要太靠前
            return truncated[:last_boundary + 1]
        
        return truncated + "..."
    
    async def _normalize_whitespace(self, text: str) -> str:
        """标准化空白字符"""
        # 替换多个空格为单个空格
        text = re.sub(r'\s+', ' ', text)
        
        # 移除行首行尾空白
        text = text.strip()
        
        # 标准化换行符
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # 移除多余的空行
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text
    
    async def _remove_special_chars(self, text: str) -> str:
        """移除特殊字符"""
        # 映射常见特殊字符
        for old_char, new_char in self.special_char_map.items():
            text = text.replace(old_char, new_char)
        
        # 移除控制字符
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text
    
    async def _detect_language(self, text: str) -> str:
        """检测语言"""
        if self.config.language != "auto":
            return self.config.language
        
        # 简单的语言检测逻辑
        chinese_chars = sum(1 for c in text if ord(c) > 127)
        total_chars = len(text)
        
        if total_chars == 0:
            return "unknown"
        
        chinese_ratio = chinese_chars / total_chars
        
        if chinese_ratio > 0.3:
            return "zh"
        elif chinese_ratio < 0.1:
            return "en"
        else:
            return "mixed"
    
    async def _assess_text_quality(self, text: str, language: str) -> float:
        """评估文本质量"""
        score = 1.0
        
        # 长度分数
        length_score = min(len(text) / 100, 1.0)  # 100字符为满分
        score *= 0.3 + 0.7 * length_score
        
        # 重复字符惩罚
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        max_repetition = max(char_counts.values()) if char_counts else 1
        repetition_penalty = min(max_repetition / len(text), 0.5)
        score *= (1 - repetition_penalty)
        
        # 标点符号分数
        punctuation_count = len(re.findall(r'[.!?。！？，,]', text))
        punctuation_score = min(punctuation_count / (len(text) / 10), 1.0)
        score *= 0.8 + 0.2 * punctuation_score
        
        return max(0.0, min(1.0, score))
    
    async def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量处理文本"""
        tasks = [self.process_text(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'processed_text': texts[i],
                    'original_text': texts[i],
                    'error': str(result),
                    'processing_steps': [],
                    'detected_language': 'unknown',
                    'token_count': 0,
                    'character_count': len(texts[i]),
                    'quality_score': 0.0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def extract_keywords(self, text: str, language: str = "auto", max_keywords: int = 10) -> List[Dict[str, Any]]:
        """提取关键词"""
        if language == "auto":
            language = await self._detect_language(text)
        
        # 分词（简化实现）
        if language == "zh":
            words = await self._chinese_tokenize(text)
        else:
            words = await self._english_tokenize(text)
        
        # 计算词频
        word_freq = {}
        for word in words:
            if len(word) < 2:  # 忽略单字符
                continue
            if language in self.stopwords and word in self.stopwords[language]:
                continue
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 返回关键词及其权重
        keywords = []
        for word, freq in sorted_words[:max_keywords]:
            keywords.append({
                'word': word,
                'frequency': freq,
                'weight': freq / len(words) if words else 0
            })
        
        return keywords
    
    async def _chinese_tokenize(self, text: str) -> List[str]:
        """中文分词（简化版）"""
        # 简单的中文字符分割
        words = []
        current_word = ""
        
        for char in text:
            if re.match(r'[\u4e00-\u9fa5]', char):  # 中文字符
                current_word += char
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                # 处理标点符号
                if not char.isspace():
                    words.append(char)
        
        if current_word:
            words.append(current_word)
        
        return words
    
    async def _english_tokenize(self, text: str) -> List[str]:
        """英文分词"""
        # 使用正则表达式分词
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """文本摘要"""
        # 简单的摘要算法
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 3:
            return text
        
        # 选择前几句作为摘要
        summary_sentences = sentences[:min(3, len(sentences))]
        summary = '。'.join(summary_sentences)
        
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary
    
    async def translate_text(self, text: str, target_language: str) -> str:
        """文本翻译（占位符）"""
        # 这里应该集成翻译API
        # 简化实现：返回原文本并添加标记
        return f"[翻译到{target_language}]{text}[翻译结束]"
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        logger.info(f"文本处理配置已更新: {kwargs}")
    
    def get_supported_languages(self) -> List[str]:
        """获取支持的语言"""
        return self.supported_languages.copy()
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.__dict__.copy()