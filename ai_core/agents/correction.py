"""
语音识别结果智能纠错模块
提供多种纠错算法和语言模型支持
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum
import difflib
import json
from collections import Counter, defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class CorrectionType(Enum):
    """纠错类型"""
    SPELLING = "spelling"  # 拼写错误
    HOMOPHONE = "homophone"  # 同音词
    CONTEXT = "context"  # 上下文纠错
    GRAMMAR = "grammar"  # 语法错误
    DOMAIN = "domain"  # 领域特定纠错
    ACOUSTIC = "acoustic"  # 声学相似


@dataclass
class CorrectionResult:
    """纠错结果"""
    original_text: str
    corrected_text: str
    confidence: float
    corrections: List[Dict[str, Any]]
    correction_type: CorrectionType
    timestamp: float


@dataclass
class WordCorrection:
    """词汇纠错信息"""
    original_word: str
    corrected_word: str
    position: Tuple[int, int]  # (start, end)
    confidence: float
    suggestions: List[Tuple[str, float]]  # (word, confidence)
    correction_type: CorrectionType


class SpellingCorrector:
    """拼写纠错器"""
    
    def __init__(self, language: str = "zh-CN"):
        self.language = language
        self.word_freq = self._load_word_frequency()
        self.edit_distance_cache = {}
        
    def _load_word_frequency(self) -> Dict[str, int]:
        """加载词频数据"""
        # 这里应该加载真实的词频数据
        # 简化版本，返回一些常见词汇
        common_words = {
            "的": 1000000,
            "是": 800000,
            "在": 700000,
            "有": 600000,
            "和": 500000,
            "了": 900000,
            "不": 400000,
            "人": 300000,
            "我": 800000,
            "你": 600000,
            "他": 500000,
            "她": 400000,
            "它": 300000,
            "们": 350000,
            "个": 450000,
            "一": 700000,
            "二": 200000,
            "三": 150000,
            "上": 400000,
            "下": 300000,
            "中": 350000,
            "大": 250000,
            "小": 300000,
            "新": 200000,
            "旧": 150000,
            "好": 400000,
            "坏": 200000,
            "多": 300000,
            "少": 250000,
            "高": 200000,
            "低": 180000,
            "长": 220000,
            "短": 200000,
            "快": 250000,
            "慢": 200000,
            "重": 150000,
            "轻": 180000,
            "热": 200000,
            "冷": 180000,
            "明": 150000,
            "暗": 120000,
            "强": 200000,
            "弱": 180000,
            "难": 180000,
            "易": 150000,
            "远": 200000,
            "近": 180000,
            "前": 250000,
            "后": 220000,
            "左": 150000,
            "右": 150000,
            "内": 180000,
            "外": 200000
        }
        return common_words
    
    def correct_spelling(self, text: str) -> List[WordCorrection]:
        """拼写纠错"""
        corrections = []
        words = self._split_words(text)
        
        for i, word in enumerate(words):
            if self._is_punctuation(word):
                continue
                
            # 检查是否需要纠错
            if not self._is_valid_word(word):
                suggestions = self._get_spelling_suggestions(word)
                if suggestions:
                    best_suggestion = suggestions[0]
                    corrections.append(WordCorrection(
                        original_word=word,
                        corrected_word=best_suggestion[0],
                        position=self._get_word_position(text, word, i),
                        confidence=best_suggestion[1],
                        suggestions=suggestions,
                        correction_type=CorrectionType.SPELLING
                    ))
        
        return corrections
    
    def _split_words(self, text: str) -> List[str]:
        """分词"""
        # 简单的中文分词
        words = []
        current_word = ""
        
        for char in text:
            if self._is_chinese_char(char) or self._is_english_char(char):
                current_word += char
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(char)
        
        if current_word:
            words.append(current_word)
        
        return words
    
    def _is_chinese_char(self, char: str) -> bool:
        """检查是否为中文字符"""
        return '\u4e00' <= char <= '\u9fff'
    
    def _is_english_char(self, char: str) -> bool:
        """检查是否为英文字符"""
        return char.isalpha() and ord(char) < 128
    
    def _is_punctuation(self, word: str) -> bool:
        """检查是否为标点符号"""
        punctuation = "，。！？；：""''（）【】《》-——…··、"
        return all(p in punctuation for p in word) if word else True
    
    def _is_valid_word(self, word: str) -> bool:
        """检查词汇是否有效"""
        # 检查词频
        if word in self.word_freq:
            return True
        
        # 检查是否为常见模式
        if len(word) == 1:
            return True  # 单字通常有效
        
        # 检查编辑距离
        for valid_word in self.word_freq:
            if self._edit_distance(word, valid_word) <= 1:
                return True
        
        return False
    
    def _get_spelling_suggestions(self, word: str) -> List[Tuple[str, float]]:
        """获取拼写建议"""
        suggestions = []
        
        # 基于编辑距离的建议
        for valid_word, freq in self.word_freq.items():
            distance = self._edit_distance(word, valid_word)
            if distance <= 2:  # 只考虑编辑距离<=2的词汇
                # 计算置信度
                confidence = 1.0 / (1.0 + distance) * (freq / max(self.word_freq.values()))
                suggestions.append((valid_word, confidence))
        
        # 按置信度排序
        suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return suggestions[:5]  # 返回前5个建议
    
    def _edit_distance(self, word1: str, word2: str) -> int:
        """计算编辑距离"""
        cache_key = (word1, word2)
        if cache_key in self.edit_distance_cache:
            return self.edit_distance_cache[cache_key]
        
        if len(word1) == 0:
            result = len(word2)
        elif len(word2) == 0:
            result = len(word1)
        else:
            # 动态规划计算编辑距离
            matrix = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
            
            for i in range(len(word1) + 1):
                matrix[i][0] = i
            for j in range(len(word2) + 1):
                matrix[0][j] = j
            
            for i in range(1, len(word1) + 1):
                for j in range(1, len(word2) + 1):
                    if word1[i-1] == word2[j-1]:
                        matrix[i][j] = matrix[i-1][j-1]
                    else:
                        matrix[i][j] = min(
                            matrix[i-1][j] + 1,  # 删除
                            matrix[i][j-1] + 1,  # 插入
                            matrix[i-1][j-1] + 1  # 替换
                        )
            
            result = matrix[len(word1)][len(word2)]
        
        self.edit_distance_cache[cache_key] = result
        return result
    
    def _get_word_position(self, text: str, word: str, word_index: int) -> Tuple[int, int]:
        """获取词汇在文本中的位置"""
        # 简单的位置计算
        words = self._split_words(text)
        start_pos = 0
        
        for i, w in enumerate(words[:word_index]):
            start_pos += len(w)
        
        end_pos = start_pos + len(word)
        return (start_pos, end_pos)


class HomophoneCorrector:
    """同音词纠错器"""
    
    def __init__(self):
        self.homophone_groups = self._load_homophone_groups()
        
    def _load_homophone_groups(self) -> Dict[str, List[str]]:
        """加载同音词组"""
        return {
            "做": ["作", "坐"],
            "坐": ["做", "作"],
            "作": ["做", "坐"],
            "的": ["得", "地"],
            "得": ["的", "地"],
            "地": ["的", "得"],
            "在": ["再", "才"],
            "再": ["在", "才"],
            "才": ["在", "再"],
            "有": ["又", "由"],
            "又": ["有", "由"],
            "由": ["有", "又"],
            "和": ["河", "合"],
            "河": ["和", "合"],
            "合": ["和", "河"],
            "为": ["位", "未"],
            "位": ["为", "未"],
            "未": ["为", "位"],
            "要": ["药", "约"],
            "药": ["要", "约"],
            "约": ["要", "药"],
            "很": ["恨", "痕"],
            "恨": ["很", "痕"],
            "痕": ["很", "恨"],
            "听": ["厅", "亭"],
            "厅": ["听", "亭"],
            "亭": ["听", "厅"],
            "情": ["晴", "请"],
            "晴": ["情", "请"],
            "请": ["情", "晴"],
            "明": ["名", "鸣"],
            "名": ["明", "鸣"],
            "鸣": ["明", "名"],
            "心": ["新", "辛"],
            "新": ["心", "辛"],
            "辛": ["心", "新"]
        }
    
    def correct_homophones(self, text: str) -> List[WordCorrection]:
        """同音词纠错"""
        corrections = []
        words = self._split_words(text)
        
        for i, word in enumerate(words):
            if word in self.homophone_groups:
                # 基于上下文的同音词纠错
                context = self._get_context(words, i)
                best_correction = self._select_best_homophone(word, context)
                
                if best_correction and best_correction != word:
                    corrections.append(WordCorrection(
                        original_word=word,
                        corrected_word=best_correction,
                        position=self._get_word_position(text, word, i),
                        confidence=0.8,  # 同音词纠错置信度
                        suggestions=[(homophone, 0.8) for homophone in self.homophone_groups[word]],
                        correction_type=CorrectionType.HOMOPHONE
                    ))
        
        return corrections
    
    def _split_words(self, text: str) -> List[str]:
        """分词"""
        # 复用SpellingCorrector的分词逻辑
        words = []
        current_word = ""
        
        for char in text:
            if ('\u4e00' <= char <= '\u9fff') or char.isalpha():
                current_word += char
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(char)
        
        if current_word:
            words.append(current_word)
        
        return words
    
    def _get_context(self, words: List[str], index: int, window: int = 2) -> List[str]:
        """获取上下文"""
        start = max(0, index - window)
        end = min(len(words), index + window + 1)
        return words[start:end]
    
    def _select_best_homophone(self, word: str, context: List[str]) -> Optional[str]:
        """选择最佳同音词"""
        # 简化的上下文选择逻辑
        homophones = self.homophone_groups.get(word, [])
        
        # 基于词频选择
        word_freq = {
            "做": 50000, "作": 30000, "坐": 20000,
            "的": 1000000, "得": 100000, "地": 80000,
            "在": 700000, "再": 100000, "才": 80000
        }
        
        best_word = word
        best_freq = word_freq.get(word, 0)
        
        for homophone in homophones:
            freq = word_freq.get(homophone, 0)
            if freq > best_freq:
                best_word = homophone
                best_freq = freq
        
        return best_word
    
    def _get_word_position(self, text: str, word: str, word_index: int) -> Tuple[int, int]:
        """获取词汇位置"""
        words = self._split_words(text)
        start_pos = sum(len(w) for w in words[:word_index])
        end_pos = start_pos + len(word)
        return (start_pos, end_pos)


class ContextCorrector:
    """上下文纠错器"""
    
    def __init__(self):
        self.ngram_model = self._load_ngram_model()
        self.word_cooccurrence = self._load_cooccurrence()
        
    def _load_ngram_model(self) -> Dict[str, Dict[str, float]]:
        """加载N-gram模型"""
        # 简化的bigram模型
        return {
            "的": {"是": 0.3, "人": 0.2, "东": 0.1, "西": 0.1},
            "是": {"一": 0.2, "的": 0.3, "我": 0.1, "你": 0.1},
            "在": {"上": 0.2, "下": 0.1, "中": 0.2, "里": 0.1},
            "有": {"一": 0.2, "很": 0.1, "多": 0.1, "些": 0.1},
            "和": {"你": 0.2, "我": 0.2, "他": 0.1, "她": 0.1},
            "了": {"是": 0.1, "有": 0.1, "在": 0.1, "去": 0.1},
            "不": {"是": 0.2, "要": 0.1, "会": 0.1, "能": 0.1},
            "我": {"是": 0.3, "有": 0.2, "在": 0.1, "和": 0.1},
            "你": {"是": 0.3, "有": 0.2, "在": 0.1, "和": 0.1},
            "他": {"是": 0.3, "有": 0.2, "在": 0.1, "和": 0.1}
        }
    
    def _load_cooccurrence(self) -> Dict[str, Dict[str, int]]:
        """加载词共现数据"""
        return {
            "电": {"脑": 500, "话": 300, "影": 200},
            "手": {"机": 400, "表": 100, "套": 80},
            "汽": {"车": 600, "油": 200, "站": 150},
            "火": {"车": 300, "票": 200, "站": 250},
            "飞": {"机": 400, "票": 150, "场": 200},
            "图": {"书": 300, "片": 200, "像": 150},
            "音": {"乐": 400, "响": 100, "频": 80},
            "视": {"频": 300, "觉": 100, "听": 80},
            "文": {"字": 200, "件": 300, "章": 150},
            "电": {"视": 200, "影": 150, "脑": 500}
        }
    
    def correct_context(self, text: str) -> List[WordCorrection]:
        """上下文纠错"""
        corrections = []
        words = self._split_words(text)
        
        for i, word in enumerate(words):
            if len(word) != 1:  # 只考虑单字
                continue
            
            # 检查上下文一致性
            context_corrections = self._check_context_consistency(word, words, i)
            corrections.extend(context_corrections)
        
        return corrections
    
    def _split_words(self, text: str) -> List[str]:
        """分词"""
        words = []
        current_word = ""
        
        for char in text:
            if ('\u4e00' <= char <= '\u9fff') or char.isalpha():
                current_word += char
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(char)
        
        if current_word:
            words.append(current_word)
        
        return words
    
    def _check_context_consistency(
        self, 
        word: str, 
        words: List[str], 
        index: int
    ) -> List[WordCorrection]:
        """检查上下文一致性"""
        corrections = []
        
        # 检查bigram一致性
        if index > 0:
            prev_word = words[index - 1]
            if prev_word in self.ngram_model:
                word_probs = self.ngram_model[prev_word]
                if word not in word_probs:
                    # 找到最可能的词
                    best_word = max(word_probs.items(), key=lambda x: x[1])
                    if best_word[1] > 0.1:  # 阈值
                        corrections.append(WordCorrection(
                            original_word=word,
                            corrected_word=best_word[0],
                            position=(index, index + 1),
                            confidence=best_word[1],
                            suggestions=list(word_probs.items())[:3],
                            correction_type=CorrectionType.CONTEXT
                        ))
        
        # 检查词共现
        if index < len(words) - 1:
            next_word = words[index + 1]
            if word in self.word_cooccurrence:
                cooccur_words = self.word_cooccurrence[word]
                if next_word not in cooccur_words:
                    # 找到最可能共现的词
                    best_word = max(cooccur_words.items(), key=lambda x: x[1])
                    if best_word[1] > 50:  # 阈值
                        corrections.append(WordCorrection(
                            original_word=word,
                            corrected_word=best_word[0],
                            position=(index, index + 1),
                            confidence=min(best_word[1] / 100, 1.0),
                            suggestions=list(cooccur_words.items())[:3],
                            correction_type=CorrectionType.CONTEXT
                        ))
        
        return corrections


class SpeechCorrector:
    """语音识别结果智能纠错器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.spelling_corrector = SpellingCorrector(config.get("language", "zh-CN"))
        self.homophone_corrector = HomophoneCorrector()
        self.context_corrector = ContextCorrector()
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.enable_types = {
            CorrectionType.SPELLING: config.get("enable_spelling", True),
            CorrectionType.HOMOPHONE: config.get("enable_homophone", True),
            CorrectionType.CONTEXT: config.get("enable_context", True)
        }
        
    async def correct(
        self, 
        text: str,
        audio_features: Optional[Dict] = None
    ) -> CorrectionResult:
        """执行综合纠错"""
        start_time = asyncio.get_event_loop().time()
        all_corrections = []
        original_text = text
        
        # 拼写纠错
        if self.enable_types[CorrectionType.SPELLING]:
            spelling_corrections = self.spelling_corrector.correct_spelling(text)
            all_corrections.extend(spelling_corrections)
        
        # 同音词纠错
        if self.enable_types[CorrectionType.HOMOPHONE]:
            homophone_corrections = self.homophone_corrector.correct_homophones(text)
            all_corrections.extend(homophone_corrections)
        
        # 上下文纠错
        if self.enable_types[CorrectionType.CONTEXT]:
            context_corrections = self.context_corrector.correct_context(text)
            all_corrections.extend(context_corrections)
        
        # 应用纠错
        corrected_text = self._apply_corrections(text, all_corrections)
        
        # 计算整体置信度
        confidence = self._calculate_overall_confidence(all_corrections)
        
        # 限制纠错数量
        if len(all_corrections) > 10:
            all_corrections = sorted(all_corrections, key=lambda x: x.confidence, reverse=True)[:10]
            corrected_text = self._apply_corrections(text, all_corrections)
        
        return CorrectionResult(
            original_text=original_text,
            corrected_text=corrected_text,
            confidence=confidence,
            corrections=[{
                "original": c.original_word,
                "corrected": c.corrected_word,
                "position": c.position,
                "confidence": c.confidence,
                "type": c.correction_type.value,
                "suggestions": c.suggestions
            } for c in all_corrections],
            correction_type=CorrectionType.CONTEXT,  # 整体纠错类型
            timestamp=start_time
        )
    
    def _apply_corrections(self, text: str, corrections: List[WordCorrection]) -> str:
        """应用纠错到文本"""
        if not corrections:
            return text
        
        # 按位置排序，从后往前替换
        corrections.sort(key=lambda x: x.position[0], reverse=True)
        
        result_text = text
        for correction in corrections:
            start, end = correction.position
            result_text = result_text[:start] + correction.corrected_word + result_text[end:]
        
        return result_text
    
    def _calculate_overall_confidence(self, corrections: List[WordCorrection]) -> float:
        """计算整体置信度"""
        if not corrections:
            return 1.0
        
        # 基于纠错数量和置信度计算
        correction_count = len(corrections)
        avg_confidence = np.mean([c.confidence for c in corrections])
        
        # 纠错越多，置信度越低
        penalty = min(correction_count * 0.1, 0.5)
        
        return max(0.0, avg_confidence - penalty)
    
    def get_correction_stats(self, text: str) -> Dict[str, Any]:
        """获取纠错统计信息"""
        corrections = []
        
        if self.enable_types[CorrectionType.SPELLING]:
            corrections.extend(self.spelling_corrector.correct_spelling(text))
        
        if self.enable_types[CorrectionType.HOMOPHONE]:
            corrections.extend(self.homophone_corrector.correct_homophones(text))
        
        if self.enable_types[CorrectionType.CONTEXT]:
            corrections.extend(self.context_corrector.correct_context(text))
        
        stats = {
            "total_corrections": len(corrections),
            "correction_types": Counter(c.correction_type for c in corrections),
            "avg_confidence": np.mean([c.confidence for c in corrections]) if corrections else 1.0,
            "high_confidence_corrections": len([c for c in corrections if c.confidence > 0.8])
        }
        
        return stats
    
    def batch_correct(self, texts: List[str]) -> List[CorrectionResult]:
        """批量纠错"""
        results = []
        
        for text in texts:
            try:
                result = asyncio.run(self.correct(text))
                results.append(result)
            except Exception as e:
                logger.error(f"文本纠错失败: {text}, 错误: {e}")
                results.append(CorrectionResult(
                    original_text=text,
                    corrected_text=text,
                    confidence=0.0,
                    corrections=[],
                    correction_type=CorrectionType.SPELLING,
                    timestamp=asyncio.get_event_loop().time()
                ))
        
        return results