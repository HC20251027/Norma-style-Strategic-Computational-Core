"""
参数推断和验证系统
智能推断工具调用参数并进行验证
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import ast
import yaml
from urllib.parse import urlparse
import ipaddress

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """参数类型枚举"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    FILE_PATH = "file_path"
    URL = "url"
    EMAIL = "email"
    PHONE = "phone"
    DATE = "date"
    JSON = "json"
    CODE = "code"
    REGEX = "regex"


@dataclass
class ParameterSpec:
    """参数规范"""
    name: str
    type: ParameterType
    required: bool = False
    default: Any = None
    description: str = ""
    validation_rules: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    examples: List[Any] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)


@dataclass
class InferredParameter:
    """推断的参数"""
    name: str
    value: Any
    confidence: float
    source: str
    validation_status: str = "pending"
    inferred_type: Optional[ParameterType] = None


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class ParameterInferenceEngine:
    """参数推断引擎"""
    
    def __init__(self):
        self.type_patterns = {
            ParameterType.EMAIL: re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            ParameterType.PHONE: re.compile(r'^[\+]?[1-9][\d]{0,15}$'),
            ParameterType.URL: re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            ParameterType.DATE: re.compile(r'^\d{4}-\d{2}-\d{2}$|^\d{2}/\d{2}/\d{4}$'),
            ParameterType.FILE_PATH: re.compile(r'^[\w\-./\\]+$'),
            ParameterType.REGEX: re.compile(r'^/.*/[a-z]*$|^.*\*.*$')
        }
        
        self.inference_rules = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """设置默认推断规则"""
        # 数字推断规则
        self.inference_rules.append({
            'pattern': r'\b\d+\b',
            'type': ParameterType.INTEGER,
            'confidence': 0.8
        })
        
        # 浮点数推断规则
        self.inference_rules.append({
            'pattern': r'\b\d*\.\d+\b',
            'type': ParameterType.FLOAT,
            'confidence': 0.9
        })
        
        # 布尔值推断规则
        self.inference_rules.append({
            'pattern': r'\b(true|false|True|False|是|否|对|错)\b',
            'type': ParameterType.BOOLEAN,
            'confidence': 0.9
        })
        
        # JSON推断规则
        self.inference_rules.append({
            'pattern': r'^\s*[\{\[][\s\S]*[\}\]]\s*$',
            'type': ParameterType.JSON,
            'confidence': 0.95
        })
    
    def infer_parameter_type(self, value: str) -> Tuple[ParameterType, float]:
        """推断参数类型"""
        if not isinstance(value, str):
            return ParameterType.STRING, 0.5
        
        value = value.strip()
        
        # 检查预定义模式
        for param_type, pattern in self.type_patterns.items():
            if pattern.match(value):
                return param_type, 0.9
        
        # 检查推断规则
        for rule in self.inference_rules:
            if re.search(rule['pattern'], value, re.IGNORECASE):
                return rule['type'], rule['confidence']
        
        # 默认为字符串
        return ParameterType.STRING, 0.5
    
    def extract_parameters_from_text(self, text: str, parameter_specs: List[ParameterSpec]) -> List[InferredParameter]:
        """从文本中提取参数"""
        inferred_params = []
        
        for spec in parameter_specs:
            param_value = self._extract_parameter_by_spec(text, spec)
            if param_value is not None:
                inferred_type, confidence = self.infer_parameter_type(str(param_value))
                
                inferred_param = InferredParameter(
                    name=spec.name,
                    value=param_value,
                    confidence=confidence,
                    source="text_extraction",
                    inferred_type=inferred_type
                )
                inferred_params.append(inferred_param)
        
        return inferred_params
    
    def _extract_parameter_by_spec(self, text: str, spec: ParameterSpec) -> Optional[Any]:
        """根据规范提取参数"""
        # 尝试直接匹配参数名
        patterns = [spec.name] + spec.aliases
        
        for pattern in patterns:
            # 尝试多种提取模式
            extracted_value = self._extract_by_pattern(text, pattern)
            if extracted_value is not None:
                return self._convert_value(extracted_value, spec.type)
        
        # 尝试基于类型的推断
        return self._infer_by_type(text, spec)
    
    def _extract_by_pattern(self, text: str, pattern: str) -> Optional[str]:
        """根据模式提取值"""
        # 模式1: "参数名: 值"
        match = re.search(f'{pattern}\\s*[:：]\\s*([^\\s,，]+)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip('"\'')
        
        # 模式2: "参数名 = 值"
        match = re.search(f'{pattern}\\s*=\\s*([^\\s,，]+)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip('"\'')
        
        # 模式3: "参数名 值"
        match = re.search(f'{pattern}\\s+([^\\s,，]+)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip('"\'')
        
        return None
    
    def _infer_by_type(self, text: str, spec: ParameterSpec) -> Optional[Any]:
        """基于类型推断参数"""
        if spec.type == ParameterType.LIST:
            # 查找逗号分隔的值
            match = re.search(r'([^,，]+(?:[,，][^,，]+)*)', text)
            if match:
                values = [v.strip() for v in match.group(1).split(',，')]
                return values
        
        elif spec.type == ParameterType.JSON:
            # 查找JSON格式的文本
            match = re.search(r'(\\{[^}]*\\}|\\[[^\\]]*\\])', text)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
        
        elif spec.type == ParameterType.CODE:
            # 查找代码块
            match = re.search(r'```(?:\\w+)?\\n([\\s\\S]*?)```', text)
            if match:
                return match.group(1).strip()
            
            # 查找单行代码
            match = re.search(r'`([^`]+)`', text)
            if match:
                return match.group(1)
        
        return None
    
    def _convert_value(self, value: str, target_type: ParameterType) -> Any:
        """转换值到目标类型"""
        if not isinstance(value, str):
            return value
        
        try:
            if target_type == ParameterType.INTEGER:
                return int(value)
            elif target_type == ParameterType.FLOAT:
                return float(value)
            elif target_type == ParameterType.BOOLEAN:
                return value.lower() in ('true', '1', 'yes', '是', '对')
            elif target_type == ParameterType.LIST:
                return [v.strip() for v in value.split(',，')]
            elif target_type == ParameterType.DICT:
                return json.loads(value)
            elif target_type == ParameterType.JSON:
                return json.loads(value)
            else:
                return value
        except (ValueError, json.JSONDecodeError):
            return value  # 返回原始字符串
    
    def validate_parameters(self, parameters: Dict[str, Any], specs: List[ParameterSpec]) -> Dict[str, ValidationResult]:
        """验证参数"""
        results = {}
        
        # 创建参数规范映射
        spec_map = {spec.name: spec for spec in specs}
        
        for param_name, param_value in parameters.items():
            if param_name in spec_map:
                result = self._validate_single_parameter(param_value, spec_map[param_name])
                results[param_name] = result
            else:
                # 未定义的参数，生成警告
                results[param_name] = ValidationResult(
                    is_valid=True,
                    warnings=[f"参数 '{param_name}' 未在规范中定义"]
                )
        
        # 检查必需参数
        for spec in specs:
            if spec.required and spec.name not in parameters:
                if spec.name not in results:
                    results[spec.name] = ValidationResult(is_valid=False)
                results[spec.name].errors.append(f"必需参数 '{spec.name}' 缺失")
        
        return results
    
    def _validate_single_parameter(self, value: Any, spec: ParameterSpec) -> ValidationResult:
        """验证单个参数"""
        errors = []
        warnings = []
        suggestions = []
        
        # 类型验证
        if not self._validate_type(value, spec.type):
            errors.append(f"参数类型错误，期望 {spec.type.value}，得到 {type(value).__name__}")
        
        # 约束验证
        if spec.constraints:
            constraint_errors = self._validate_constraints(value, spec.constraints)
            errors.extend(constraint_errors)
        
        # 验证规则验证
        for rule in spec.validation_rules:
            rule_result = self._apply_validation_rule(value, rule)
            if not rule_result['valid']:
                errors.extend(rule_result['errors'])
            if rule_result['warnings']:
                warnings.extend(rule_result['warnings'])
        
        # 生成建议
        if errors:
            suggestions = self._generate_suggestions(value, spec)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_type(self, value: Any, expected_type: ParameterType) -> bool:
        """验证参数类型"""
        if expected_type == ParameterType.STRING:
            return isinstance(value, str)
        elif expected_type == ParameterType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == ParameterType.FLOAT:
            return isinstance(value, float)
        elif expected_type == ParameterType.BOOLEAN:
            return isinstance(value, bool)
        elif expected_type == ParameterType.LIST:
            return isinstance(value, list)
        elif expected_type == ParameterType.DICT:
            return isinstance(value, dict)
        elif expected_type == ParameterType.FILE_PATH:
            return isinstance(value, str) and self._is_valid_file_path(value)
        elif expected_type == ParameterType.URL:
            return isinstance(value, str) and self._is_valid_url(value)
        elif expected_type == ParameterType.EMAIL:
            return isinstance(value, str) and self._is_valid_email(value)
        elif expected_type == ParameterType.DATE:
            return isinstance(value, str) and self._is_valid_date(value)
        elif expected_type == ParameterType.JSON:
            try:
                json.loads(value) if isinstance(value, str) else json.dumps(value)
                return True
            except:
                return False
        
        return True
    
    def _is_valid_file_path(self, path: str) -> bool:
        """验证文件路径"""
        import os
        try:
            # 检查路径格式
            if not path or '/' in path or '\\' in path:
                return True  # 允许相对路径
            return True
        except:
            return False
    
    def _is_valid_url(self, url: str) -> bool:
        """验证URL"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _is_valid_email(self, email: str) -> bool:
        """验证邮箱"""
        return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))
    
    def _is_valid_date(self, date_str: str) -> bool:
        """验证日期格式"""
        date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']
        for fmt in date_formats:
            try:
                datetime.strptime(date_str, fmt)
                return True
            except ValueError:
                continue
        return False
    
    def _validate_constraints(self, value: Any, constraints: Dict[str, Any]) -> List[str]:
        """验证约束条件"""
        errors = []
        
        for constraint_name, constraint_value in constraints.items():
            if constraint_name == 'min_length' and isinstance(value, (str, list)):
                if len(value) < constraint_value:
                    errors.append(f"长度不能小于 {constraint_value}")
            
            elif constraint_name == 'max_length' and isinstance(value, (str, list)):
                if len(value) > constraint_value:
                    errors.append(f"长度不能大于 {constraint_value}")
            
            elif constraint_name == 'min_value' and isinstance(value, (int, float)):
                if value < constraint_value:
                    errors.append(f"值不能小于 {constraint_value}")
            
            elif constraint_name == 'max_value' and isinstance(value, (int, float)):
                if value > constraint_value:
                    errors.append(f"值不能大于 {constraint_value}")
            
            elif constraint_name == 'pattern' and isinstance(value, str):
                if not re.match(constraint_value, value):
                    errors.append(f"不匹配模式: {constraint_value}")
            
            elif constraint_name == 'choices' and isinstance(value, str):
                if value not in constraint_value:
                    errors.append(f"必须是以下值之一: {constraint_value}")
        
        return errors
    
    def _apply_validation_rule(self, value: Any, rule: str) -> Dict[str, Any]:
        """应用验证规则"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        # 简单的验证规则实现
        if rule.startswith('regex:'):
            pattern = rule[6:]
            if isinstance(value, str) and not re.match(pattern, value):
                result['valid'] = False
                result['errors'].append(f"不匹配正则表达式: {pattern}")
        
        elif rule.startswith('function:'):
            func_name = rule[9:]
            # 这里可以调用自定义验证函数
            # result = self._call_validation_function(func_name, value)
            pass
        
        return result
    
    def _generate_suggestions(self, value: Any, spec: ParameterSpec) -> List[str]:
        """生成修正建议"""
        suggestions = []
        
        # 基于类型生成建议
        if spec.type == ParameterType.EMAIL and isinstance(value, str):
            if '@' not in value:
                suggestions.append("邮箱地址需要包含 '@' 符号")
        
        elif spec.type == ParameterType.URL and isinstance(value, str):
            if not value.startswith(('http://', 'https://')):
                suggestions.append("URL需要以 'http://' 或 'https://' 开头")
        
        elif spec.type == ParameterType.FILE_PATH and isinstance(value, str):
            if ' ' in value:
                suggestions.append("文件路径不应包含空格")
        
        # 基于示例生成建议
        if spec.examples:
            suggestions.append(f"示例值: {spec.examples[0] if spec.examples else 'N/A'}")
        
        return suggestions
    
    def auto_complete_parameters(self, partial_params: Dict[str, Any], specs: List[ParameterSpec]) -> Dict[str, Any]:
        """自动补全参数"""
        completed_params = partial_params.copy()
        
        for spec in specs:
            if spec.name not in completed_params and spec.default is not None:
                completed_params[spec.name] = spec.default
        
        return completed_params
    
    def suggest_missing_parameters(self, context: str, specs: List[ParameterSpec]) -> List[str]:
        """建议缺失的参数"""
        suggestions = []
        
        for spec in specs:
            if spec.required:
                # 基于上下文推断是否需要此参数
                if self._is_parameter_likely_needed(context, spec):
                    suggestions.append(f"可能需要参数: {spec.name} ({spec.description})")
        
        return suggestions
    
    def _is_parameter_likely_needed(self, context: str, spec: ParameterSpec) -> bool:
        """判断参数是否可能需要"""
        # 简单的关键词匹配
        context_lower = context.lower()
        
        for keyword in spec.keywords if hasattr(spec, 'keywords') else []:
            if keyword.lower() in context_lower:
                return True
        
        return False