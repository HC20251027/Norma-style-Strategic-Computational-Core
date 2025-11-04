#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诺玛AI系统环境变量配置验证脚本
用于检查部署环境变量是否正确配置
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple, Dict

class EnvironmentChecker:
    def __init__(self):
        self.required_vars = {
            'DEEPSEEK_API_KEY': {
                'description': 'DeepSeek API密钥',
                'required': True,
                'validation': self._validate_api_key
            },
            'DATABASE_URL': {
                'description': '数据库连接URL',
                'required': True,
                'validation': self._validate_database_url
            },
            'SECRET_KEY': {
                'description': '应用密钥',
                'required': True,
                'validation': self._validate_secret_key
            },
            'DEBUG_MODE': {
                'description': '调试模式',
                'required': True,
                'validation': self._validate_debug_mode
            },
            'CORS_ORIGINS': {
                'description': '跨域设置',
                'required': True,
                'validation': self._validate_cors_origins
            }
        }
        
        self.optional_vars = {
            'PORT': {
                'description': '服务器端口',
                'default': '8000',
                'validation': self._validate_port
            },
            'LOG_LEVEL': {
                'description': '日志级别',
                'default': 'INFO',
                'validation': self._validate_log_level
            },
            'MAX_FILE_SIZE': {
                'description': '最大文件大小(MB)',
                'default': '100',
                'validation': self._validate_file_size
            },
            'SESSION_TIMEOUT': {
                'description': '会话超时时间(分钟)',
                'default': '30',
                'validation': self._validate_session_timeout
            },
            'REDIS_URL': {
                'description': 'Redis连接URL',
                'required': False,
                'validation': self._validate_redis_url
            }
        }

    def _validate_api_key(self, value: str) -> Tuple[bool, str]:
        """验证API密钥"""
        if not value or value == 'your_deepseek_api_key_here':
            return False, "API密钥未配置或使用默认值"
        
        if not value.startswith('sk-'):
            return False, "API密钥格式不正确，应以'sk-'开头"
        
        if len(value) < 40:
            return False, "API密钥长度不足"
        
        return True, "API密钥格式正确"

    def _validate_database_url(self, value: str) -> Tuple[bool, str]:
        """验证数据库URL"""
        if not value:
            return False, "数据库URL未配置"
        
        # 支持的数据库类型
        supported_schemes = ['sqlite', 'postgresql', 'mysql']
        scheme = value.split('://')[0] if '://' in value else ''
        
        if scheme not in supported_schemes:
            return False, f"不支持的数据库类型: {scheme}，支持的类型: {', '.join(supported_schemes)}"
        
        if scheme == 'sqlite':
            # SQLite路径验证
            path = value.replace('sqlite:///', '')
            if path.startswith('./'):
                path = path[2:]
            
            # 检查目录是否存在
            dir_path = os.path.dirname(path)
            if dir_path and not os.path.exists(dir_path):
                return False, f"SQLite数据库目录不存在: {dir_path}"
        
        return True, f"数据库URL配置正确 ({scheme})"

    def _validate_secret_key(self, value: str) -> Tuple[bool, str]:
        """验证应用密钥"""
        if not value or value == 'your_secret_key_here_generate_with_secrets_token_urlsafe':
            return False, "应用密钥未配置或使用默认值"
        
        if len(value) < 32:
            return False, "应用密钥长度不足，至少需要32字符"
        
        # 检查是否包含特殊字符
        if not re.search(r'[A-Za-z]', value) or not re.search(r'[0-9]', value):
            return False, "应用密钥应包含字母和数字"
        
        return True, "应用密钥强度符合要求"

    def _validate_debug_mode(self, value: str) -> Tuple[bool, str]:
        """验证调试模式"""
        if not value:
            return False, "调试模式未配置"
        
        value_lower = value.lower()
        if value_lower in ['true', 'false']:
            return True, f"调试模式设置正确 ({value_lower})"
        else:
            return False, "调试模式值应为 'true' 或 'false'"

    def _validate_cors_origins(self, value: str) -> Tuple[bool, str]:
        """验证跨域设置"""
        if not value:
            return False, "跨域设置未配置"
        
        origins = [origin.strip() for origin in value.split(',')]
        
        if '*' in origins:
            return False, "生产环境不建议使用通配符 '*'"
        
        # 验证每个域名格式
        invalid_origins = []
        for origin in origins:
            if not (origin.startswith('http://') or origin.startswith('https://')):
                invalid_origins.append(origin)
        
        if invalid_origins:
            return False, f"跨域域名格式错误: {', '.join(invalid_origins)}"
        
        return True, f"跨域设置正确 ({len(origins)}个域名)"

    def _validate_port(self, value: str) -> Tuple[bool, str]:
        """验证端口号"""
        try:
            port = int(value)
            if 1 <= port <= 65535:
                return True, f"端口号正确 ({port})"
            else:
                return False, "端口号应在1-65535范围内"
        except ValueError:
            return False, "端口号应为数字"

    def _validate_log_level(self, value: str) -> Tuple[bool, str]:
        """验证日志级别"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if value.upper() in valid_levels:
            return True, f"日志级别正确 ({value.upper()})"
        else:
            return False, f"日志级别应为: {', '.join(valid_levels)}"

    def _validate_file_size(self, value: str) -> Tuple[bool, str]:
        """验证文件大小"""
        try:
            size = int(value)
            if size > 0:
                return True, f"最大文件大小正确 ({size}MB)"
            else:
                return False, "文件大小必须大于0"
        except ValueError:
            return False, "文件大小应为数字"

    def _validate_session_timeout(self, value: str) -> Tuple[bool, str]:
        """验证会话超时"""
        try:
            timeout = int(value)
            if timeout > 0:
                return True, f"会话超时时间正确 ({timeout}分钟)"
            else:
                return False, "会话超时时间必须大于0"
        except ValueError:
            return False, "会话超时时间应为数字"

    def _validate_redis_url(self, value: str) -> Tuple[bool, str]:
        """验证Redis URL"""
        if not value:
            return True, "Redis未配置（可选）"
        
        if not value.startswith('redis://'):
            return False, "Redis URL格式不正确，应以'redis://'开头"
        
        return True, "Redis URL格式正确"

    def check_environment(self) -> Dict:
        """检查环境变量配置"""
        results = {
            'total_checks': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'details': []
        }
        
        print("🔍 开始检查环境变量配置...\n")
        
        # 检查必要变量
        for var_name, config in self.required_vars.items():
            results['total_checks'] += 1
            value = os.getenv(var_name)
            
            is_valid, message = config['validation'](value)
            
            detail = {
                'name': var_name,
                'description': config['description'],
                'value': value,
                'status': 'passed' if is_valid else 'failed',
                'message': message
            }
            
            results['details'].append(detail)
            
            if is_valid:
                results['passed'] += 1
                print(f"✅ {var_name}: {message}")
            else:
                results['failed'] += 1
                print(f"❌ {var_name}: {message}")
        
        print("\n" + "="*50 + "\n")
        
        # 检查可选变量
        print("📋 可选变量检查:\n")
        
        for var_name, config in self.optional_vars.items():
            results['total_checks'] += 1
            value = os.getenv(var_name) or config.get('default', '')
            
            if not value:
                print(f"⚠️  {var_name}: 未配置（使用默认值: {config['default']}）")
                results['warnings'] += 1
                continue
            
            is_valid, message = config['validation'](value)
            
            detail = {
                'name': var_name,
                'description': config['description'],
                'value': value,
                'status': 'warning' if not is_valid else 'passed',
                'message': message
            }
            
            results['details'].append(detail)
            
            if is_valid:
                results['passed'] += 1
                print(f"✅ {var_name}: {message}")
            else:
                results['warnings'] += 1
                print(f"⚠️  {var_name}: {message}")
        
        return results

    def print_summary(self, results: Dict):
        """打印检查摘要"""
        print("\n" + "="*50)
        print("📊 配置检查摘要")
        print("="*50)
        print(f"总检查项: {results['total_checks']}")
        print(f"✅ 通过: {results['passed']}")
        print(f"❌ 失败: {results['failed']}")
        print(f"⚠️  警告: {results['warnings']}")
        
        if results['failed'] == 0:
            print("\n🎉 所有必要配置项都已正确配置！")
            if results['warnings'] > 0:
                print("⚠️  建议检查警告项目以优化配置")
        else:
            print(f"\n❌ 发现 {results['failed']} 个配置问题，请修复后重新部署")
        
        print("\n" + "="*50)

    def generate_report(self, results: Dict, output_file: str = "env_check_report.txt"):
        """生成检查报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("诺玛AI系统环境变量配置检查报告\n")
            f.write("="*50 + "\n")
            f.write(f"检查时间: {os.popen('date').read().strip()}\n\n")
            
            f.write("检查摘要:\n")
            f.write(f"- 总检查项: {results['total_checks']}\n")
            f.write(f"- 通过: {results['passed']}\n")
            f.write(f"- 失败: {results['failed']}\n")
            f.write(f"- 警告: {results['warnings']}\n\n")
            
            f.write("详细结果:\n")
            f.write("-"*30 + "\n")
            
            for detail in results['details']:
                status_icon = "✅" if detail['status'] == 'passed' else "❌" if detail['status'] == 'failed' else "⚠️"
                f.write(f"{status_icon} {detail['name']}\n")
                f.write(f"   描述: {detail['description']}\n")
                f.write(f"   值: {detail['value'] if detail['value'] else '未设置'}\n")
                f.write(f"   结果: {detail['message']}\n\n")
        
        print(f"\n📄 检查报告已保存到: {output_file}")

def main():
    """主函数"""
    print("🚀 诺玛AI系统环境变量配置检查工具")
    print("="*50 + "\n")
    
    # 检查.env文件是否存在
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ 错误: .env文件不存在")
        print("请复制 .env.example 为 .env 并配置相应值")
        print("\n示例:")
        print("cp .env.example .env")
        sys.exit(1)
    
    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv()
    
    # 执行检查
    checker = EnvironmentChecker()
    results = checker.check_environment()
    checker.print_summary(results)
    
    # 生成报告
    checker.generate_report(results)
    
    # 返回适当的退出码
    if results['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("❌ 错误: 缺少python-dotenv依赖")
        print("请安装: pip install python-dotenv")
        sys.exit(1)
    
    main()