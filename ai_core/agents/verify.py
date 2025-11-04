#!/usr/bin/env python3
"""
LLM多模态核心系统验证脚本
验证所有模块是否能正确导入和初始化
"""

import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, '/workspace/backend/src')

def test_imports():
    """测试所有模块导入"""
    print("=" * 60)
    print("LLM多模态核心系统 - 模块导入验证")
    print("=" * 60)
    
    tests = [
        # 核心模块
        ("llm", "基础模块"),
        ("llm.config", "配置管理"),
        ("llm.utils", "工具函数"),
        
        # 核心子模块
        ("llm.core", "核心模块"),
        ("llm.core.orchestrator", "LLM编排器"),
        ("llm.core.router", "模型路由"),
        ("llm.core.cache", "响应缓存"),
        ("llm.core.load_balancer", "负载均衡"),
        
        # 多模态子模块
        ("llm.multimodal", "多模态处理"),
        ("llm.multimodal.text_processor", "文本处理"),
        ("llm.multimodal.image_processor", "图像处理"),
        ("llm.multimodal.audio_processor", "音频处理"),
        ("llm.multimodal.video_processor", "视频处理"),
        
        # 流式子模块
        ("llm.streaming", "流式生成"),
        ("llm.streaming.stream_manager", "流管理"),
        
        # 接口子模块
        ("llm.interfaces", "统一接口"),
        ("llm.interfaces.llm_interface", "LLM接口"),
        ("llm.interfaces.manager", "管理系统"),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, description in tests:
        try:
            module = __import__(module_name, fromlist=[''])
            print(f"✅ {description:<20} - {module_name}")
            passed += 1
        except Exception as e:
            print(f"❌ {description:<20} - {module_name}")
            print(f"   错误: {str(e)[:100]}")
            failed += 1
    
    print("=" * 60)
    print(f"测试结果: {passed} 成功, {failed} 失败")
    print("=" * 60)
    
    return failed == 0

def test_basic_functionality():
    """测试基本功能"""
    print("\n基本功能测试:")
    print("-" * 60)
    
    try:
        # 测试配置加载
        from llm.config import config, ModelType, Provider
        print("✅ 配置系统加载成功")
        
        # 测试工具函数
        from llm.utils import generate_cache_key, format_file_size
        cache_key = generate_cache_key({"test": "data"})
        print(f"✅ 工具函数工作正常 (缓存键: {cache_key[:16]}...)")
        
        # 测试处理器创建
        from llm.multimodal.text_processor import TextProcessor
        processor = TextProcessor()
        print("✅ 文本处理器创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        return False

def test_module_structure():
    """测试模块结构"""
    print("\n模块结构验证:")
    print("-" * 60)
    
    expected_files = [
        '/workspace/backend/src/llm/__init__.py',
        '/workspace/backend/src/llm/config.py',
        '/workspace/backend/src/llm/utils.py',
        '/workspace/backend/src/llm/core/orchestrator.py',
        '/workspace/backend/src/llm/core/router.py',
        '/workspace/backend/src/llm/core/cache.py',
        '/workspace/backend/src/llm/core/load_balancer.py',
        '/workspace/backend/src/llm/multimodal/text_processor.py',
        '/workspace/backend/src/llm/multimodal/image_processor.py',
        '/workspace/backend/src/llm/multimodal/audio_processor.py',
        '/workspace/backend/src/llm/multimodal/video_processor.py',
        '/workspace/backend/src/llm/streaming/stream_manager.py',
        '/workspace/backend/src/llm/interfaces/llm_interface.py',
        '/workspace/backend/src/llm/interfaces/manager.py',
        '/workspace/backend/src/llm/examples.py',
        '/workspace/backend/src/llm/README.md',
        '/workspace/backend/src/llm/IMPLEMENTATION_REPORT.md',
    ]
    
    missing_files = []
    for file_path in expected_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺失文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print(f"✅ 所有 {len(expected_files)} 个文件都存在")
        return True

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("LLM多模态核心系统验证")
    print("=" * 60 + "\n")
    
    # 测试模块结构
    structure_ok = test_module_structure()
    
    # 测试导入
    imports_ok = test_imports()
    
    # 测试基本功能
    functionality_ok = test_basic_functionality()
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结:")
    print("=" * 60)
    print(f"模块结构: {'✅ 通过' if structure_ok else '❌ 失败'}")
    print(f"模块导入: {'✅ 通过' if imports_ok else '❌ 失败'}")
    print(f"基本功能: {'✅ 通过' if functionality_ok else '❌ 失败'}")
    
    if structure_ok and imports_ok and functionality_ok:
        print("\n🎉 所有验证通过！LLM多模态核心系统实现成功！")
        return 0
    else:
        print("\n⚠️  部分验证失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)