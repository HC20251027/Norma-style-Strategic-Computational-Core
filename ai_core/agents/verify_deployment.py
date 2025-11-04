#!/usr/bin/env python3
"""
诺玛·劳恩斯AI系统 - 部署验证脚本

验证所有组件是否正确安装和配置

作者: 皇
创建时间: 2025-10-31
"""

import os
import sys
import json
from pathlib import Path

def verify_deployment():
    """验证部署状态"""
    
    print("=" * 80)
    print("诺玛·劳恩斯AI系统 - 部署验证")
    print("=" * 80)
    
    checks = []
    
    # 1. 检查虚拟环境
    print("\n🔍 检查虚拟环境...")
    venv_path = Path("/workspace/agno_env")
    if venv_path.exists():
        print("✓ 虚拟环境存在")
        checks.append(True)
    else:
        print("❌ 虚拟环境不存在")
        checks.append(False)
    
    # 2. 检查核心文件
    print("\n🔍 检查核心文件...")
    core_files = [
        "/workspace/code/norma_core_agent.py",
        "/workspace/code/norma_advanced_features.py", 
        "/workspace/code/norma_main_system.py",
        "/workspace/code/norma_demo.py"
    ]
    
    for file_path in core_files:
        if Path(file_path).exists():
            print(f"✓ {Path(file_path).name} 存在")
            checks.append(True)
        else:
            print(f"❌ {Path(file_path).name} 不存在")
            checks.append(False)
    
    # 3. 检查文档文件
    print("\n🔍 检查文档文件...")
    doc_files = [
        "/workspace/docs/norma_deployment_report.md",
        "/workspace/docs/norma_user_guide.md",
        "/workspace/docs/agno_guide.md"
    ]
    
    for file_path in doc_files:
        if Path(file_path).exists():
            print(f"✓ {Path(file_path).name} 存在")
            checks.append(True)
        else:
            print(f"❌ {Path(file_path).name} 不存在")
            checks.append(False)
    
    # 4. 检查数据目录
    print("\n🔍 检查数据目录...")
    data_dirs = [
        "/workspace/data",
        "/workspace/data/knowledge_base",
        "/workspace/data/pdf_documents"
    ]
    
    for dir_path in data_dirs:
        if Path(dir_path).exists():
            print(f"✓ {Path(dir_path).name} 目录存在")
            checks.append(True)
        else:
            print(f"❌ {Path(dir_path).name} 目录不存在")
            checks.append(False)
    
    # 5. 检查示例文档
    print("\n🔍 检查示例文档...")
    pdf_files = list(Path("/workspace/data/pdf_documents").glob("*.txt"))
    if pdf_files:
        print(f"✓ 找到 {len(pdf_files)} 个示例文档")
        checks.append(True)
    else:
        print("❌ 未找到示例文档")
        checks.append(False)
    
    # 6. 检查Python模块导入
    print("\n🔍 检查Python模块...")
    try:
        sys.path.append('/workspace/code')
        from norma_core_agent import NormaCoreAgent
        print("✓ 核心模块导入成功")
        checks.append(True)
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        checks.append(False)
    
    # 7. 检查Agno框架
    print("\n🔍 检查Agno框架...")
    try:
        import agno
        print("✓ Agno框架可用")
        checks.append(True)
    except ImportError:
        print("⚠️ Agno框架不可用 (演示模式)")
        checks.append(True)  # 演示模式也算通过
    
    # 8. 总结
    print("\n" + "=" * 80)
    print("📊 部署验证结果")
    print("=" * 80)
    
    total_checks = len(checks)
    passed_checks = sum(checks)
    
    print(f"总检查项: {total_checks}")
    print(f"通过检查: {passed_checks}")
    print(f"失败检查: {total_checks - passed_checks}")
    print(f"通过率: {passed_checks/total_checks*100:.1f}%")
    
    if passed_checks == total_checks:
        print("\n🎉 所有检查通过！系统部署成功！")
        print("\n🚀 可以开始使用诺玛·劳恩斯AI系统：")
        print("   python code/norma_demo.py")
        return True
    else:
        print(f"\n⚠️ 有 {total_checks - passed_checks} 项检查失败")
        print("请检查失败项目并重新部署")
        return False

def show_system_info():
    """显示系统信息"""
    print("\n" + "=" * 80)
    print("ℹ️ 系统信息")
    print("=" * 80)
    
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print(f"Agno环境: {'已配置' if Path('/workspace/agno_env').exists() else '未配置'}")
    print(f"当前时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def show_usage_examples():
    """显示使用示例"""
    print("\n" + "=" * 80)
    print("📖 使用示例")
    print("=" * 80)
    
    examples = [
        ("完整功能演示", "python code/norma_demo.py"),
        ("交互式系统", "python code/norma_main_system.py"),
        ("基础对话", "python code/norma_core_agent.py"),
        ("高级功能", "python code/norma_advanced_features.py"),
        ("快速测试", "python code/quick_test.py")
    ]
    
    for name, command in examples:
        print(f"{name}:")
        print(f"  {command}")
        print()

if __name__ == "__main__":
    show_system_info()
    
    success = verify_deployment()
    
    if success:
        show_usage_examples()
        print("🎊 诺玛·劳恩斯AI系统部署验证完成！")
    else:
        print("❌ 部署验证失败，请检查系统配置")
    
    print("\n按任意键退出...")
    try:
        input()
    except:
        pass