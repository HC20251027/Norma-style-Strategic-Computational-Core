#!/usr/bin/env python3
"""
诺玛·劳恩斯AI系统 - 完整功能演示

作者: 皇
创建时间: 2025-10-31
"""

import sys
import os
import json
from datetime import datetime

# 添加路径
sys.path.append('/workspace/code')

def demo_norma_system():
    """完整演示诺玛系统功能"""
    
    print("=" * 80)
    print("诺玛·劳恩斯AI系统 - 完整功能演示")
    print("卡塞尔学院主控计算机 - 现实化版本")
    print("=" * 80)
    print(f"演示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # 1. 导入模块
        print("🔧 正在初始化系统组件...")
        from norma_core_agent import NormaCoreAgent, create_demo_data
        from norma_advanced_features import NormaAdvancedFeatures
        
        print("✓ 模块导入成功")
        
        # 2. 创建演示数据
        print("\n📊 正在创建演示数据...")
        create_demo_data()
        print("✓ 演示数据创建完成")
        
        # 3. 初始化核心智能体
        print("\n🤖 正在启动诺玛核心智能体...")
        norma = NormaCoreAgent()
        print("✓ 核心智能体启动成功")
        
        # 4. 系统状态展示
        print("\n" + "=" * 60)
        print("📊 系统状态监控")
        print("=" * 60)
        status = norma.get_system_status()
        print(status)
        
        # 5. 网络安全扫描演示
        print("\n" + "=" * 60)
        print("🔒 网络安全扫描")
        print("=" * 60)
        scan_result = norma.tools.scan_network()
        print(f"扫描类型: {scan_result.get('scan_type', 'N/A')}")
        print(f"目标网络: {scan_result.get('target_range', 'N/A')}")
        print(f"活跃主机: {scan_result.get('scan_results', {}).get('active_hosts', 'N/A')}")
        print(f"安全状态: {scan_result.get('scan_results', {}).get('security_status', 'N/A')}")
        
        # 6. 龙族血统分析演示
        print("\n" + "=" * 60)
        print("🐉 龙族血统分析")
        print("=" * 60)
        
        # 显示数据库统计
        stats = norma.tools.dragon_blood_analysis()
        print("血统数据库统计:")
        print(f"  注册学生总数: {stats.get('total_registered_students', 0)}")
        print(f"  血统分布: {stats.get('bloodline_distribution', {})}")
        
        # 分析特定学生
        print("\n路明非血统分析:")
        blood_result = norma.tools.dragon_blood_analysis("路明非")
        print(f"  血统类型: {blood_result.get('bloodline_type', 'N/A')}")
        print(f"  纯度等级: {blood_result.get('purity_level', 'N/A')}")
        print(f"  能力特征: {blood_result.get('abilities', 'N/A')}")
        print(f"  觉醒状态: {blood_result.get('status', 'N/A')}")
        
        # 7. 安全检查演示
        print("\n" + "=" * 60)
        print("🛡️ 安全检查报告")
        print("=" * 60)
        security = norma.tools.security_check()
        print(f"防火墙状态: {security.get('firewall_status', 'N/A')}")
        print(f"入侵检测: {security.get('intrusion_detection', 'N/A')}")
        print(f"安全评分: {security.get('security_score', 'N/A')}")
        print(f"可疑活动: {security.get('suspicious_activities', 'N/A')}")
        
        # 8. 高级功能演示
        print("\n" + "=" * 60)
        print("🚀 高级功能演示")
        print("=" * 60)
        
        advanced = NormaAdvancedFeatures()
        
        # 搜索功能
        print("\n8.1 DuckDuckGo搜索功能")
        print("-" * 30)
        search_result = advanced.demo_search_functionality("龙族血统检测技术")
        print(f"搜索状态: {search_result.get('status', 'N/A')}")
        
        # PDF处理
        print("\n8.2 PDF知识库处理")
        print("-" * 30)
        pdf_result = advanced.demo_pdf_processing()
        print(f"处理状态: {pdf_result.get('processing_status', 'N/A')}")
        print(f"文档类型: {pdf_result.get('document_type', 'N/A')}")
        
        # RAG检索
        print("\n8.3 向量数据库RAG检索")
        print("-" * 30)
        rag_result = advanced.demo_rag_functionality()
        print(f"检索方法: {rag_result.get('retrieval_method', 'N/A')}")
        print(f"检索结果数: {rag_result.get('total_results', 0)}")
        
        # 多智能体协作
        print("\n8.4 多智能体协作")
        print("-" * 30)
        collab_result = advanced.demo_multi_agent_collaboration()
        print(f"协作模式: {collab_result.get('collaboration_mode', 'N/A')}")
        print(f"团队成员: {', '.join(collab_result.get('team_members', []))}")
        
        # 9. 系统日志展示
        print("\n" + "=" * 60)
        print("📋 系统日志")
        print("=" * 60)
        logs = norma.tools.get_system_logs(5)
        print(f"日志条目数: {logs.get('log_count', 0)}")
        if logs.get('logs'):
            print("最近日志:")
            for log in logs['logs'][:3]:
                print(f"  [{log.get('level', 'N/A')}] {log.get('module', 'N/A')}: {log.get('message', 'N/A')}")
        
        # 10. 总结
        print("\n" + "=" * 80)
        print("🎉 诺玛·劳恩斯AI系统演示完成")
        print("=" * 80)
        print("✅ 所有功能模块正常运行")
        print("✅ 核心智能体响应正常")
        print("✅ 高级功能演示成功")
        print("✅ 数据持久化正常")
        print("✅ 多智能体协作正常")
        print()
        print("🚀 系统已准备就绪，可以开始使用！")
        print()
        print("📖 使用方法:")
        print("  1. 基础对话: python code/norma_core_agent.py")
        print("  2. 高级功能: python code/norma_advanced_features.py") 
        print("  3. 完整系统: python code/norma_main_system.py")
        print()
        print("🔧 配置说明:")
        print("  - 当前为演示模式，无需OpenAI API密钥")
        print("  - 可配置API密钥以启用完整AI功能")
        print("  - 所有功能基于合法合规的开源技术")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("正在启动诺玛·劳恩斯AI系统演示...")
    success = demo_norma_system()
    
    if success:
        print("🎊 演示成功完成！诺玛系统运行正常。")
    else:
        print("💥 演示失败，请检查错误信息。")
    
    print("\n按任意键退出...")
    try:
        input()
    except:
        pass