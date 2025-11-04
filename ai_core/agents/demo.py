#!/usr/bin/env python3
"""
诺玛品牌特色系统演示脚本
展示系统各项功能和特性

作者: 皇
创建时间: 2025-10-31
"""

import asyncio
import json
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from core.norma_brand_agent import NormaBrandAgent
from ui.brand_interface import NormaBrandInterface, NormaTheme
from ui.visual_elements import NormaVisualElements
from interactions.welcome_system import NormaWelcomeSystem
from interactions.interaction_manager import NormaInteractionManager
from utils.brand_consistency import BrandConsistencyManager
from config.settings import get_config

async def demo_brand_personality():
    """演示品牌人格系统"""
    print("\n" + "="*50)
    print("演示1: 诺玛品牌人格系统")
    print("="*50)
    
    # 创建诺玛品牌智能体
    norma = NormaBrandAgent("demo_user")
    
    # 演示不同类型的交互
    test_queries = [
        "你好，我是新用户",
        "请分析一下系统性能",
        "遇到了一些技术问题",
        "今天过得怎么样？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n测试 {i}: {query}")
        print("-" * 30)
        
        response_data = await norma.process_brand_interaction(query, "demo")
        print(f"诺玛: {response_data['content']}")
        print(f"情感状态: {response_data['emotion_state']}")
        print(f"响应类型: {response_data['response_type']}")

async def demo_visual_system():
    """演示视觉系统"""
    print("\n" + "="*50)
    print("演示2: 诺玛视觉元素系统")
    print("="*50)
    
    # 创建视觉系统
    visual_system = NormaVisualElements()
    
    # 展示图标系统
    print("\n可用图标:")
    for icon_name, icon_data in list(visual_system.icons.items())[:5]:
        print(f"  - {icon_name}: {icon_data['type'].value}")
    
    # 展示动画系统
    print("\n可用动画:")
    for anim_name in list(visual_system.animations.keys())[:5]:
        print(f"  - {anim_name}")
    
    # 生成CSS动画
    print("\n生成的CSS动画代码片段:")
    css_animations = visual_system.generate_css_animations()
    print(css_animations[:500] + "..." if len(css_animations) > 500 else css_animations)

async def demo_brand_interface():
    """演示品牌界面系统"""
    print("\n" + "="*50)
    print("演示3: 诺玛品牌界面系统")
    print("="*50)
    
    # 创建品牌界面
    brand_interface = NormaBrandInterface(NormaTheme.MODERN)
    
    # 展示色彩方案
    color_scheme = brand_interface.get_color_scheme()
    print(f"\n当前色彩方案:")
    print(f"  主色: {color_scheme.primary}")
    print(f"  辅助色: {color_scheme.secondary}")
    print(f"  强调色: {color_scheme.accent}")
    
    # 展示组件库
    print(f"\n组件库:")
    for component_type in brand_interface.component_library.keys():
        print(f"  - {component_type}")
    
    # 生成HTML模板
    print(f"\n生成主页面模板:")
    html_template = brand_interface.generate_html_template("main_page")
    print(f"模板长度: {len(html_template)} 字符")
    print("模板已生成，可用于Web界面")

async def demo_welcome_system():
    """演示欢迎系统"""
    print("\n" + "="*50)
    print("演示4: 诺玛欢迎系统")
    print("="*50)
    
    # 创建欢迎系统
    welcome_system = NormaWelcomeSystem()
    
    # 演示不同场景的欢迎
    scenarios = [
        ("first_time", "新用户首次访问"),
        ("returning", "老用户回归"),
        ("morning", "早晨问候"),
        ("afternoon", "下午问候")
    ]
    
    for context, description in scenarios:
        print(f"\n{description}:")
        greeting = welcome_system.get_greeting("demo_user", context)
        print(f"  {greeting}")
    
    # 演示交互式欢迎
    print(f"\n交互式欢迎:")
    interactive_welcome = welcome_system.generate_interactive_welcome("demo_user", {})
    print(f"  问候语: {interactive_welcome['greeting']}")
    print(f"  快捷操作: {len(interactive_welcome['interactive_elements']['quick_actions'])} 个")

async def demo_interaction_manager():
    """演示交互管理器"""
    print("\n" + "="*50)
    print("演示5: 诺玛交互管理器")
    print("="*50)
    
    # 创建交互管理器
    interaction_manager = NormaInteractionManager()
    
    # 开始一个交互会话
    from norma_agent_enhanced.interactions.interaction_manager import InteractionType
    session_id = interaction_manager.start_interaction_session(
        "demo_user", 
        interaction_type=InteractionType.CONVERSATION
    )
    
    print(f"会话已启动: {session_id}")
    
    # 模拟交互过程
    test_messages = [
        "你好诺玛",
        "我想了解系统状态",
        "谢谢你的帮助"
    ]
    
    for message in test_messages:
        print(f"\n用户: {message}")
        
        # 模拟响应生成
        async def mock_response_generator(user_input, session):
            return f"这是诺玛对 '{user_input}' 的回应"
        
        # 处理交互
        response_chunks = []
        async for chunk in interaction_manager.process_interaction(
            session_id, message, mock_response_generator
        ):
            response_chunks.append(chunk)
        
        print(f"诺玛: {''.join(response_chunks)}")
    
    # 结束会话
    session_report = interaction_manager.end_interaction_session(session_id)
    print(f"\n会话结束报告:")
    print(f"  持续时间: {session_report.get('duration', 0):.1f} 秒")
    print(f"  消息数量: {session_report.get('message_count', 0)}")
    print(f"  质量分数: {session_report.get('quality_score', 0):.2f}")

async def demo_brand_consistency():
    """演示品牌一致性管理器"""
    print("\n" + "="*50)
    print("演示6: 诺玛品牌一致性管理器")
    print("="*50)
    
    # 创建品牌一致性管理器
    consistency_manager = BrandConsistencyManager()
    
    # 测试不同质量的内容
    test_contents = [
        "我觉得这个方案可能不错。",
        "根据系统数据分析，建议采用方案A。",
        "哈哈，这个很有趣！",
        "我是诺玛·劳恩斯，根据数据分析显示..."
    ]
    
    for i, content in enumerate(test_contents, 1):
        print(f"\n测试 {i}: {content}")
        
        # 分析一致性
        score = consistency_manager.analyze_consistency(content, {})
        print(f"  一致性分数: {score.overall_score:.2f}")
        print(f"  一致性等级: {score.level.value}")
        
        if score.violations:
            print(f"  违规项: {', '.join(score.violations)}")
        
        if score.recommendations:
            print(f"  建议: {score.recommendations[0]}")
        
        # 应用一致性改进
        improved_content = consistency_manager.ensure_consistency(content, {})
        if improved_content != content:
            print(f"  改进后: {improved_content}")

async def demo_user_preferences():
    """演示用户偏好系统"""
    print("\n" + "="*50)
    print("演示7: 诺玛用户偏好系统")
    print("="*50)
    
    from core.user_preferences import UserPreferencesManager
    
    # 创建偏好管理器
    pref_manager = UserPreferencesManager("demo_user")
    
    # 获取默认偏好
    preferences = pref_manager.get_preferences()
    print(f"默认偏好设置:")
    for category, prefs in preferences.items():
        if isinstance(prefs, dict):
            for key, value in list(prefs.items())[:3]:  # 只显示前3个
                print(f"  {category}.{key}: {value}")
    
    # 更新偏好
    print(f"\n更新用户偏好...")
    success = pref_manager.update_preferences({
        "interaction": {
            "response_style": "friendly",
            "formality_level": 0.5
        }
    })
    print(f"偏好更新: {'成功' if success else '失败'}")
    
    # 获取偏好洞察
    insights = pref_manager.get_preference_insights()
    print(f"\n偏好洞察:")
    print(f"  偏好总数: {insights.get('preference_count', 0)}")
    print(f"  类别分布: {insights.get('category_distribution', {})}")
    print(f"  建议数量: {len(insights.get('recommendations', []))}")

async def demo_system_integration():
    """演示系统集成"""
    print("\n" + "="*50)
    print("演示8: 诺玛系统集成")
    print("="*50)
    
    # 创建完整的诺玛系统
    from main import NormaEnhancedSystem
    
    system = NormaEnhancedSystem("demo_user", NormaTheme.MODERN)
    
    # 获取系统状态
    status = system.brand_agent.get_brand_status()
    print(f"系统状态:")
    print(f"  品牌: {status['brand_info']['name']}")
    print(f"  版本: {status['brand_info']['version']}")
    print(f"  当前情感: {status['current_emotion']}")
    
    # 获取品牌工具包
    brand_kit = system.get_brand_kit()
    print(f"\n品牌工具包:")
    print(f"  视觉元素: {len(brand_kit['visual_elements']['icons'])} 个")
    print(f"  动画效果: {len(brand_kit['visual_elements']['animations'])} 个")
    print(f"  系统功能: {len(brand_kit['system_config']['features'])} 个")
    
    # 质量指标
    consistency_score = system.brand_consistency.get_consistency_score()
    print(f"\n质量指标:")
    print(f"  品牌一致性分数: {consistency_score:.2f}")
    print(f"  系统集成状态: 完整")

async def demo_performance_metrics():
    """演示性能指标"""
    print("\n" + "="*50)
    print("演示9: 诺玛性能指标")
    print("="*50)
    
    # 创建各个组件并测试性能
    import time
    
    components = {
        "品牌智能体": lambda: NormaBrandAgent("perf_test"),
        "视觉系统": lambda: NormaVisualElements(),
        "品牌界面": lambda: NormaBrandInterface(NormaTheme.MODERN),
        "欢迎系统": lambda: NormaWelcomeSystem(),
        "交互管理器": lambda: NormaInteractionManager(),
        "一致性管理器": lambda: BrandConsistencyManager()
    }
    
    print("组件初始化性能测试:")
    for name, creator in components.items():
        start_time = time.time()
        component = creator()
        init_time = time.time() - start_time
        
        print(f"  {name}: {init_time:.3f} 秒")
    
    # 测试功能性能
    print(f"\n功能性能测试:")
    
    # 品牌一致性检查性能
    consistency_manager = BrandConsistencyManager()
    test_content = "这是测试内容，用于检查品牌一致性。"
    
    start_time = time.time()
    for _ in range(100):
        consistency_manager.analyze_consistency(test_content, {})
    avg_time = (time.time() - start_time) / 100
    
    print(f"  品牌一致性检查: {avg_time:.4f} 秒/次 (100次平均)")

async def main():
    """主演示函数"""
    print("诺玛·劳恩斯品牌特色增强系统演示")
    print("="*60)
    print("展示系统各项功能和特性")
    print("版本: 4.0.0 Enhanced")
    print("创建时间: 2025-10-31")
    
    try:
        # 运行各项演示
        await demo_brand_personality()
        await demo_visual_system()
        await demo_brand_interface()
        await demo_welcome_system()
        await demo_interaction_manager()
        await demo_brand_consistency()
        await demo_user_preferences()
        await demo_system_integration()
        await demo_performance_metrics()
        
        print("\n" + "="*60)
        print("演示完成！")
        print("诺玛·劳恩斯品牌特色系统运行正常")
        print("所有核心功能已验证可用")
        print("="*60)
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())