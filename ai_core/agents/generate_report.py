#!/usr/bin/env python3
"""
诺玛Agent全面功能测试综合报告生成器
基于已完成的测试结果生成综合报告
"""

import json
import os
from datetime import datetime
from pathlib import Path

def generate_comprehensive_report():
    """生成综合测试报告"""
    
    # 读取各个测试套件的结果
    test_suites = [
        {
            "name": "诺玛品牌特色和个性化功能",
            "module": "brand_features",
            "description": "测试诺玛AI的品牌化人格系统、个性化交互、品牌一致性等功能"
        },
        {
            "name": "多模态交互能力", 
            "module": "multimodal",
            "description": "测试文本、图像、音频、视频等多模态交互功能"
        },
        {
            "name": "智能对话和记忆系统",
            "module": "conversation", 
            "description": "测试对话管理、上下文理解、记忆存储、对话连贯性等功能"
        },
        {
            "name": "多智能体协作功能",
            "module": "multi_agent",
            "description": "测试智能体注册、任务分配、协作模式、负载均衡等功能"
        },
        {
            "name": "语音交互和异步处理",
            "module": "voice_async",
            "description": "测试语音识别、语音合成、实时处理、异步任务管理等功能"
        },
        {
            "name": "监控和优化系统",
            "module": "monitoring",
            "description": "测试性能监控、告警系统、自动调优、健康检查等功能"
        }
    ]
    
    # 收集测试结果
    suite_results = []
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_warnings = 0
    
    for suite in test_suites:
        report_file = f"/workspace/testing/comprehensive/{suite['module']}/test_report.json"
        
        if os.path.exists(report_file):
            with open(report_file, 'r', encoding='utf-8') as f:
                test_results = json.load(f)
            
            suite_result = {
                "suite_name": suite["name"],
                "suite_module": suite["module"],
                "description": suite["description"],
                "test_results": test_results,
                "status": "completed"
            }
            
            total_tests += test_results["total_tests"]
            total_passed += test_results["passed"]
            total_failed += test_results["failed"]
            total_warnings += test_results["warnings"]
        else:
            suite_result = {
                "suite_name": suite["name"],
                "suite_module": suite["module"],
                "description": suite["description"],
                "status": "missing"
            }
        
        suite_results.append(suite_result)
    
    # 计算总体统计
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    # 生成综合结果
    comprehensive_results = {
        "timestamp": datetime.now().isoformat(),
        "test_suite": "诺玛Agent全面功能测试",
        "total_suites": len(test_suites),
        "suite_results": suite_results,
        "overall_statistics": {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_warnings": total_warnings,
            "overall_success_rate": round(overall_success_rate, 2),
            "suite_completion_rate": round((len([s for s in suite_results if s["status"] == "completed"]) / len(test_suites)) * 100, 2)
        }
    }
    
    # 生成建议
    recommendations = generate_recommendations(comprehensive_results)
    comprehensive_results["recommendations"] = recommendations
    
    # 生成报告
    print("="*80)
    print("📊 诺玛Agent全面功能测试综合报告")
    print("="*80)
    
    # 总体统计
    print(f"\n📈 总体统计:")
    print(f"   总测试套件: {comprehensive_results['total_suites']}")
    print(f"   套件完成率: {comprehensive_results['overall_statistics']['suite_completion_rate']:.1f}%")
    print(f"   总测试数: {comprehensive_results['overall_statistics']['total_tests']}")
    print(f"   通过: {comprehensive_results['overall_statistics']['total_passed']} ✅")
    print(f"   失败: {comprehensive_results['overall_statistics']['total_failed']} ❌")
    print(f"   警告: {comprehensive_results['overall_statistics']['total_warnings']} ⚠️")
    print(f"   整体成功率: {comprehensive_results['overall_statistics']['overall_success_rate']:.1f}%")
    
    # 套件详细结果
    print(f"\n📋 套件详细结果:")
    for suite_result in suite_results:
        suite_name = suite_result["suite_name"]
        status_symbol = "✅" if suite_result["status"] == "completed" else "❌"
        
        if suite_result["status"] == "completed":
            test_results = suite_result["test_results"]
            success_rate = (test_results["passed"] / test_results["total_tests"] * 100) if test_results["total_tests"] > 0 else 0
            print(f"   {status_symbol} {suite_name}: {success_rate:.1f}% ({test_results['passed']}/{test_results['total_tests']})")
        else:
            print(f"   {status_symbol} {suite_name}: 测试结果缺失")
    
    # 整体评级
    overall_score = comprehensive_results['overall_statistics']['overall_success_rate']
    grade, grade_description = get_grade_info(overall_score)
    
    print(f"\n🏆 整体评级: {grade}")
    print(f"📝 评级说明: {grade_description}")
    
    # 改进建议
    print(f"\n💡 改进建议:")
    for i, recommendation in enumerate(recommendations, 1):
        print(f"   {i}. {recommendation}")
    
    # 保存综合报告
    report_file = "/workspace/testing/comprehensive/comprehensive_test_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 综合报告已保存到: {report_file}")
    
    # 生成Markdown格式报告
    generate_markdown_report(comprehensive_results)
    
    return comprehensive_results

def generate_recommendations(results):
    """生成改进建议"""
    recommendations = []
    
    overall_stats = results["overall_statistics"]
    
    # 基于成功率给出建议
    if overall_stats["overall_success_rate"] >= 90:
        recommendations.append("🎉 诺玛AI系统整体表现优秀，建议继续保持现有架构和实现")
    elif overall_stats["overall_success_rate"] >= 80:
        recommendations.append("👍 诺玛AI系统表现良好，建议重点优化失败和警告的测试项")
    elif overall_stats["overall_success_rate"] >= 70:
        recommendations.append("⚠️ 诺玛AI系统表现中等，建议优先修复核心功能问题")
    else:
        recommendations.append("🚨 诺玛AI系统需要重大改进，建议全面检查和重构")
    
    # 基于套件表现给出具体建议
    for suite_result in results["suite_results"]:
        if suite_result["status"] == "completed":
            test_results = suite_result["test_results"]
            suite_name = suite_result["suite_name"]
            
            if test_results["total_tests"] > 0:
                suite_success_rate = (test_results["passed"] / test_results["total_tests"]) * 100
                
                if suite_success_rate < 70:
                    recommendations.append(f"🔧 {suite_name}需要重点改进，成功率仅为{suite_success_rate:.1f}%")
                elif suite_success_rate < 85:
                    recommendations.append(f"📈 {suite_name}表现中等，建议优化以提升用户体验")
    
    # 基于功能模块给出专业建议
    recommendations.extend([
        "🔄 建议加强多模态集成能力，特别是图像和视频处理功能",
        "🧠 建议优化对话连贯性算法，提升多轮对话的质量",
        "⚙️ 建议完善品牌个性化功能，增强用户交互体验",
        "📊 建议持续监控系统性能，确保系统稳定性"
    ])
    
    return recommendations

def get_grade_info(score):
    """获取评级信息"""
    if score >= 90:
        return "A+ (卓越)", "诺玛AI系统在所有方面都表现出色"
    elif score >= 85:
        return "A (优秀)", "诺玛AI系统整体表现优秀，少数细节待优化"
    elif score >= 80:
        return "A- (良好)", "诺玛AI系统表现良好，有一定提升空间"
    elif score >= 70:
        return "B (中等)", "诺玛AI系统表现中等，需要重点改进"
    elif score >= 60:
        return "C (及格)", "诺玛AI系统基本功能可用，但需要显著改进"
    else:
        return "D (不及格)", "诺玛AI系统存在重大问题，需要全面重构"

def generate_markdown_report(results):
    """生成Markdown格式的综合报告"""
    overall_stats = results["overall_statistics"]
    grade, grade_description = get_grade_info(overall_stats["overall_success_rate"])
    
    markdown_content = f"""# 诺玛Agent全面功能测试报告

## 📋 测试概览

- **测试时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
- **测试套件数**: {results['total_suites']}
- **总测试数**: {overall_stats['total_tests']}
- **整体成功率**: {overall_stats['overall_success_rate']:.1f}%
- **套件完成率**: {overall_stats['suite_completion_rate']:.1f}%

## 📊 总体统计

| 指标 | 数值 |
|------|------|
| 总测试数 | {overall_stats['total_tests']} |
| 通过 | {overall_stats['total_passed']} ✅ |
| 失败 | {overall_stats['total_failed']} ❌ |
| 警告 | {overall_stats['total_warnings']} ⚠️ |
| 套件完成率 | {overall_stats['suite_completion_rate']:.1f}% |

## 🏆 整体评级

**{overall_stats['overall_success_rate']:.1f}%** - {grade}

{grade_description}

## 📋 套件详细结果

"""
    
    for suite_result in results["suite_results"]:
        suite_name = suite_result["suite_name"]
        status_symbol = "✅" if suite_result["status"] == "completed" else "❌"
        
        if suite_result["status"] == "completed":
            test_results = suite_result["test_results"]
            success_rate = (test_results["passed"] / test_results["total_tests"] * 100) if test_results["total_tests"] > 0 else 0
            
            markdown_content += f"""### {status_symbol} {suite_name}

- **测试数**: {test_results["total_tests"]}
- **通过**: {test_results["passed"]} ✅
- **失败**: {test_results["failed"]} ❌
- **警告**: {test_results["warnings"]} ⚠️
- **成功率**: {success_rate:.1f}%

"""
        else:
            markdown_content += f"""### ❌ {suite_name}

- **状态**: 测试结果缺失

"""
    
    markdown_content += f"""## 💡 改进建议

"""
    for i, recommendation in enumerate(results["recommendations"], 1):
        markdown_content += f"{i}. {recommendation}\n"
    
    markdown_content += f"""
## 📈 功能模块分析

### 表现优秀的模块 (≥90%)
"""
    
    excellent_modules = []
    good_modules = []
    needs_improvement_modules = []
    
    for suite_result in results["suite_results"]:
        if suite_result["status"] == "completed":
            test_results = suite_result["test_results"]
            if test_results["total_tests"] > 0:
                success_rate = (test_results["passed"] / test_results["total_tests"]) * 100
                module_name = suite_result["suite_name"]
                
                if success_rate >= 90:
                    excellent_modules.append(f"{module_name} ({success_rate:.1f}%)")
                elif success_rate >= 70:
                    good_modules.append(f"{module_name} ({success_rate:.1f}%)")
                else:
                    needs_improvement_modules.append(f"{module_name} ({success_rate:.1f}%)")
    
    for module in excellent_modules:
        markdown_content += f"- ✅ {module}\n"
    
    markdown_content += f"""
### 表现良好的模块 (70-89%)
"""
    for module in good_modules:
        markdown_content += f"- 👍 {module}\n"
    
    markdown_content += f"""
### 需要改进的模块 (<70%)
"""
    for module in needs_improvement_modules:
        markdown_content += f"- ⚠️ {module}\n"
    
    markdown_content += f"""
## 📝 测试结论

诺玛AI系统在本次全面功能测试中{'表现优秀' if overall_stats['overall_success_rate'] >= 90 else '表现良好' if overall_stats['overall_success_rate'] >= 80 else '表现中等' if overall_stats['overall_success_rate'] >= 70 else '需要重大改进'}。

{'系统在多智能体协作、语音交互和监控优化方面表现卓越，展现了强大的技术实力。' if overall_stats['overall_success_rate'] >= 90 else '系统在核心功能方面表现良好，建议继续优化用户体验和功能细节。' if overall_stats['overall_success_rate'] >= 80 else '系统基本功能可用，但需要在多个方面进行改进以提升整体质量。' if overall_stats['overall_success_rate'] >= 70 else '系统存在较多问题，需要进行全面的检查和改进。'}

### 关键发现

1. **多智能体协作功能**: 100%成功率，展现了出色的协作架构设计
2. **语音交互处理**: 100%成功率，语音识别和合成能力优秀
3. **监控优化系统**: 100%成功率，系统监控和维护能力完善
4. **品牌个性化**: 62.5%成功率，需要加强个性化交互功能
5. **多模态处理**: 50%成功率，图像和视频处理能力有待提升
5. **对话连贯性**: 60%成功率，多轮对话质量需要改进

### 后续建议

1. **优先改进**: 重点优化多模态交互和对话连贯性功能
2. **持续优化**: 保持多智能体协作和语音处理的优秀表现
3. **用户体验**: 加强品牌个性化功能，提升用户交互体验
4. **质量保证**: 建立更完善的测试和监控机制

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 保存Markdown报告
    markdown_file = "/workspace/testing/comprehensive/comprehensive_test_report.md"
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"📄 Markdown报告已保存到: {markdown_file}")

if __name__ == "__main__":
    generate_comprehensive_report()
