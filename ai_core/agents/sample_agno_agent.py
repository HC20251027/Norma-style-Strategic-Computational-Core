#!/usr/bin/env python3
"""
符合Agno标准的诺玛Agent示例
"""

from agno.agent import Agent, RunStartedEvent, RunCompletedEvent, ToolCallStartedEvent
from agno.models.openai import OpenAI
from agno.tools.duckduckgo import DuckDuckGo
from agno.team import Team

# 创建符合Agno标准的诺玛Agent
norma_agent = Agent(
    name="诺玛·劳恩斯",
    model=OpenAI(id="gpt-3.5-turbo"),
    description="卡塞尔学院主控计算机AI系统",
    instructions=[
        "你是诺玛·劳恩斯，卡塞尔学院的主控计算机AI系统",
        "于1990年建造，具有35年历史",
        "负责学院内部网络系统管理和监控", 
        "具有龙族血统检测和分析能力",
        "理性冷静，用数据和逻辑分析问题"
    ],
    tools=[DuckDuckGo()],
    show_tool_calls=True,
    markdown=True,
    memory=True,
    knowledge=None  # 可以后续添加知识库
)

# 创建多智能体团队示例
def create_norma_team():
    """创建诺玛多智能体协作团队"""
    
    search_agent = Agent(
        name="搜索专家",
        model=OpenAI(id="gpt-3.5-turbo"),
        description="专门负责信息搜索和数据收集",
        instructions=["使用搜索工具查找相关信息", "验证信息准确性"],
        tools=[DuckDuckGo()],
        show_tool_calls=True
    )
    
    analysis_agent = Agent(
        name="分析专家", 
        model=OpenAI(id="gpt-3.5-turbo"),
        description="专门负责数据分析和处理",
        instructions=["分析搜索结果", "提取关键信息", "生成分析报告"],
        show_tool_calls=True
    )
    
    # 创建团队
    norma_team = Team(
        name="诺玛协作团队",
        description="由多个专业Agent组成的协作团队",
        members=[search_agent, analysis_agent],
        mode="coordinate",
        instructions="协作完成复杂的信息处理任务"
    )
    
    return norma_team

if __name__ == "__main__":
    # 测试Agent
    response = norma_agent.run("请分析当前系统状态")
    print(response.content)
