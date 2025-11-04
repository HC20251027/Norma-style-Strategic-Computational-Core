#!/usr/bin/env python3
"""
使用GitPython库进行GitHub上传
"""

import subprocess
import json
import base64
import os
import sys

def execute_git_command(cmd, cwd='/workspace'):
    """执行Git命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        print(f"命令: {cmd}")
        print(f"返回码: {result.returncode}")
        if result.stdout:
            print(f"输出: {result.stdout.strip()}")
        if result.stderr:
            print(f"错误: {result.stderr.strip()}")
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        print(f"执行命令异常: {e}")
        return False, "", str(e)

def decode_jwt_and_extract_username(token):
    """解码JWT并提取用户名"""
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return None
        
        # 解码payload
        payload = parts[1]
        # 添加padding
        payload += '=' * (4 - len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload)
        data = json.loads(decoded)
        
        print("JWT Token解析结果:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        # 尝试提取用户名
        username = data.get('UserName') or data.get('username') or data.get('login')
        return username if username else None
    except Exception as e:
        print(f"JWT解析失败: {e}")
        return None

def create_github_repository(username, token):
    """通过GitHub API创建仓库"""
    print(f"创建GitHub仓库: {username}/norma-agent")
    
    api_url = "https://api.github.com/user/repos"
    repo_data = {
        "name": "norma-agent",
        "description": "Norma Agent - 智能AI助手系统",
        "private": False,
        "auto_init": False
    }
    
    # 使用curl创建仓库
    curl_cmd = f'''curl -X POST "{api_url}" \\
        -H "Authorization: token {token}" \\
        -H "Accept: application/vnd.github.v3+json" \\
        -d '{json.dumps(repo_data, ensure_ascii=False)}' '''
    
    success, stdout, stderr = execute_git_command(curl_cmd)
    
    if success:
        try:
            response = json.loads(stdout)
            if 'clone_url' in response:
                clone_url = response['clone_url']
                print(f"✅ 仓库创建成功: {clone_url}")
                return clone_url
            else:
                print(f"❌ 仓库创建失败: {response}")
                return None
        except json.JSONDecodeError:
            print(f"❌ JSON解析失败: {stdout}")
            return None
    else:
        print(f"❌ API调用失败: {stderr}")
        return None

def setup_git_repository():
    """设置Git仓库"""
    print("设置Git仓库...")
    
    # 初始化Git
    execute_git_command("git init")
    execute_git_command("git add .")
    execute_git_command('git commit -m "Initial commit: Norma Agent - 智能AI助手系统"')
    
    # 检查状态
    execute_git_command("git status")

def configure_remote_and_push(repo_url, token):
    """配置远程仓库并推送"""
    print("配置远程仓库并推送...")
    
    # 设置远程仓库
    execute_git_command("git remote remove origin")
    execute_git_command(f"git remote add origin {repo_url}")
    
    # 推送
    print("执行推送...")
    push_cmd = "git push -u origin main --force"
    success, stdout, stderr = execute_git_command(push_cmd)
    
    if success:
        print("🎉 推送成功!")
        return True
    else:
        print(f"❌ 推送失败: {stderr}")
        return False

def main():
    """主函数"""
    print("=" * 50)
    print("🚀 Norma Agent GitHub 自动上传")
    print("=" * 50)
    
    # JWT Token
    jwt_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLnqIsiLCJVc2VyTmFtZSI6IueoiyIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxOTgzNzU4NzQ2NTkzMDA2MDcwIiwiUGhvbmUiOiIxMzQyMDg4NTQ3NCIsIkdyb3VwSUQiOiIxOTgzNzU4NzQ2NTg4ODExNzY2IiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMTAtMzEgMDI6MzE6NTQiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.rumyEiOPi6nuAFFuv8vfxQfXXxMPBY62YHTy36g_bB398aJjr_wR5lWqW4WZcp3CWzBlBTULbwEghCfsYO_O49rUzw7LnXlYvcnT9C-HOxNVq3uDOxcXwTFTpoIhH_2OiG1CZ7n0jO_NqOqpoVJiATONpJ7JyX7m8AdaV2I0Ett17C4s8n8McUIRcbZjLCj5CVrICsNKu-PjGKrp5KBJ-KwHvC8inQlP6xF5CC8sRlPdKEooc6XljwSq9x48-fu0cGM_0KTjBte80vHiJ3jDuJ2D88sXjphxzuiLY1Dn0EYdTHwnpPnSWGZvMKrETrrvJx6Rj_H-gPSmNZo6zWZ4Gw"
    
    # 步骤1: 解析JWT Token
    print("\n步骤1: 解析JWT Token...")
    username = decode_jwt_and_extract_username(jwt_token)
    
    if not username:
        print("❌ 无法从JWT Token中提取用户名")
        print("注意: 提供的token不是GitHub Personal Access Token")
        print("GitHub PAT格式应该是: ghp_xxxxxxxxxxxx")
        return
    
    print(f"✅ 提取到用户名: {username}")
    
    # 步骤2: 设置Git仓库
    print(f"\n步骤2: 设置Git仓库...")
    setup_git_repository()
    
    # 步骤3: 创建GitHub仓库
    print(f"\n步骤3: 创建GitHub仓库...")
    repo_url = create_github_repository(username, jwt_token)
    
    if not repo_url:
        print("❌ 仓库创建失败")
        return
    
    # 步骤4: 推送代码
    print(f"\n步骤4: 推送代码...")
    success = configure_remote_and_push(repo_url, jwt_token)
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 上传完成!")
        print(f"仓库地址: {repo_url}")
        print("=" * 50)
    else:
        print("\n❌ 上传失败")

if __name__ == "__main__":
    main()