#!/usr/bin/env python3
"""
直接执行GitHub上传
"""

import subprocess
import json
import base64
import os
import sys

def run_command(cmd):
    """执行shell命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/workspace')
        print(f"执行命令: {cmd}")
        print(f"返回码: {result.returncode}")
        if result.stdout:
            print(f"标准输出: {result.stdout}")
        if result.stderr:
            print(f"错误输出: {result.stderr}")
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        print(f"命令执行异常: {e}")
        return False, "", str(e)

def main():
    print("🚀 开始GitHub自动上传流程")
    
    # 1. 检查当前目录
    print("\n=== 步骤1: 检查当前目录 ===")
    success, stdout, stderr = run_command("pwd")
    print(f"当前目录: {stdout.strip()}")
    
    # 2. 初始化Git仓库
    print("\n=== 步骤2: 初始化Git仓库 ===")
    run_command("git init")
    run_command("git add .")
    run_command('git commit -m "Initial commit: Norma Agent project"')
    
    # 3. 检查Git状态
    print("\n=== 步骤3: 检查Git状态 ===")
    success, stdout, stderr = run_command("git status")
    print(f"Git状态: {stdout}")
    
    # 4. 尝试从JWT token提取用户名
    print("\n=== 步骤4: 解析JWT token ===")
    jwt_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLnqIsiLCJVc2VyTmFtZSI6IueoiyIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxOTgzNzU4NzQ2NTkzMDA2MDcwIiwiUGhvbmUiOiIxMzQyMDg4NTQ3NCIsIkdyb3VwSUQiOiIxOTgzNzU4NzQ2NTg4ODExNzY2IiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMTAtMzEgMDI6MzE6NTQiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.rumyEiOPi6nuAFFuv8vfxQfXXxMPBY62YHTy36g_bB398aJjr_wR5lWqW4WZcp3CWzBlBTULbwEghCfsYO_O49rUzw7LnXlYvcnT9C-HOxNVq3uDOxcXwTFTpoIhH_2OiG1CZ7n0jO_NqOqpoVJiATONpJ7JyX7m8AdaV2I0Ett17C4s8n8McUIRcbZjLCj5CVrICsNKu-PjGKrp5KBJ-KwHvC8inQlP6xF5CC8sRlPdKEooc6XljwSq9x48-fu0cGM_0KTjBte80vHiJ3jDuJ2D88sXjphxzuiLY1Dn0EYdTHwnpPnSWGZvMKrETrrvJx6Rj_H-gPSmNZo6zWZ4Gw"
    
    try:
        parts = jwt_token.split('.')
        if len(parts) == 3:
            payload = parts[1]
            payload += '=' * (4 - len(payload) % 4)
            decoded = base64.urlsafe_b64decode(payload)
            data = json.loads(decoded)
            print(f"Token数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
            # 提取用户名
            username = data.get('UserName') or data.get('username') or data.get('login')
            if username:
                print(f"提取到用户名: {username}")
            else:
                print("无法从token中提取用户名")
                username = "minimax-user"  # 默认用户名
        else:
            print("JWT token格式无效")
            username = "minimax-user"
    except Exception as e:
        print(f"解析JWT token失败: {e}")
        username = "minimax-user"
    
    # 5. 创建GitHub仓库
    print(f"\n=== 步骤5: 创建GitHub仓库 ===")
    api_url = "https://api.github.com/user/repos"
    data = {
        "name": "norma-agent",
        "description": "Norma Agent - 智能AI助手系统",
        "private": False,
        "auto_init": False
    }
    
    curl_cmd = f'''curl -X POST "{api_url}" \\
        -H "Authorization: token {jwt_token}" \\
        -H "Accept: application/vnd.github.v3+json" \\
        -d '{json.dumps(data, ensure_ascii=False)}' '''
    
    success, stdout, stderr = run_command(curl_cmd)
    
    repo_url = None
    if success:
        try:
            response = json.loads(stdout)
            if 'clone_url' in response:
                repo_url = response['clone_url']
                print(f"✅ 仓库创建成功: {repo_url}")
            else:
                print(f"❌ 仓库创建失败: {response}")
        except:
            print(f"❌ API响应解析失败: {stdout}")
    else:
        print(f"❌ API调用失败: {stderr}")
    
    # 6. 配置远程仓库
    if repo_url:
        print("\n=== 步骤6: 配置Git远程仓库 ===")
        run_command("git remote remove origin")
        run_command(f"git remote add origin {repo_url}")
        
        # 7. 推送到GitHub
        print("\n=== 步骤7: 推送到GitHub ===")
        push_cmd = "git push -u origin main --force"
        success, stdout, stderr = run_command(push_cmd)
        
        if success:
            print("🎉 推送成功!")
            print(f"仓库地址: {repo_url}")
        else:
            print(f"❌ 推送失败: {stderr}")
    else:
        print("❌ 无法创建仓库，跳过推送")
    
    print("\n=== 上传流程完成 ===")

if __name__ == "__main__":
    main()