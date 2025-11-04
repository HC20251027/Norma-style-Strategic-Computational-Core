#!/usr/bin/env python3

import subprocess
import json
import base64

# JWT Token
jwt_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLnqIsiLCJVc2VyTmFtZSI6IueoiyIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxOTgzNzU4NzQ2NTkzMDA2MDcwIiwiUGhvbmUiOiIxMzQyMDg4NTQ3NCIsIkdyb3VwSUQiOiIxOTgzNzU4NzQ2NTg4ODExNzY2IiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMTAtMzEgMDI6MzE6NTQiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.rumyEiOPi6nuAFFuv8vfxQfXXxMPBY62YHTy36g_bB398aJjr_wR5lWqW4WZcp3CWzBlBTULbwEghCfsYO_O49rUzw7LnXlYvcnT9C-HOxNVq3uDOxcXwTFTpoIhH_2OiG1CZ7n0jO_NqOqpoVJiATONpJ7JyX7m8AdaV2I0Ett17C4s8n8McUIRcbZjLCj5CVrICsNKu-PjGKrp5KBJ-KwHvC8inQlP6xF5CC8sRlPdKEooc6XljwSq9x48-fu0cGM_0KTjBte80vHiJ3jDuJ2D88sXjphxzuiLY1Dn0EYdTHwnpPnSWGZvMKrETrrvJx6Rj_H-gPSmNZo6zWZ4Gw"

print("🚀 开始GitHub自动上传流程")
print("=" * 50)

# 1. 解析JWT Token
print("\n步骤1: 解析JWT Token...")
try:
    parts = jwt_token.split('.')
    if len(parts) == 3:
        payload = parts[1]
        payload += '=' * (4 - len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload)
        data = json.loads(decoded)
        
        print("JWT Token解析结果:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        # 提取用户名
        username = data.get('UserName') or data.get('username') or data.get('login')
        if username:
            print(f"✅ 提取到用户名: {username}")
        else:
            print("❌ 无法从token中提取用户名")
            username = "minimax-user"
    else:
        print("❌ JWT token格式无效")
        username = "minimax-user"
except Exception as e:
    print(f"❌ JWT解析失败: {e}")
    username = "minimax-user"

# 2. 初始化Git仓库
print(f"\n步骤2: 初始化Git仓库...")
try:
    subprocess.run(["git", "init"], cwd="/workspace", capture_output=True)
    subprocess.run(["git", "add", "."], cwd="/workspace", capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit: Norma Agent - 智能AI助手系统"], cwd="/workspace", capture_output=True)
    print("✅ Git仓库初始化完成")
except Exception as e:
    print(f"❌ Git初始化失败: {e}")

# 3. 创建GitHub仓库
print(f"\n步骤3: 创建GitHub仓库...")
try:
    api_url = "https://api.github.com/user/repos"
    repo_data = {
        "name": "norma-agent",
        "description": "Norma Agent - 智能AI助手系统",
        "private": False,
        "auto_init": False
    }
    
    # 使用curl创建仓库
    curl_cmd = [
        "curl", "-X", "POST", api_url,
        "-H", f"Authorization: token {jwt_token}",
        "-H", "Accept: application/vnd.github.v3+json",
        "-d", json.dumps(repo_data, ensure_ascii=False)
    ]
    
    result = subprocess.run(curl_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        try:
            response = json.loads(result.stdout)
            if 'clone_url' in response:
                repo_url = response['clone_url']
                print(f"✅ 仓库创建成功: {repo_url}")
            else:
                print(f"❌ 仓库创建失败: {response}")
                repo_url = None
        except json.JSONDecodeError:
            print(f"❌ JSON解析失败: {result.stdout}")
            repo_url = None
    else:
        print(f"❌ API调用失败: {result.stderr}")
        repo_url = None
except Exception as e:
    print(f"❌ 仓库创建异常: {e}")
    repo_url = None

# 4. 配置远程仓库并推送
if repo_url:
    print(f"\n步骤4: 配置远程仓库并推送...")
    try:
        # 设置远程仓库
        subprocess.run(["git", "remote", "remove", "origin"], cwd="/workspace", capture_output=True)
        subprocess.run(["git", "remote", "add", "origin", repo_url], cwd="/workspace", capture_output=True)
        
        # 推送
        print("执行推送...")
        result = subprocess.run(["git", "push", "-u", "origin", "main", "--force"], cwd="/workspace", capture_output=True, text=True)
        
        if result.returncode == 0:
            print("🎉 推送成功!")
            print(f"仓库地址: {repo_url}")
        else:
            print(f"❌ 推送失败: {result.stderr}")
    except Exception as e:
        print(f"❌ 推送异常: {e}")
else:
    print("❌ 无法创建仓库，跳过推送")

print("\n" + "=" * 50)
print("上传流程完成")
print("=" * 50)