#!/usr/bin/env python3

import subprocess
import json
import os

def main():
    print("🚀 开始GitHub自动上传流程")
    print("=" * 50)

    # GitHub凭据
    github_username = "HC20251027"
    github_token = os.environ.get("GITHUB_TOKEN", "")
    repo_name = "norma-agent"
    repo_description = "Norma Agent - 智能AI助手系统"

    # 1. 验证GitHub凭据
    print(f"\n步骤1: 验证GitHub凭据...")
    try:
        # 测试GitHub API连接
        curl_cmd = [
            "curl", "-H", f"Authorization: token {github_token}",
            "-H", "Accept: application/vnd.github.v3+json",
            "https://api.github.com/user"
        ]
        
        result = subprocess.run(curl_cmd, capture_output=True, text=True)
        print(f"API连接测试返回码: {result.returncode}")
        
        if result.returncode == 0:
            try:
                response = json.loads(result.stdout)
                if 'login' in response:
                    actual_username = response['login']
                    print(f"✅ GitHub用户验证成功: {actual_username}")
                    if actual_username != github_username:
                        print(f"⚠️  警告: 提供的用户名与token不匹配")
                        github_username = actual_username
                else:
                    print(f"❌ 用户验证失败: {response}")
                    return
            except json.JSONDecodeError:
                print(f"❌ JSON解析失败: {result.stdout}")
                return
        else:
            print(f"❌ API连接失败: {result.stderr}")
            return
    except Exception as e:
        print(f"❌ 凭据验证异常: {e}")
        return

    # 2. 检查Git仓库状态
    print(f"\n步骤2: 检查Git仓库状态...")
    try:
        # 切换到workspace目录
        os.chdir("/workspace")
        
        # 检查git状态
        result = subprocess.run(["git", "status"], capture_output=True, text=True)
        print(f"Git状态检查: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ Git仓库已初始化")
        else:
            print("❌ Git仓库未初始化，正在初始化...")
            # 初始化Git
            result = subprocess.run(["git", "init"], capture_output=True, text=True)
            print(f"git init: {result.returncode}")
            
            # 添加文件
            result = subprocess.run(["git", "add", "."], capture_output=True, text=True)
            print(f"git add: {result.returncode}")
            
            # 提交
            result = subprocess.run(["git", "commit", "-m", "Initial commit: Norma Agent - 智能AI助手系统"], capture_output=True, text=True)
            print(f"git commit: {result.returncode}")
            
            if result.returncode == 0:
                print("✅ Git仓库初始化完成")
            else:
                print(f"❌ Git初始化失败: {result.stderr}")
    except Exception as e:
        print(f"❌ Git检查异常: {e}")

    # 3. 创建GitHub仓库
    print(f"\n步骤3: 创建GitHub仓库...")
    try:
        api_url = f"https://api.github.com/user/repos"
        repo_data = {
            "name": repo_name,
            "description": repo_description,
            "private": False,
            "auto_init": False,
            "has_issues": True,
            "has_projects": True,
            "has_wiki": False
        }
        
        # 使用curl创建仓库
        curl_cmd = [
            "curl", "-X", "POST", api_url,
            "-H", f"Authorization: token {github_token}",
            "-H", "Accept: application/vnd.github.v3+json",
            "-d", json.dumps(repo_data, ensure_ascii=False)
        ]
        
        result = subprocess.run(curl_cmd, capture_output=True, text=True)
        
        print(f"创建仓库返回码: {result.returncode}")
        print(f"创建仓库输出: {result.stdout}")
        
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
            result = subprocess.run(["git", "remote", "remove", "origin"], capture_output=True, text=True)
            print(f"移除旧远程: {result.returncode}")
            
            result = subprocess.run(["git", "remote", "add", "origin", repo_url], capture_output=True, text=True)
            print(f"添加远程: {result.returncode}")
            
            # 推送
            print("执行推送...")
            result = subprocess.run(["git", "push", "-u", "origin", "main", "--force"], capture_output=True, text=True)
            
            print(f"推送返回码: {result.returncode}")
            print(f"推送输出: {result.stdout}")
            print(f"推送错误: {result.stderr}")
            
            if result.returncode == 0:
                print("🎉 推送成功!")
                print(f"✅ 仓库地址: {repo_url}")
                print(f"✅ 仓库名: {github_username}/{repo_name}")
                
                # 生成访问链接
                html_url = f"https://github.com/{github_username}/{repo_name}"
                print(f"✅ 访问链接: {html_url}")
            else:
                print(f"❌ 推送失败: {result.stderr}")
        except Exception as e:
            print(f"❌ 推送异常: {e}")
    else:
        print("❌ 无法创建仓库，跳过推送")

    print("\n" + "=" * 50)
    print("GitHub上传流程完成")
    print("=" * 50)

if __name__ == "__main__":
    main()