#!/usr/bin/env python3

import subprocess
import json
import os
import time

def main():
    print("🚀 执行GitHub代码推送")
    print("=" * 50)

    github_username = "HC20251027"
    github_token = os.environ.get("GITHUB_TOKEN", "")
    repo_name = "norma-agent"

    # 切换到workspace目录
    os.chdir("/workspace")

    try:
        # 1. 检查Git状态
        print("检查Git状态...")
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=30)
        print(f"Git状态: {len(result.stdout.splitlines())} 个变更文件")
        
        # 2. 重新提交所有文件
        print("重新提交所有文件...")
        result = subprocess.run(["git", "add", "."], capture_output=True, text=True, timeout=30)
        print(f"Git add: {result.returncode}")
        
        result = subprocess.run(["git", "commit", "-m", "Norma Agent - 完整项目代码上传"], capture_output=True, text=True, timeout=30)
        print(f"Git commit: {result.returncode}")
        
        # 3. 设置远程仓库
        print("设置远程仓库...")
        result = subprocess.run(["git", "remote", "remove", "origin"], capture_output=True, text=True, timeout=10)
        print(f"移除旧远程: {result.returncode}")
        
        result = subprocess.run(["git", "remote", "add", "origin", f"https://{github_token}@github.com/{github_username}/{repo_name}.git"], capture_output=True, text=True, timeout=10)
        print(f"添加远程: {result.returncode}")
        
        # 4. 分批推送 - 先推送小的提交
        print("执行推送...")
        
        # 使用更长的超时时间和详细输出
        result = subprocess.run([
            "git", "push", "-u", "origin", "main", "--force"
        ], capture_output=True, text=True, timeout=600)
        
        print(f"推送返回码: {result.returncode}")
        print(f"推送输出长度: {len(result.stdout)} 字符")
        print(f"推送错误长度: {len(result.stderr)} 字符")
        
        if result.returncode == 0:
            print("🎉 推送成功!")
            print(f"✅ 仓库地址: https://github.com/{github_username}/{repo_name}")
            print(f"✅ 访问链接: https://github.com/{github_username}/{repo_name}")
        else:
            print(f"❌ 推送失败:")
            if result.stderr:
                print(f"错误信息: {result.stderr}")
            if result.stdout:
                print(f"输出信息: {result.stdout}")
                
    except subprocess.TimeoutExpired:
        print("❌ 推送超时 - 文件可能过多")
        print("建议: 考虑分批上传或使用GitHub Desktop")
    except Exception as e:
        print(f"❌ 推送异常: {e}")

    print("\n" + "=" * 50)
    print("GitHub推送流程完成")
    print("=" * 50)

if __name__ == "__main__":
    main()