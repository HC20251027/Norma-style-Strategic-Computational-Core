#!/usr/bin/env python3
"""
诺玛AI系统后端部署验证脚本
用于验证部署配置的正确性
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_file_exists(file_path):
    """检查文件是否存在"""
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        return False
    print(f"✅ 文件存在: {file_path}")
    return True

def check_requirements():
    """检查requirements.txt"""
    requirements_file = "requirements.txt"
    if not check_file_exists(requirements_file):
        return False
    
    try:
        with open(requirements_file, 'r', encoding='utf-8') as f:
            content = f.read()
            required_packages = ['fastapi', 'uvicorn', 'sqlalchemy']
            missing_packages = []
            
            for package in required_packages:
                if package not in content:
                    missing_packages.append(package)
            
            if missing_packages:
                print(f"❌ requirements.txt中缺少必要的包: {missing_packages}")
                return False
            
            print("✅ requirements.txt检查通过")
            return True
    except Exception as e:
        print(f"❌ 读取requirements.txt失败: {e}")
        return False

def check_procfile():
    """检查Procfile"""
    procfile = "Procfile"
    if not check_file_exists(procfile):
        return False
    
    try:
        with open(procfile, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'uvicorn' not in content:
                print("❌ Procfile中缺少uvicorn启动命令")
                return False
            print("✅ Procfile检查通过")
            return True
    except Exception as e:
        print(f"❌ 读取Procfile失败: {e}")
        return False

def check_railway_json():
    """检查railway.json"""
    railway_file = "railway.json"
    if not check_file_exists(railway_file):
        return False
    
    try:
        with open(railway_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            required_keys = ['build', 'deploy']
            for key in required_keys:
                if key not in data:
                    print(f"❌ railway.json中缺少必要的键: {key}")
                    return False
            print("✅ railway.json检查通过")
            return True
    except json.JSONDecodeError as e:
        print(f"❌ railway.json格式错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 读取railway.json失败: {e}")
        return False

def check_render_yaml():
    """检查render.yaml"""
    render_file = "render.yaml"
    if not check_file_exists(render_file):
        return False
    
    try:
        with open(render_file, 'r', encoding='utf-8') as f:
            content = f.read()
            required_elements = ['services:', 'startCommand:']
            missing_elements = []
            
            for element in required_elements:
                if element not in content:
                    missing_elements.append(element)
            
            if missing_elements:
                print(f"❌ render.yaml中缺少必要元素: {missing_elements}")
                return False
            
            print("✅ render.yaml检查通过")
            return True
    except Exception as e:
        print(f"❌ 读取render.yaml失败: {e}")
        return False

def check_dockerfile():
    """检查Dockerfile"""
    dockerfile = "Dockerfile"
    if not check_file_exists(dockerfile):
        return False
    
    try:
        with open(dockerfile, 'r', encoding='utf-8') as f:
            content = f.read()
            required_elements = ['FROM python:', 'WORKDIR', 'CMD']
            missing_elements = []
            
            for element in required_elements:
                if element not in content:
                    missing_elements.append(element)
            
            if missing_elements:
                print(f"❌ Dockerfile中缺少必要元素: {missing_elements}")
                return False
            
            print("✅ Dockerfile检查通过")
            return True
    except Exception as e:
        print(f"❌ 读取Dockerfile失败: {e}")
        return False

def check_docker_compose():
    """检查docker-compose.yml"""
    compose_file = "docker-compose.yml"
    if not check_file_exists(compose_file):
        return False
    
    try:
        with open(compose_file, 'r', encoding='utf-8') as f:
            content = f.read()
            required_elements = ['version:', 'services:', 'norma-ai-backend:']
            missing_elements = []
            
            for element in required_elements:
                if element not in content:
                    missing_elements.append(element)
            
            if missing_elements:
                print(f"❌ docker-compose.yml中缺少必要元素: {missing_elements}")
                return False
            
            print("✅ docker-compose.yml检查通过")
            return True
    except Exception as e:
        print(f"❌ 读取docker-compose.yml失败: {e}")
        return False

def test_docker_build():
    """测试Docker构建"""
    print("\n🔧 测试Docker构建...")
    try:
        result = subprocess.run(
            ["docker", "build", "-t", "norma-ai-backend-test", "."],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("✅ Docker构建成功")
            # 清理测试镜像
            subprocess.run(["docker", "rmi", "norma-ai-backend-test"], 
                         capture_output=True)
            return True
        else:
            print(f"❌ Docker构建失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Docker构建超时")
        return False
    except FileNotFoundError:
        print("⚠️  Docker未安装，跳过Docker构建测试")
        return True
    except Exception as e:
        print(f"❌ Docker构建测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 诺玛AI系统后端部署配置验证")
    print("=" * 50)
    
    # 切换到deploy目录
    deploy_dir = Path(__file__).parent
    os.chdir(deploy_dir)
    
    checks = [
        ("Requirements.txt", check_requirements),
        ("Procfile", check_procfile),
        ("Railway.json", check_railway_json),
        ("Render.yaml", check_render_yaml),
        ("Dockerfile", check_dockerfile),
        ("Docker Compose", check_docker_compose),
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        print(f"\n📋 检查 {name}:")
        if check_func():
            passed += 1
    
    # 测试Docker构建
    print(f"\n🔧 Docker构建测试:")
    if test_docker_build():
        passed += 1
    total += 1
    
    print("\n" + "=" * 50)
    print(f"📊 验证结果: {passed}/{total} 项检查通过")
    
    if passed == total:
        print("🎉 所有部署配置验证通过！")
        print("\n📖 部署指南:")
        print("- Heroku: 使用Procfile")
        print("- Railway: 使用railway.json")
        print("- Render: 使用render.yaml")
        print("- Docker: 使用Dockerfile")
        print("- 本地开发: 使用docker-compose.yml")
    else:
        print("⚠️  部分配置存在问题，请检查上述错误信息")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)