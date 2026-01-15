#!/usr/bin/env python3
"""
Setup script for GitHub Actions Auto-Deploy to Render
"""

import subprocess
import sys
import webbrowser
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def open_urls():
    """Open required URLs in browser."""
    urls = [
        ("GitHub", "https://github.com/new"),
        ("Render", "https://render.com"),
        ("Docker Hub", "https://hub.docker.com/settings/security")
    ]
    
    print("\n🌐 Opening setup pages in your browser...")
    for name, url in urls:
        print(f"📱 Opening {name}: {url}")
        webbrowser.open(url)

def check_git_status():
    """Check if git is properly configured."""
    print("\n🔍 Checking Git configuration...")
    
    # Check if git is initialized
    if not Path(".git").exists():
        print("❌ Git not initialized. Please run 'git init' first.")
        return False
    
    # Check if there are commits
    result = run_command("git log --oneline -1", "Checking git history")
    if not result:
        print("❌ No commits found. Please commit your changes first.")
        return False
    
    # Check remote
    remote_result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True)
    if "origin" not in remote_result.stdout:
        print("⚠️ No remote 'origin' found. You'll need to add one.")
    
    return True

def generate_github_commands():
    """Generate git commands for user to run."""
    commands = """
# Git Commands to Run (replace YOUR_USERNAME with your GitHub username):

git remote add origin https://github.com/YOUR_USERNAME/crop-price-prediction-mlops.git
git branch -M main
git push -u origin main
"""
    return commands

def main():
    """Main setup function."""
    print("🚀 GitHub Actions Auto-Deploy Setup for Render")
    print("=" * 50)
    
    # Check prerequisites
    if not check_git_status():
        print("\n❌ Please fix the git issues before continuing.")
        return
    
    print("\n📋 SETUP STEPS:")
    print("1. Create GitHub Repository")
    print("2. Set up Render Web Service")
    print("3. Configure GitHub Secrets")
    print("4. Test Auto-Deploy")
    
    # Open required websites
    open_urls()
    
    # Generate git commands
    print("\n📝 Git Commands to Run:")
    print(generate_github_commands())
    
    print("\n🔐 Required GitHub Secrets:")
    print("1. RENDER_DEPLOY_WEBHOOK: Get from Render service settings")
    print("2. DOCKER_USERNAME: Your Docker Hub username")
    print("3. DOCKER_PASSWORD: Docker Hub access token")
    
    print("\n📖 Detailed Instructions:")
    print("=" * 30)
    
    instructions = """
STEP 1: CREATE GITHUB REPOSITORY
1. Go to GitHub (opened in browser)
2. Click '+' → 'New repository'
3. Name: 'crop-price-prediction-mlops'
4. Make it PUBLIC (for free Render deployment)
5. Click 'Create repository'

STEP 2: SET UP RENDER WEB SERVICE
1. Go to Render.com (opened in browser)
2. Click 'New +' → 'Web Service'
3. Connect GitHub account
4. Select your repository
5. Configure:
   - Name: crop-price-api
   - Runtime: Docker
   - Branch: main
   - Instance Type: Free
6. Click 'Create Web Service'

STEP 3: GET RENDER WEBHOOK
1. In Render dashboard → your service
2. Click 'Settings' tab
3. Find 'Webhook' section
4. Copy the webhook URL

STEP 4: SET UP GITHUB SECRETS
1. In GitHub → your repository → Settings
2. Click 'Secrets and variables' → 'Actions'
3. Add these secrets:
   - RENDER_DEPLOY_WEBHOOK: [paste webhook URL]
   - DOCKER_USERNAME: [your Docker Hub username]
   - DOCKER_PASSWORD: [your Docker Hub access token]

STEP 5: PUSH TO GITHUB
1. Run the git commands shown above
2. Watch GitHub Actions run automatically
3. Your app will deploy to Render!

STEP 6: ACCESS YOUR APP
Your app will be available at:
https://crop-price-api.onrender.com

API Endpoints:
- https://crop-price-api.onrender.com/
- https://crop-price-api.onrender.com/health
- https://crop-price-api.onrender.com/predict
"""
    
    print(instructions)
    
    print("\n✅ Setup guide ready! Follow the steps above.")
    print("🌐 Required websites are opening in your browser.")
    
    # Ask user if they want to proceed
    response = input("\n❓ Have you created the GitHub repository? (y/n): ")
    if response.lower() == 'y':
        print("\n🎉 Great! Now follow the remaining steps in the setup guide.")
    else:
        print("\n📱 Please create the GitHub repository first, then run this script again.")

if __name__ == "__main__":
    main()
