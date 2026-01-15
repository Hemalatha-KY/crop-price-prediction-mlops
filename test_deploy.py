#!/usr/bin/env python3
"""
Test script to trigger GitHub Actions auto-deploy
"""

import subprocess
import sys
from datetime import datetime

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

def main():
    """Make a small change to trigger GitHub Actions."""
    print("🚀 Testing GitHub Actions Auto-Deploy")
    print("=" * 40)
    
    # Create a small change to trigger deployment
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Update README with deployment timestamp
    with open("README.md", "a") as f:
        f.write(f"\n\n## 🚀 Deployment Test\nLast deployment test: {timestamp}\n")
    
    print("📝 Added deployment timestamp to README.md")
    
    # Commit and push changes
    if run_command("git add README.md", "Staging changes"):
        if run_command(f'git commit -m "Test auto-deploy {timestamp}"', "Committing changes"):
            if run_command("git push origin main", "Pushing to GitHub"):
                print("\n✅ Changes pushed to GitHub!")
                print("🔥 GitHub Actions will now:")
                print("   1. Run tests")
                print("   2. Train models")
                print("   3. Build Docker image")
                print("   4. Deploy to Render automatically")
                print("\n📱 Monitor deployment at:")
                print("   - GitHub: https://github.com/YOUR_USERNAME/crop-price-prediction-mlops/actions")
                print("   - Render: https://dashboard.render.com")
                print("\n🌐 Your app will be available at:")
                print("   - https://crop-price-api.onrender.com")
            else:
                print("❌ Failed to push to GitHub")
        else:
            print("❌ Failed to commit changes")
    else:
        print("❌ Failed to stage changes")

if __name__ == "__main__":
    main()
