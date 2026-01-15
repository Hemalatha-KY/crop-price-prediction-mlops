#!/usr/bin/env python3
"""
Monitor GitHub Actions deployment progress
"""

import webbrowser
import time

def open_monitoring_urls():
    """Open monitoring URLs in browser."""
    urls = [
        ("GitHub Actions", "https://github.com/Hemalatha-KY/crop-price-prediction-mlops/actions"),
        ("Render Dashboard", "https://dashboard.render.com"),
        ("Test API", "https://crop-price-api.onrender.com/health")
    ]
    
    print("🔍 Opening monitoring URLs...")
    for name, url in urls:
        print(f"📱 {name}: {url}")
        try:
            webbrowser.open(url)
            time.sleep(1)  # Small delay between opening tabs
        except:
            print(f"❌ Could not open {name}")

def main():
    print("🚀 GitHub Actions Deployment Monitor")
    print("=" * 40)
    
    print("\n📋 What to Monitor:")
    print("1. GitHub Actions - Watch the pipeline run")
    print("2. Render Dashboard - Check deployment status")
    print("3. API Health - Test the live application")
    
    print("\n⏱️ Expected Timeline:")
    print("- Tests: 2-3 minutes")
    print("- Model Training: 5-10 minutes")
    print("- Docker Build: 3-5 minutes")
    print("- Render Deployment: 2-5 minutes")
    print("- Total: ~15-20 minutes")
    
    print("\n🎯 Success Indicators:")
    print("✅ All GitHub Actions jobs pass (green checkmarks)")
    print("✅ Render service shows 'Live' status")
    print("✅ API health check returns 200 OK")
    
    print("\n🔗 Your Final App URLs:")
    print("- Main App: https://crop-price-api.onrender.com")
    print("- API Docs: https://crop-price-api.onrender.com")
    print("- Health: https://crop-price-api.onrender.com/health")
    print("- Predict: https://crop-price-api.onrender.com/predict")
    
    # Open monitoring URLs
    open_monitoring_urls()
    
    print("\n🎉 Deployment fixes pushed! Monitor the progress above.")

if __name__ == "__main__":
    main()
