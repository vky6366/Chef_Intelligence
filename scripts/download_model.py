"""
TinyLlama Model Downloader
===========================
Pre-downloads the TinyLlama model to avoid slow first-time loading.

Usage:
    python scripts/download_model.py

This script should be run BEFORE starting Docker to cache the model locally.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def download_model_with_progress():
    """Download TinyLlama model with progress indication"""
    
    # Use the same cache directory as config.py
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    cache_dir = project_root / "models" / "cache"
    
    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🚀 TinyLlama Model Downloader for Chef Intelligence")
    print("=" * 70)
    print(f"📦 Model: {model_name}")
    print(f"💾 Cache Directory: {cache_dir}")
    print(f"📊 Expected Size: ~2.2 GB")
    print(f"🖥️  System: {sys.platform}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    print("=" * 70)
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(cache_dir.parent)
        free_gb = free // (2**30)
        print(f"💿 Free Disk Space: {free_gb} GB")
        
        if free_gb < 3:
            print("⚠️  WARNING: Low disk space! Need at least 3GB free.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("❌ Download cancelled.")
                return False
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
    
    print("\n" + "=" * 70)
    
    try:
        # Download tokenizer
        print("\n[Step 1/2] 📥 Downloading Tokenizer...")
        print("⏳ Please wait...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir)
        )
        print("✅ Tokenizer downloaded successfully!")
        print(f"   └─ Vocab size: {tokenizer.vocab_size}")
        
        # Download model
        print("\n[Step 2/2] 📥 Downloading Model...")
        print("⏳ This may take 5-20 minutes depending on your internet speed...")
        print("   Downloading 2.2GB of model weights...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        )
        
        print("✅ Model downloaded successfully!")
        
        # Get model info
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   └─ Parameters: {param_count:,} (~1.1B)")
        
        # Verify cache
        cached_files = list(cache_dir.rglob("*"))
        cache_size = sum(f.stat().st_size for f in cached_files if f.is_file())
        cache_size_mb = cache_size / (1024 * 1024)
        
        print(f"\n📊 Cache Statistics:")
        print(f"   └─ Total files: {len(cached_files)}")
        print(f"   └─ Cache size: {cache_size_mb:.1f} MB")
        print(f"   └─ Location: {cache_dir}")
        
        print("\n" + "=" * 70)
        print("🎉 SUCCESS! Model is ready to use.")
        print("=" * 70)
        print("\n📝 Next Steps:")
        print("   1. Update docker-compose.yml to mount the cache directory")
        print("   2. Run: docker-compose build")
        print("   3. Run: docker-compose up")
        print("\n💡 The model will now load instantly in Docker!")
        print("=" * 70)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user.")
        print("   You can resume by running this script again.")
        return False
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check your internet connection")
        print("   2. Ensure you have ~3GB free disk space")
        print("   3. Try: pip install --upgrade transformers torch")
        print("   4. Check HuggingFace status: https://status.huggingface.co/")
        print(f"   5. Error details: {type(e).__name__}")
        
        # Clean up partial downloads
        if cache_dir.exists():
            print("\n🧹 Cleaning up partial downloads...")
            try:
                import shutil
                for item in cache_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                print("   └─ Cleanup complete")
            except Exception as cleanup_err:
                print(f"   └─ Cleanup error: {cleanup_err}")
        
        return False


def check_existing_model():
    """Check if model is already downloaded"""
    cache_dir = project_root / "models" / "cache"
    
    if not cache_dir.exists():
        return False
    
    # Check for model files
    model_files = list(cache_dir.rglob("*.bin")) + list(cache_dir.rglob("*.safetensors"))
    
    if model_files:
        cache_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        cache_size_mb = cache_size / (1024 * 1024)
        
        print("=" * 70)
        print("✅ Model Already Downloaded!")
        print("=" * 70)
        print(f"📁 Location: {cache_dir}")
        print(f"💾 Size: {cache_size_mb:.1f} MB")
        print(f"📄 Model files: {len(model_files)}")
        print("=" * 70)
        
        response = input("\n⚠️  Re-download model? (y/n): ")
        return response.lower() != 'y'
    
    return False


if __name__ == "__main__":
    print("\n")
    
    # Check if model already exists
    if check_existing_model():
        print("👍 Using existing model cache. Exiting...")
        sys.exit(0)
    
    # Download model
    success = download_model_with_progress()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)