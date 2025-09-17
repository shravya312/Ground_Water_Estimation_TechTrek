#!/usr/bin/env python3
"""
Meta Tensor Fix Script for SentenceTransformer

This script helps resolve the "Cannot copy out of meta tensor" error
that occurs with newer versions of PyTorch and transformers.

Run this script before starting the Streamlit app if you encounter meta tensor issues.
"""

import os
import torch
import gc
from sentence_transformers import SentenceTransformer

def fix_meta_tensor_issue():
    """Fix meta tensor issues by pre-loading and fixing the model."""
    print("🔧 Fixing meta tensor issues...")
    
    # Set environment variables
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    
    try:
        # Clear any existing models
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("📥 Loading SentenceTransformer model...")
        
        # Method 1: Try with explicit CPU handling
        try:
            model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu",
                model_kwargs={
                    "torch_dtype": torch.float32,
                    "use_safetensors": False,
                    "low_cpu_mem_usage": False,
                },
            )
            
            # Fix meta tensors
            print("🔧 Fixing meta tensors...")
            for name, param in model.named_parameters():
                if hasattr(param, 'is_meta') and param.is_meta:
                    new_param = torch.nn.Parameter(torch.zeros_like(param, device='cpu'))
                    param.data = new_param.data
            
            # Test the model
            print("🧪 Testing model...")
            test_embedding = model.encode(["test"], show_progress_bar=False)
            
            if test_embedding is not None and len(test_embedding) > 0:
                print("✅ Model loaded and tested successfully!")
                print("✅ Meta tensor issue resolved!")
                return True
            else:
                raise Exception("Model encoding test failed")
                
        except Exception as e1:
            print(f"❌ Method 1 failed: {e1}")
            
            # Method 2: Try without device specification
            try:
                print("🔄 Trying alternative loading method...")
                model = SentenceTransformer("all-MiniLM-L6-v2")
                
                # Fix meta tensors
                for name, param in model.named_parameters():
                    if hasattr(param, 'is_meta') and param.is_meta:
                        new_param = torch.nn.Parameter(torch.zeros_like(param, device='cpu'))
                        param.data = new_param.data
                
                # Move to CPU
                model = model.to("cpu")
                
                # Test the model
                test_embedding = model.encode(["test"], show_progress_bar=False)
                
                if test_embedding is not None and len(test_embedding) > 0:
                    print("✅ Model loaded and tested successfully!")
                    print("✅ Meta tensor issue resolved!")
                    return True
                else:
                    raise Exception("Model encoding test failed")
                    
            except Exception as e2:
                print(f"❌ Method 2 failed: {e2}")
                print("❌ Could not resolve meta tensor issue automatically.")
                print("💡 The app will still work with BM25-only search.")
                return False
                
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Meta Tensor Fix Script for Groundwater Chatbot")
    print("=" * 50)
    
    success = fix_meta_tensor_issue()
    
    if success:
        print("\n🎉 Success! You can now run the Streamlit app:")
        print("   streamlit run app3.py")
    else:
        print("\n⚠️  Meta tensor issue could not be resolved automatically.")
        print("   The app will still work with BM25-only search.")
        print("   Run: streamlit run app3.py")
    
    print("\n" + "=" * 50)
