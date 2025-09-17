# Troubleshooting Guide

## PyTorch Meta Tensor Issue

If you encounter the error: "Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device."

### Quick Fixes:

1. **Use the Fix Script** (Recommended):
   ```bash
   python fix_meta_tensors.py
   ```

2. **Update PyTorch and Transformers**:
   ```bash
   pip install --upgrade torch torchvision torchaudio
   pip install --upgrade transformers sentence-transformers
   ```

3. **Clear PyTorch Cache**:
   ```bash
   # Clear PyTorch cache
   python -c "import torch; torch.cuda.empty_cache()"
   
   # Clear sentence-transformers cache
   rm -rf ~/.cache/torch/sentence_transformers/
   ```

3. **Use the Retry Button**:
   - In the app, look for the "🔄 Retry Model Initialization" button in the sidebar
   - Click it to attempt reinitializing the model

4. **Alternative: Use BM25-Only Mode**:
   - The app will automatically fall back to BM25-only search if dense embeddings fail
   - This still provides good search results, just without the hybrid approach

### Environment Variables:

Add these to your environment or `.env` file:
```bash
CUDA_VISIBLE_DEVICES=""
TOKENIZERS_PARALLELISM=false
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

### Manual Model Reset:

If the issue persists, you can manually reset the model by restarting the Streamlit app:
```bash
# Stop the current app (Ctrl+C)
# Then restart
streamlit run app3.py
```

### System Requirements:

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- Sentence-Transformers 2.2+

### Known Issues:

1. **Meta Tensor Issue**: Common with newer PyTorch versions and certain model configurations
2. **Memory Issues**: If you have limited RAM, the model might fail to load
3. **CUDA Issues**: Make sure CUDA is properly configured if using GPU

### Fallback Behavior:

The app is designed to gracefully handle model initialization failures:
- ✅ **With Dense Embeddings**: Full hybrid search (dense + sparse)
- ⚠️ **Without Dense Embeddings**: BM25-only search (still effective)

Both modes support multilingual functionality and will provide good search results.
