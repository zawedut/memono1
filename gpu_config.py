"""
GPU Configuration for MEMO-BOT
Forces all AI services to use CUDA GPU
"""
import os
import sys

def setup_gpu():
    """Configure GPU settings for all AI frameworks"""
    
    # ============= PYTORCH GPU (MUST BE IMPORTED FIRST) =============
    # Fix for WinError 127: PyTorch must load its DLLs before TensorFlow on Windows
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"🎮 PyTorch CUDA GPU(s) detected: {device_count}")
            for i in range(device_count):
                print(f"   - [{i}] {torch.cuda.get_device_name(i)}")
            
            # Set default device to GPU
            torch.set_default_device('cuda')
            
            # Enable cuDNN benchmark for faster convolutions
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # Use TensorFloat-32 for faster computation on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            print("✅ PyTorch configured to use CUDA GPU")
        else:
            print("⚠️ PyTorch: CUDA not available, using CPU")
            
    except Exception as e:
        print(f"⚠️ PyTorch GPU setup error: {e}")

    # ============= TENSORFLOW GPU =============
    # Set TensorFlow to use GPU
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
    
    try:
        import tensorflow as tf
        
        # List GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🎮 TensorFlow GPU(s) detected: {len(gpus)}")
            for gpu in gpus:
                print(f"   - {gpu.name}")
                # Enable memory growth to avoid OOM
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Force visible devices
            tf.config.set_visible_devices(gpus, 'GPU')
            print("✅ TensorFlow configured to use GPU")
        else:
            print("⚠️ TensorFlow: No GPU found, using CPU")
            
    except Exception as e:
        print(f"⚠️ TensorFlow GPU setup error: {e}")
    
    return get_device()

def get_device():
    """Get the best available device"""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except:
        pass
    return 'cpu'

def check_gpu_status():
    """Print detailed GPU status"""
    print("\n" + "="*50)
    print("🔍 GPU STATUS CHECK")
    print("="*50)
    
    # PyTorch (FIRST)
    try:
        import torch
        print(f"\n📦 PyTorch Version: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            print(f"   Current GPU: {torch.cuda.current_device()}")
            print(f"   GPU Name: {torch.cuda.get_device_name()}")
            
            # Memory info
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU Memory: {gpu_mem:.1f} GB")
    except Exception as e:
        print(f"   PyTorch Error: {e}")
    
    # TensorFlow
    try:
        import tensorflow as tf
        print(f"\n📦 TensorFlow Version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"   GPU Count: {len(gpus)}")
        for gpu in gpus:
            print(f"   - {gpu}")
    except Exception as e:
        print(f"   TensorFlow Error: {e}")
    
    print("="*50 + "\n")

# Auto-setup when module is imported
if __name__ != "__main__":
    # Only setup if imported, not if run directly
    pass
