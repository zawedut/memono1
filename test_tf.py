import sys
try:
    import tensorflow as tf
    print(f"TensorFlow Version: {tf.__version__}")
    from tensorflow.keras.models import Sequential
    print("Import tensorflow.keras.models.Sequential successful")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
