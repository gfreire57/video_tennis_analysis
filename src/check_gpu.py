"""
GPU Detection and Capability Check Script
Run this script to verify if your system can use GPU acceleration for TensorFlow training
"""

import sys

def check_gpu():
    """Comprehensive GPU check for TensorFlow"""

    print("=" * 70)
    print("TensorFlow GPU Detection and Capability Check")
    print("=" * 70)
    print()

    # Step 1: Check TensorFlow installation
    print("Step 1: Checking TensorFlow installation...")
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow installed: version {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow not installed!")
        print("   Run: poetry install")
        return False

    print()

    # Step 2: Check CUDA support
    print("Step 2: Checking CUDA support...")
    is_built_with_cuda = tf.test.is_built_with_cuda()

    if is_built_with_cuda:
        print("✅ TensorFlow is built with CUDA support")
    else:
        print("⚠️  TensorFlow is NOT built with CUDA support")
        print("   This means you're using the CPU-only version")
        print("   To enable GPU, you need to install tensorflow[and-cuda]")

    print()

    # Step 3: Check for GPU devices
    print("Step 3: Detecting GPU devices...")
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f"✅ Found {len(gpus)} GPU device(s):")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")

            # Get GPU details if possible
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                if gpu_details:
                    print(f"      Details: {gpu_details}")
            except:
                pass
    else:
        print("❌ No GPU devices found")

        if is_built_with_cuda:
            print("   Possible reasons:")
            print("   - No NVIDIA GPU in your system")
            print("   - GPU drivers not installed")
            print("   - CUDA/cuDNN not installed or not compatible")
        else:
            print("   You're using CPU-only TensorFlow")

    print()

    # Step 4: Test GPU availability
    print("Step 4: Testing GPU availability...")
    gpu_available = tf.test.is_gpu_available(cuda_only=False)

    if gpu_available:
        print("✅ GPU is available for TensorFlow operations")
    else:
        print("⚠️  GPU is NOT available for TensorFlow operations")

    print()

    # Step 5: Check CUDA and cuDNN versions
    print("Step 5: Checking CUDA and cuDNN...")

    try:
        # Try to get build info
        print(f"   Built with CUDA: {tf.test.is_built_with_cuda()}")

        # Check for CUDA availability
        if hasattr(tf.sysconfig, 'get_build_info'):
            build_info = tf.sysconfig.get_build_info()
            if 'cuda_version' in build_info:
                print(f"   CUDA version: {build_info['cuda_version']}")
            if 'cudnn_version' in build_info:
                print(f"   cuDNN version: {build_info['cudnn_version']}")
    except Exception as e:
        print(f"   Could not retrieve CUDA/cuDNN info: {e}")

    print()

    # Step 6: Run a simple GPU test
    if gpus:
        print("Step 6: Running GPU computation test...")
        try:
            with tf.device('/GPU:0'):
                # Create random tensors
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])

                # Perform matrix multiplication
                c = tf.matmul(a, b)

                print("✅ GPU computation test PASSED")
                print(f"   Successfully performed matrix multiplication on GPU")
                print(f"   Result shape: {c.shape}")
        except Exception as e:
            print(f"❌ GPU computation test FAILED: {e}")
    else:
        print("Step 6: Skipping GPU test (no GPU available)")

    print()

    # Step 7: Memory configuration test
    if gpus:
        print("Step 7: Testing GPU memory configuration...")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Successfully enabled memory growth (dynamic allocation)")
        except RuntimeError as e:
            print(f"⚠️  Could not configure GPU memory: {e}")
            print("   This is usually fine, memory will be allocated statically")

    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    if gpus and is_built_with_cuda:
        print("✅ GPU TRAINING READY!")
        print(f"   You have {len(gpus)} GPU(s) available for training")
        print("   TensorFlow will automatically use GPU acceleration")
        print()
        print("Recommendations:")
        print("   - Enable mixed precision for faster training:")
        print("     Set 'use_mixed_precision': True in train_model.py CONFIG")
        print("   - Monitor GPU usage during training with: nvidia-smi")

    elif not is_built_with_cuda and gpus:
        print("⚠️  GPU DETECTED BUT NOT USABLE")
        print("   You have GPU hardware but TensorFlow is CPU-only")
        print()
        print("To enable GPU support:")
        print("   1. Remove current TensorFlow:")
        print("      poetry remove tensorflow")
        print("   2. Install GPU-enabled version:")
        print("      poetry add tensorflow[and-cuda]>=2.17.0,<2.19.0")
        print("   3. Run poetry install")
        print("   4. Run this script again to verify")

    else:
        print("ℹ️  CPU-ONLY TRAINING")
        print("   No GPU available - training will use CPU")
        print()
        print("This is fine for:")
        print("   - Small datasets")
        print("   - Testing and debugging")
        print("   - Systems without NVIDIA GPU")
        print()
        print("CPU training will be slower but will work correctly")

    print()
    print("=" * 70)

    return len(gpus) > 0


def check_nvidia_gpu():
    """Check for NVIDIA GPU using nvidia-smi"""
    print("\n" + "=" * 70)
    print("NVIDIA GPU Check (nvidia-smi)")
    print("=" * 70)
    print()

    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'],
                              capture_output=True,
                              text=True,
                              timeout=5)

        if result.returncode == 0:
            print("✅ NVIDIA GPU detected:")
            print()
            print(result.stdout)
        else:
            print("⚠️  nvidia-smi command failed")
            print("   No NVIDIA GPU detected or drivers not installed")
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found")
        print("   This usually means:")
        print("   - No NVIDIA GPU in your system, OR")
        print("   - NVIDIA drivers are not installed")
    except subprocess.TimeoutExpired:
        print("⚠️  nvidia-smi command timed out")
    except Exception as e:
        print(f"⚠️  Error checking for NVIDIA GPU: {e}")

    print("=" * 70)


if __name__ == "__main__":
    # Check for NVIDIA GPU first
    check_nvidia_gpu()

    print("\n")

    # Check TensorFlow GPU support
    gpu_ready = check_gpu()

    # Exit code: 0 if GPU ready, 1 if not
    sys.exit(0 if gpu_ready else 1)
