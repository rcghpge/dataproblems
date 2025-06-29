# Bash commands + Timer runs

# CPU-only runs
# For TensorFlow (before TensorFlow 2.x, or if you want to be explicit)
# This environment variable restricts TensorFlow to use only the CPU.
CUDA_VISIBLE_DEVICES="" python your_ml_script.py

# For other specialized frameworks check for  CLI flags --use-cpu
# For PyTorch (often no specific env var needed, but can disable CUDA)
# Your PyTorch script might have logic like:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# To force CPU, you can either set CUDA_VISIBLE_DEVICES=""
# or modify the script to always use "cpu".

# General CPU-bound program (no special flags needed)
time ./your_cpu_program


# GPU-only runs
# This is often the default if a GPU is present and configured correctly
time python your_ml_script.py

# Explicitly select a specific GPU (e.g., GPU 0) if you have multiple
CUDA_VISIBLE_DEVICES="0" time python your_ml_script.py

# Select multiple GPUs (e.g., GPU 0 and GPU 1)
CUDA_VISIBLE_DEVICES="0,1" time python your_ml_script.py


# CPU + GPU
# If your application is built for CPU+GPU parallelization, you just run it normally:
time python your_hybrid_ml_script.py
time ./your_hybrid_compute_program
