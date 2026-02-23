FROM registry.fedoraproject.org/fedora:43

# 1. System Base & Build Tools
# Added 'gperftools-libs' for tcmalloc (fixes double-free)
COPY scripts/install_deps.sh /tmp/install_deps.sh
RUN sh /tmp/install_deps.sh

# 2. Install "TheRock" ROCm SDK (Tarball Method)
WORKDIR /tmp
ARG ROCM_MAJOR_VER=7
ARG GFX=gfx1151
# We pass ARGs to the script via ENV or rely on defaults. 
# But let's be explicit and export them for the RUN command.
COPY scripts/install_rocm_sdk.sh /tmp/install_rocm_sdk.sh
RUN chmod +x /tmp/install_rocm_sdk.sh && \
  export ROCM_MAJOR_VER=$ROCM_MAJOR_VER && \
  export GFX=$GFX && \
  /tmp/install_rocm_sdk.sh

# 4. Python Venv Setup
RUN /usr/bin/python3.13 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PIP_NO_CACHE_DIR=1
RUN printf 'source /opt/venv/bin/activate\n' > /etc/profile.d/venv.sh
RUN python -m pip install --upgrade pip wheel packaging "setuptools<80.0.0"

# 5. Install PyTorch (TheRock Nightly)
RUN python -m pip install \
  --index-url https://rocm.nightlies.amd.com/v2-staging/gfx1151/ \
  --pre torch torchaudio torchvision

WORKDIR /opt

# Flash-Attention
ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"

RUN git clone https://github.com/ROCm/flash-attention.git &&\ 
  cd flash-attention &&\
  git checkout main_perf &&\
  python setup.py install && \
  cd /opt && rm -rf /opt/flash-attention

# 6. Clone vLLM
RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm
WORKDIR /opt/vllm

# --- PATCHING ---
RUN echo "import sys, re" > patch_strix.py && \
  echo "from pathlib import Path" >> patch_strix.py && \
  # Patch 1: __init__.py
  echo "p = Path('vllm/platforms/__init__.py')" >> patch_strix.py && \
  echo "txt = p.read_text()" >> patch_strix.py && \
  echo "txt = txt.replace('import amdsmi', '# import amdsmi')" >> patch_strix.py && \
  echo "txt = re.sub(r'is_rocm = .*', 'is_rocm = True', txt)" >> patch_strix.py && \
  echo "txt = re.sub(r'if len\(amdsmi\.amdsmi_get_processor_handles\(\)\) > 0:', 'if True:', txt)" >> patch_strix.py && \
  echo "txt = txt.replace('amdsmi.amdsmi_init()', 'pass')" >> patch_strix.py && \
  echo "txt = txt.replace('amdsmi.amdsmi_shut_down()', 'pass')" >> patch_strix.py && \
  echo "p.write_text(txt)" >> patch_strix.py && \
  # Patch 2: rocm.py
  echo "p = Path('vllm/platforms/rocm.py')" >> patch_strix.py && \
  echo "txt = p.read_text()" >> patch_strix.py && \
  echo "header = 'import sys\nfrom unittest.mock import MagicMock\nsys.modules[\"amdsmi\"] = MagicMock()\n'" >> patch_strix.py && \
  echo "txt = header + txt" >> patch_strix.py && \
  echo "txt = re.sub(r'device_type = .*', 'device_type = \"rocm\"', txt)" >> patch_strix.py && \
  echo "txt = re.sub(r'device_name = .*', 'device_name = \"gfx1151\"', txt)" >> patch_strix.py && \
  echo "txt += '\n    def get_device_name(self, device_id: int = 0) -> str:\n        return \"AMD-gfx1151\"\n'" >> patch_strix.py && \
  echo "p.write_text(txt)" >> patch_strix.py && \
  # Patch 3: Fix C10_HIP_CHECK undeclared identifier in selective_scan_fwd.hip
  echo "p_hip = Path('csrc/mamba/mamba_ssm/selective_scan_fwd.hip')" >> patch_strix.py && \
  echo "if p_hip.exists():" >> patch_strix.py && \
  echo "    txt_hip = p_hip.read_text()" >> patch_strix.py && \
  echo "    macro_def = '#ifndef C10_HIP_CHECK\n#include <c10/hip/HIPException.h>\n#ifndef C10_HIP_CHECK\n#define C10_HIP_CHECK(error) if (error != hipSuccess) { abort(); }\n#endif\n#endif\n'" >> patch_strix.py && \
  echo "    txt_hip = macro_def + txt_hip" >> patch_strix.py && \
  echo "    p_hip.write_text(txt_hip)" >> patch_strix.py && \
  # -----------
  echo "print('Successfully patched vLLM for Strix Halo')" >> patch_strix.py && \
  python patch_strix.py && \
  sed -i 's/gfx1200;gfx1201/gfx1151/' CMakeLists.txt  

# 7. Build vLLM (Wheel Method) with CLANG Host Compiler
RUN python -m pip install --upgrade cmake ninja packaging wheel numpy "setuptools-scm>=8" "setuptools<80.0.0" scikit-build-core pybind11
ENV ROCM_HOME="/opt/rocm"
ENV HIP_PATH="/opt/rocm"
ENV VLLM_TARGET_DEVICE="rocm"
ENV PYTORCH_ROCM_ARCH="gfx1151"
ENV HIP_ARCHITECTURES="gfx1151"          
ENV AMDGPU_TARGETS="gfx1151"              
ENV MAX_JOBS="4"

# --- CRITICAL FIX FOR SEGFAULT ---
# We force the Host Compiler (CC/CXX) to be the ROCm Clang, not Fedora GCC.
# This aligns the ABI of the compiled vLLM extensions with PyTorch.
ENV CC="/opt/rocm/llvm/bin/clang"
ENV CXX="/opt/rocm/llvm/bin/clang++"

RUN export HIP_DEVICE_LIB_PATH=$(find /opt/rocm -type d -name bitcode -print -quit) && \
  echo "Compiling with Bitcode: $HIP_DEVICE_LIB_PATH" && \
  export CMAKE_ARGS="-DROCM_PATH=/opt/rocm -DHIP_PATH=/opt/rocm -DAMDGPU_TARGETS=gfx1151 -DHIP_ARCHITECTURES=gfx1151" && \   
  python -m pip wheel --no-build-isolation --no-deps -w /tmp/dist -v . && \
  python -m pip install /tmp/dist/*.whl

RUN python -m pip install ray

# --- bitsandbytes (ROCm) ---
WORKDIR /opt
RUN git clone -b rocm_enabled_multi_backend https://github.com/ROCm/bitsandbytes.git
WORKDIR /opt/bitsandbytes

# Explicitly set HIP_PLATFORM (Docker ENV, not /etc/profile)
ENV HIP_PLATFORM="amd"
ENV CMAKE_PREFIX_PATH="/opt/rocm"

# Force CMake to use the System ROCm Compiler (/opt/rocm/llvm/bin/clang++)
RUN cmake -S . \
  -DGPU_TARGETS="gfx1151" \
  -DBNB_ROCM_ARCH="gfx1151" \
  -DCOMPUTE_BACKEND=hip \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  && \
  make -j$(nproc) && \
  python -m pip install --no-cache-dir . --no-build-isolation --no-deps

# 8. Final Cleanup & Runtime
WORKDIR /opt
RUN chmod -R a+rwX /opt && \
  find /opt/venv -type f -name "*.so" -exec strip -s {} + 2>/dev/null || true && \
  find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + && \
  rm -rf /root/.cache/pip || true && \
  dnf clean all && rm -rf /var/cache/dnf/*

COPY scripts/01-rocm-env-for-triton.sh /etc/profile.d/01-rocm-env-for-triton.sh
COPY scripts/99-toolbox-banner.sh /etc/profile.d/99-toolbox-banner.sh
COPY scripts/zz-venv-last.sh /etc/profile.d/zz-venv-last.sh
COPY scripts/start_vllm.py /opt/start-vllm
COPY scripts/start_vllm_cluster.py /opt/start-vllm-cluster
COPY scripts/measure_bandwidth.sh /opt/measure_bandwidth.sh
COPY scripts/cluster_manager.py /opt/cluster_manager.py
COPY scripts/models.py /opt/models.py

COPY benchmarks/max_context_results.json /opt/max_context_results.json
COPY benchmarks/run_vllm_bench.py /opt/run_vllm_bench.py
COPY benchmarks/vllm_cluster_bench.py /opt/vllm_cluster_bench.py
COPY benchmarks/find_max_context.py /opt/find_max_context.py
COPY rdma_cluster/compare_eth_vs_rdma.sh /opt/compare_eth_vs_rdma.sh
COPY scripts/configure_cluster.sh /opt/configure_cluster.sh
RUN chmod +x /opt/configure_cluster.sh

RUN chmod +x /opt/start-vllm /opt/start-vllm-cluster /opt/vllm_cluster_bench.py /opt/compare_eth_vs_rdma.sh /opt/find_max_context.py /opt/run_vllm_bench.py && \
  ln -s /opt/start-vllm /usr/local/bin/start-vllm && \
  ln -s /opt/start-vllm-cluster /usr/local/bin/start-vllm-cluster && \
  chmod 0644 /etc/profile.d/*.sh /opt/max_context_results.json /opt/models.py
RUN chmod 0644 /etc/profile.d/*.sh
RUN printf 'ulimit -S -c 0\n' > /etc/profile.d/90-nocoredump.sh && chmod 0644 /etc/profile.d/90-nocoredump.sh

# 9. Install Custom RCCL (gfx1151) - Replaces standard library with manually built one
COPY custom_libs/librccl.so.1.gz /tmp/librccl.so.1.gz
RUN echo "Installing Custom RCCL..." && \
  gzip -d /tmp/librccl.so.1.gz && \
  chmod 755 /tmp/librccl.so.1 && \
  # Replace /opt/rocm library strictly as managed_rccl_install.sh does
  cp -fv /tmp/librccl.so.1 /opt/rocm/lib/librccl.so.1.0 && \
  # Replace /opt/venv library
  find /opt/venv -name "librccl.so.1" -exec cp -fv /tmp/librccl.so.1 {} + && \
  rm /tmp/librccl.so.1

# 10. Force Upgrade Transformers (User Override)
# Required for GLM Flash. vLLM reports incompatibility with transformers >= 5, 
# but this version (5.0.0) has been tested and confirmed working.
RUN python -m pip install transformers==5.0.0

RUN chmod -R a+rwX /opt

CMD ["/bin/bash"]
