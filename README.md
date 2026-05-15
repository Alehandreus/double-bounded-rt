Get code with all dependencies
```
git clone --recurse-submodules https://github.com/Alehandreus/double-bounded-rt
```

TRAINER

Create Python venv and launch it

Install Torch with CUDA support (non CUDA versions are not supported)

```
python -m pip install --upgrade pip   #if your default pip version is quite old (torch may require newer for installation)
pip install torch --index-url https://download.pytorch.org/whl/cu<your CUDA version>
```

Install other python packages
```
pip install numpy tqdm pillow tensorboard
```

Install mesh-utils
Change in CMakeLists.txt with your CC and (b) run (or link to their docs)
```
cd trainer/mesh-utils
pip install .
```
Install tiny-cuda-nn (refer to their docs for more details)
```
cd renderer/ext/tiny-cuda-nn/bindings/torch
python setup.py install
```

RENDERER

flip-cuda/cudaImage.cuh add vector to includes

CMakeLists.txt 130 add path to your optix installation 

## Build

```bash
cmake -B build
cmake --build build --target viewer evaluate compare_images
```

With OptiX hardware ray tracing:

```bash
cmake -B build -DUSE_OPTIX=ON -DOPTIX_PTX_ARCH=compute_89
cmake --build build --target viewer
```