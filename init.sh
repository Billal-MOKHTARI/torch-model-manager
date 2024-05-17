pip install graphviz
pip install torchview
pip install torchviz
pip install -e git+https://github.com/frgfm/torch-cam.git#egg=torchcam
pip install wandb
pip install colorama
pip install neptune
pip install neptune_pytorch
pip install ydata_profiling

# Install Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
pip install diffusers transformers accelerate scipy safetensors
# Install GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
cd ..

