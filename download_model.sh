mkdir ./partialsketchcontrolnet
cd ./partialsketchcontrolnet
wget -O config.json https://huggingface.co/datasets/miamia333/BlockAndDetail/resolve/main/config.json?download=true
wget -O diffusion_pytorch_model.safetensors https://huggingface.co/datasets/miamia333/BlockAndDetail/resolve/main/diffusion_pytorch_model.safetensors?download=true
echo "Finished downloading model!"