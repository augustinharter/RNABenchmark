from safetensors.torch import save_file
from safetensors import safe_open
import torch



tensors = {}
with safe_open("/pfss/mlde/workspaces/mlde_wsp_GenomicKB_KGs/ah/RNABenchmark/checkpoint/baseline/BEACON-B/model.safetensors.original", framework="pt", device="cpu") as f:
    for key in f.keys():
        tensors[key] =  torch.randn_like(f.get_tensor(key)) * 0.01
    save_file(tensors, "/pfss/mlde/workspaces/mlde_wsp_GenomicKB_KGs/ah/RNABenchmark/checkpoint/baseline/BEACON-B/model.safetensors", metadata=f.metadata())
    
print('Randomized the weights and saved to model.safetensors')