
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import numpy as np
import os
from ipdb import set_trace

IMPORTANCE_CHECK_ON = True

IMPORTANCE_SAVE_DIR = "D:\\workspace\\cuhk\\term3\\cv\\SegVit\\llm_drop\\saved_importance"

SKIP_LAYERS = set(
    [
        # 20, 21 , 22
        14,16,18,20
    ]
)

module_import_collector:dict = {

}

def record_importance(
    module_tag,
    importance:float
):
    if module_tag not in module_import_collector.keys():
        module_import_collector[module_tag] = [importance]
    else:
        module_import_collector[module_tag].append(importance)

def compute_importance(
    input:torch.Tensor,
    output:torch.Tensor,
    metric:str = "cosine"
):
    if metric != "cosine":
        raise NotImplementedError("currently only cosine similarity is supported")
    assert input.shape == output.shape

    sim = F.cosine_similarity(input, output, dim=-1).mean().detach().cpu().item()
    return 1 - sim

def save_important(tag:str=None,saved_dir:str=IMPORTANCE_SAVE_DIR):
    global module_import_collector 
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    if tag is None:
        tag = time.strftime("%Y-%m-%d-%H-%M-%S")
    saved_f_path = os.path.join(saved_dir, f"{tag}.json")

    for idx, importances in module_import_collector.items():
        module_import_collector[idx] = np.mean(importances)
    
    with open(saved_f_path, "w") as jf:
        json.dump(module_import_collector, jf, indent=4)
    
    print("layer importance saved")