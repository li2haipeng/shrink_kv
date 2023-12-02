from typing import Any
import numpy as np
import pandas as pd
from jenkspy import JenksNaturalBreaks
import time
from dataclasses import dataclass
import torch

def selection_rules(candidates):
    num_labels = len(candidates)
    top = candidates[int(0.5 * num_labels):]
    middle = candidates[int(0.25 * num_labels): int(0.5 * num_labels)]
    bottom = candidates[: int(0.25 * num_labels)]
    return top, middle, bottom


def duplicate(testList, n):
    return [ele for ele in testList for _ in range(n)]




@dataclass
class SinkRencent:
    attention_sink_size: int = 4
    attention_sink_window_size: int = 1020
    k_seq_dim: int = 2
    v_seq_dim: int = 2
    # prompt_len: int = 0

    def __call__(self, past_key_values) -> Any:
 
        pruned_next_cache = []
        for k, v in past_key_values:
            pruned_k = torch.cat([k[:,:,:self.attention_sink_size,:], k[:,:,self.attention_sink_size + 1:,:]], dim=2)
            pruned_v = torch.cat([v[:,:,:self.attention_sink_size,:], v[:,:,self.attention_sink_size + 1:,:]], dim=2)
            pruned_next_cache.append([pruned_k, pruned_v])
        # pruned_seq_len = pruned_next_cache[0][0].size(self.k_seq_dim)

        return pruned_next_cache


@dataclass
class UpdateKVCache:
    # min_idx: int
    # attn_w: torch.FloatTensor = None
    all_self_attns: torch.FloatTensor = None

    def __post_init__(self):
        self.attn_w = torch.sum(self.all_self_attns[-1], dim=(0,1,2), keepdim=True)
        self.min_idx = np.argmin(self.attn_w.squeeze().cpu().detach().numpy()) 
        self.min_idx = self.min_idx if self.min_idx >= 5 else 4

    def __call__(self, past_key_values):
        pruned_next_cache = []
        for k, v in past_key_values:
            pruned_k = torch.cat([k[:,:,:self.min_idx,:], k[:,:,self.min_idx+1:,:]], dim=2)
            pruned_v = torch.cat([v[:,:,:self.min_idx,:], v[:,:,self.min_idx+1:,:]], dim=2)
            pruned_next_cache.append([pruned_k, pruned_v])
            # print(k.size(), v.size())
        return pruned_next_cache


def prompt_token_selection(w_path, rate):
    d = pd.read_csv(w_path, index_col=0)
    data = d.values.transpose()[0]
    groups = int(len(data)/5)
    jnb = JenksNaturalBreaks(groups)
    jnb.fit(data)
    labels = jnb.labels_
    print(labels)
    budget = int(len(data) * rate)
    candidates = [i for i in range(groups)]
    top, middle, bottom = selection_rules(candidates)
    # print(f"selection rules: top: {top}, middle: {middle}, bottom: {bottom}")
    selected = list()
    sorted_labels = np.argsort(labels)[::-1]
    for idx in sorted_labels:
        l = labels[idx]

        if idx == 0 or idx == len(data) - 1:
            selected.append(idx)
            continue
        if l in top:
            selected.append(idx)
        elif l in middle:
            if abs(labels[idx - 1] - l) <= 1 and abs(labels[idx + 1] - l) <= 1:
                selected.extend([idx-1, idx, idx+1])
        elif l in bottom:
            if labels[idx - 1] == l and labels[idx + 1] == l:
                selected.extend([idx-1, idx, idx+1])
        else:
            raise ("something wrong")
        if len(set(selected)) >= budget:
            break
    recent = [i for i in range(len(data))][-int(len(data) * 0.1):]
    selected.extend(recent)
    selected = list(set(selected))
    print(f"len of seleted: {len(selected)}, selected: {selected}")
    return selected


def decoding_token_update(attn_w):

    pass
    return 

if __name__ == "__main__":
    w_path = "/home/ubuntu/shrink_kv/results/norm_llama2_70b_attn_w/seq_100_layer.csv"
    rate = 0.8
    tic = time.time()
    prompt_token_selection(w_path, rate)
    print(f"{time.time()-tic}")