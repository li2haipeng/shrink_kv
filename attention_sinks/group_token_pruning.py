from typing import Any
import numpy as np
import pandas as pd
from jenkspy import JenksNaturalBreaks
import time
from dataclasses import dataclass
import torch
from evaluate import load as eval_load


def selection_rules(candidates):
    num_labels = len(candidates)
    top = candidates[int(0.5 * num_labels):]
    middle = candidates[int(0.25 * num_labels): int(0.5 * num_labels)]
    bottom = candidates[: int(0.25 * num_labels)]
    return top, middle, bottom


def select_tokens_with_rules(labels, rule, budget)->list:
    sorted_labels = np.argsort(labels)[::-1]
    selected = list()
    if len(rule) == 3:
        top, middle, bottom = rule
        for idx in sorted_labels:
            l = labels[idx]
            if idx == 0 or idx == len(labels) - 1:
                selected.append(idx)
                labels[idx] = -1
                continue
            if l in top:
                selected.append(idx)
                labels[idx] = -1
            elif l in middle:
                if abs(labels[idx - 1] - l) <= 1 and abs(labels[idx + 1] - l) <= 1:
                    selected.extend([idx - 1, idx, idx + 1])
                    labels[idx - 1] = -1
                    labels[idx] = -1
                    labels[idx + 1] = -1

            elif l in bottom:
                if labels[idx - 1] == l and labels[idx + 1] == l:
                    selected.extend([idx - 1, idx, idx + 1])
                    labels[idx - 1] = -1
                    labels[idx] = -1
                    labels[idx + 1] = -1
            elif l == -1:
                continue

            if len(set(selected)) >= budget:
                break
    else:
        for idx in sorted_labels:
            l = labels[idx]
            if idx == 0 or idx == len(labels) - 1:
                selected.append(idx)
                labels[idx] = -1
                continue
            elif labels[idx - 1] != -1 and labels [idx + 1] != -1:
                selected.extend([idx - 1, idx, idx + 1])
                labels[idx - 1] = -1
                labels[idx] = -1
                labels[idx + 1] = -1
            elif l == -1:
                continue

            if len(set(selected)) >= budget:
                break
    return [selected, labels]


def duplicate(testList, n):
    return [ele for ele in testList for _ in range(n)]


def pruned_prompt_eval(prompt, pruned_promt):
    bertscore = eval_load("bertscore")
    bert_score = bertscore.compute(predictions=prompt, references=pruned_promt, lang="en")


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


def prompt_token_selection(attn_w, rate, recent):
    if type(attn_w) == str:
        d = pd.read_csv(w_path, index_col=0)
        attn_w = d.values.transpose()[0]

    groups = int(len(attn_w)/5)
    jnb = JenksNaturalBreaks(groups)
    # print(attn_w)
    jnb.fit(attn_w)
    labels = jnb.labels_.tolist()
    # print(labels)
    budget = int(len(attn_w) * rate)
    candidates = [i for i in range(groups)]
    top, middle, bottom = selection_rules(candidates)
    rule = [top, middle, bottom]

    recent = [i for i in range(len(attn_w))][-int(len(attn_w) * recent):]
    labels[recent[0]:] = [-1] * len(recent)
    selected = recent
    selected.extend([0,1,2,3])

    # leftover_budget = budget-len(selected)
    groupd_selected, labels = select_tokens_with_rules(labels, rule, budget-len(selected))
    selected.extend(groupd_selected)

    selected = list(set(selected))
    if len(selected) < budget:
        selected_leftover, labels = select_tokens_with_rules(labels, [], budget-len(selected))
        selected.extend(selected_leftover)
    selected = sorted(selected)
    # print(f"len of selected: {len(selected)}, selected: {selected}")
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