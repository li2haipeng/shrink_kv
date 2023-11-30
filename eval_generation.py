from transformers import pipeline, AutoTokenizer,GenerationConfig, CodeLlamaTokenizer, AutoModelForCausalLM
import torch
import time
import os
import subprocess
import pandas as pd
import sys
from datasets import load_dataset
sys.path.append("..")
import torch.distributed as dist
from attention_sinks.group_token_pruning import prompt_token_selection
from collections import OrderedDict
import json
# from token_pruning_llama import LlamaForCausalLM
from attention_sinks import LlamaForCausalLM as pruned_LlamaForCausalLM
from transformers import LlamaTokenizer
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

def nvidia_smi():
    if local_rank == 0:
        print()
        subprocess.check_call(['nvidia-smi'])
        print()


batch_size = 1
model_name = "NousResearch/Llama-2-13b-hf"
# model_name = "/home/ubuntu/hummingbird/data/llama2-70b-hf"
m = model_name.split("/")[-1]
print(f"loading {m}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
mode = "1"
kwargs = {
    "attention_sink_size": 4,
    "attention_sink_window_size": 252,  # default: 1020
    "attention_sink_mode": mode # -1: HF; 0: original attn sink; 1: sink recent; 2: our
}
model = pruned_LlamaForCausalLM.from_pretrained(
   model_name, 
   revision="main",
   torch_dtype=torch.float16,
   device_map="auto",
   **kwargs,
)
# position_ids = torch.arange(len(inputs[0]), dtype=torch.float16, device=model.device)
# outputs = model.model(
#             input_ids=inputs.to(model.device),
#             attention_mask=None,
#             position_ids=position_ids,
#             past_key_values=None,
#             inputs_embeds=None,
#             use_cache=True,
#             output_attentions=True,
#             output_hidden_states=False,
#             return_dict=True,
#         )
# attn_w = torch.sum(outputs.attentions[-1].squeeze(), dim=1)
# attn_w = torch.sum(attn_w, dim=0)
# df = pd.DataFrame(attn_w.detach().cpu().numpy())
# prompt_w_path = "/home/ubuntu/shrink_kv/results/seq_{}_layer.csv".format(inputs.size(1))
# df.to_csv(prompt_w_path)
# selected = prompt_token_selection(prompt_w_path, rate=0.5)
# inputs = inputs[:,selected]


model.eval()
model.config.output_attentions = True

pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Param: {:.2f}".format(pytorch_total_params/1000/1000))

if tokenizer.eos_token is None:
    # special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    tokenizer.add_special_tokens({'eos_token': DEFAULT_EOS_TOKEN})
if tokenizer.bos_token is None:
    # special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    tokenizer.add_special_tokens({'bos_token': DEFAULT_BOS_TOKEN})
if tokenizer.unk_token is None:
    # special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    tokenizer.add_special_tokens({'unk_token': DEFAULT_UNK_TOKEN})
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
model.resize_token_embeddings(len(tokenizer))


dataset = load_dataset('lambada', split='validation[:100]')
iterations = 1
max_tokens = 128
print("start...")

results = OrderedDict()
for inp in dataset:
    inp_text = inp['text']
    inputs = tokenizer(inp_text, return_tensors="pt")["input_ids"]
    print(f"Prompt: {inp_text}\n")

    t0 = time.time()
    # print(f"============{i} run==============")

    outputs = model.generate(
            input_ids=inputs.to(model.device),
            max_new_tokens=max_tokens, 
            pad_token_id=tokenizer.eos_token_id,
        )

    # nvidia_smi()
    generated_tokens = outputs
    t1 = time.time()
    # print(torch.cuda.memory_summary())

    # tokens_gen_text = len(generated_tokens[0]) - inputs.shape[1]
    response = tokenizer.decode(generated_tokens[0, inputs.shape[1]:])
    print(f"Response: {response}\n")
    results[inp_text] = response
    print(f"======={len(results), time.time()-t0}====================")

m = model_name.split("/")[-1]
with open(f"model_{mode}_{max_tokens}_{m}.json", 'w') as f:
    f.write(json.dumps(results))