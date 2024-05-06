from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json

dataset = load_dataset("lytang/MeetingBank-transcript")["train"]
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
n = 0
data_json = []
for data in dataset:
    source = data["source"]
    summary = data["reference"]
    input_tokens = tokenizer.encode(source)
    output_tokens = tokenizer.encode(summary)
    total = len(input_tokens) + len(output_tokens)
    # print(total)
    if total < 32768:
        data_json.append({'idx': n, 'source':source, 'summary': summary})
        n+=1

with open("MeetingBank_32k.json",'w') as fh:
    json.dump(data_json, fh)