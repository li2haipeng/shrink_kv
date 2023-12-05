from evaluate import load
import json
from difflib import SequenceMatcher
from collections import OrderedDict
import numpy as np

wer = load("wer")
bertscore = load("bertscore")
outputs = OrderedDict()
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
f_a = open('/home/ubuntu/shrink_kv/hf_Llama-2-13b-hf.json')
# f_a = open('/home/ubuntu/shrink_kv/hf_128_llama2-70b-hf.json')
f_b = open('/home/ubuntu/shrink_kv/mode_3_128_budget_0.8_recent_0.2_Llama-2-13b-hf.json')

ref_data = json.load(f_a)
test_data = json.load(f_b)

assert len(ref_data) == len(test_data)

similarity = []
wer_scores = []
bert_precisions = []
bert_recalls = []
bert_f1 = []
n = 0
idx = []
prompts = []

for ref, test in zip(ref_data.items(), test_data.items()):
    assert len(ref[0]) == len(test[0])
    s = similar(ref[1], test[1])
    similarity.append(s)
    # wer_score = wer.compute(predictions=[test[1]], references=[ref[1]])
    # wer_scores.append(wer_score)

    bert_score = bertscore.compute(predictions=[ref[1]], references=[test[1]], lang="en")
    bert_precisions.append(bert_score["precision"][0])
    bert_recalls.append(bert_score["recall"][0])
    bert_f1.append(bert_score["f1"][0])
    wer_score = wer.compute(predictions=[test[1]], references=[ref[1]])
    wer_scores.append(wer_score)
    if s < 1:
    #     print(f"========={s}, {bert_score['f1'][0]}===========")
    #     print(ref[0])
    #     print("-------------------")
    #     print(ref[1])
    #     print("..............")
    #     print(test[1])
        n+=1

print("similar rate: ",1 - n / len(similarity))
print("similarity: {0:.6f}".format(np.mean(similarity)))
# print(wer_scores)
print("wer_score: {0:.6f}".format(np.mean(wer_scores)))
print(f"bert_precision: {np.mean(bert_precisions):.6f}, bert_recall: {np.mean(bert_recalls):.6f}, bert_f1: {np.mean(bert_f1):.6f}")