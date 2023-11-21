import jenkspy
import pandas as pd
score = pd.read_csv("/home/ubuntu/shrink_kv/results/llama2_70b_atten_w/seq_75_layer_79.csv", index_col=0)
score = score.values.transpose()[0][:71]
top = score.argsort()
print(top)
selected = sorted(top[39:])
print(selected)
print(len(selected))