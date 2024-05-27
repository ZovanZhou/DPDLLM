from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, choices=[32, 64, 128, 256, 512])
parser.add_argument("--func", type=str, choices=["original", "mintopk", "average", "minsubstr"], default="original")
parser.add_argument("--metric", type=str, choices=["max_prob", "cond_prob", "ppl", "lowercase_ppl", "zlib"], default="cond_prob")
parser.add_argument("--k", type=int, default=20)
parser.add_argument("--llm", type=str, choices=["llama-7b", "llama-13b", "pythia-2.8b"])
parser.add_argument("--dataset", type=str, choices=["wiki-spgc", "wikimia2", "wikimia2-spgc", "bookmia"])
args = parser.parse_args()

np.random.seed(0)

def plot_result(y, pred):
    prec = precision_score(y, pred)
    reca = recall_score(y, pred)
    auc = roc_auc_score(y, pred)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="weighted")
    print(f"p:{prec}\nr:{reca}\nf1:{f1}\nauc:{auc}\nacc:{acc}")
    return auc

def minsubstr(data, **kwargs):
    k = kwargs.get("k", 20)
    seq_prob, label = data
    k_seq_len = int(len(seq_prob) * (k / 100))
    min_sub_seq = []
    min_value = np.inf
    for i in range(len(seq_prob) // k_seq_len):
        tmp_min_value = np.mean(seq_prob[i*k_seq_len:(i+1)*k_seq_len])
        if tmp_min_value < min_value:
            min_value = tmp_min_value
            min_sub_seq = seq_prob[i*k_seq_len:(i+1)*k_seq_len]
    return [*min_sub_seq, label]

def mintopk(data, **kwargs):
    k = kwargs.get("k", 20)
    seq_prob, label = data
    k_seq_len = int(len(seq_prob) * (k / 100))
    seq_prob = np.sort(seq_prob)[:k_seq_len]
    return [np.mean(-np.log(seq_prob)), label]

def average(data, **kwargs):
    seq_prob, label = data
    return [np.mean(seq_prob), label]

def original(data, **kwargs):
    seq_prob, label = data
    return [*seq_prob, label]

def process_data(data, func: str = "average", **kwargs):
    process_func = None
    if func == "original":
        process_func = original
    elif func == "mintopk":
        process_func = mintopk
    elif func == "average":
        process_func = average
    elif func == "minsubstr":
        process_func = minsubstr
    parsed_data = []
    for d in data:
        # max_prob, conditional_prob, label = d
        ppl, lowercase_ppl, zlib, max_prob, cond_prob, label = d
        tmp = []
        if args.metric == "max_prob":
            tmp = [max_prob, label]
            parsed_data.append(process_func(tmp, **kwargs))
        elif args.metric == "cond_prob":
            tmp = [cond_prob, label]
            parsed_data.append(process_func(tmp, **kwargs))
        elif args.metric == "ppl":
            parsed_data.append([ppl, label])
        elif args.metric == "lowercase_ppl":
            parsed_data.append([lowercase_ppl, label])
        elif args.metric == "zlib":
            parsed_data.append([zlib, label])
    return np.array(parsed_data)


def load_data(dataset):
    def _load_sub_data(dtype: str):
        data = []
        with open(f"./dataset/{dataset}/{args.llm}_length_{args.length}_{dtype}_results.jsonl", "r") as fr:
            for line in fr.readlines():
                data.append(json.loads(line.strip()))
        return data
    train_data = process_data(_load_sub_data("train"), func=args.func, k=args.k)
    test_data = process_data(_load_sub_data("test"), func=args.func, k=args.k)

    def pack_data(data):
        X = data[:, :-1]
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=1)
        y = data[:, -1]
        return (X, y)
    
    train_X, train_y = pack_data(train_data)
    test_X, test_y = pack_data(test_data)
    return {"train": (train_X, train_y), "test": (test_X, test_y)}

dataset = load_data(args.dataset)

clf = GaussianNB()

print("-------Train--------")
train_X, train_y = dataset["train"]
print(f"size:{len(train_X)}")
clf.fit(train_X, train_y)
train_pred = clf.predict(train_X)
plot_result(train_y, train_pred)
# print(clf.classes_)
# print(clf.theta_)
# print(clf.var_)

print("-------Test--------")
test_X, test_y = dataset["test"]
print(f"size:{len(test_X)}")
test_pred = clf.predict(test_X)
test_auc = plot_result(test_y, test_pred)