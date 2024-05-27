import os

os.environ["KERAS_BACKEND"] = "torch"  # or "tensorflow" or "torch"

import keras_nlp
import numpy as np
import keras_core as keras
import json
import zlib
from tqdm import tqdm
from perplexity import Perplexity
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, choices=[32, 64, 128, 256, 512])
parser.add_argument("--llm", type=str, choices=["llama-7b", "llama-13b", "pythia-2.8b"])
parser.add_argument("--dataset", type=str, choices=["wiki-spgc", "wikimia2", "wikimia2-spgc", "bookmia"])
args = parser.parse_args()

def load_dataset(dataset):
    train_data = []
    with open(f"./dataset/{dataset}/llm_{args.llm}_length_{args.length}_train_results.jsonl", "r") as fr:
        for line in fr.readlines():
            train_data.append(json.loads(line.strip()))
    test_data = []
    with open(f"./dataset/{dataset}/llm_{args.llm}_length_{args.length}_test_results.jsonl", "r") as fr:
        for line in fr.readlines():
            test_data.append(json.loads(line.strip()))
    return {"train": train_data, "test": test_data}

dataset = load_dataset(args.dataset)
ppl_metric = Perplexity(from_logits=True)

def calc_ppl(y_true, y_pred):
    ppl_metric.update_state(y_true, y_pred)
    ppl_score = keras.ops.convert_to_numpy(ppl_metric.result()).tolist()
    ppl_metric.reset_state()
    return ppl_score

for k, v in dataset.items():
    test_ds = v
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en", sequence_length=args.length,
    )
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en", preprocessor=preprocessor
    )

    gpt2_lm.load_weights(f"./dataset/{args.dataset}/length_{args.length}_{k}.weights.h5")
    gpt2_lm.compile(
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[Perplexity(from_logits=True)],
    )

    with open(f"./dataset/{args.dataset}/{args.llm}_length_{args.length}_{k}_results.jsonl", "w") as fw:
        for d in tqdm(test_ds, ncols=80):
            x, y, _ = preprocessor([d[0]])
            logit = gpt2_lm(x)
            prob = keras.ops.convert_to_numpy(keras.ops.softmax(logit[0], axis=-1))
            max_prob = np.max(prob, axis=-1).tolist()
            conditional_prob = np.array([prob[i, y[0][i]] for i in range(len(prob))]).tolist()

            ppl = calc_ppl(y, logit)
            # Lowercase PPL
            x, y, _ = preprocessor([d[0].lower()])
            logit = gpt2_lm(x)
            lowercase_ppl = calc_ppl(y, logit)
            lowercase_radio = lowercase_ppl / ppl
            zlib_radio = ppl / len(zlib.compress(bytes(d[0], "utf-8")))

            tmp_sample = json.dumps([ppl, lowercase_radio, zlib_radio, max_prob, conditional_prob, d[1]])
            fw.write(f"{tmp_sample}\n")