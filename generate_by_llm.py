import json
import torch
import argparse
from tqdm import tqdm
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, choices=[32, 64, 128, 256, 512])
parser.add_argument("--llm", type=str, choices=["llama-7b", "llama-13b", "pythia-2.8b"])
parser.add_argument("--dataset", type=str, choices=["wiki-spgc", "wikimia2", "bookmia"])
args = parser.parse_args()

if args.llm == "llama-7b":
    model_path = "/kfdata01/kf_grp/LLM_models/english/llama-7b-hf/"
    model = pipeline(
        "text-generation", model=model_path, device_map="auto", torch_dtype=torch.float16,
    )
elif args.llm == "llama-13b":
    model_path = "/kfdata01/kf_grp/LLM_models/english/llama-13b-hf/"
    model = pipeline(
        "text-generation", model=model_path, device_map="auto", torch_dtype=torch.float16
    )
elif args.llm == "pythia-2.8b":
    model_path = "EleutherAI/pythia-2.8b-deduped"
    model = pipeline(
        "text-generation", model=model_path, device_map="auto", torch_dtype=torch.float16,
    )

def load_dataset(dataset):
    train_data, test_data = [], []
    with open(f"./dataset/{dataset}/length_{args.length}.json", "r") as fr:
        data = json.load(fr)
        train_data = data["train"]
        test_data = data["test"]
    return {"train": train_data, "test": test_data}

dataset = load_dataset(args.dataset)

for k, v in dataset.items():
    with open(f"./dataset/{args.dataset}/llm_{args.llm}_length_{args.length}_{k}_results.jsonl", "w") as fw:
        for data in tqdm(v, ncols=80):
            text, label = data
            tokens = text.split(" ")
            prompt = " ".join(tokens[:args.length // 2])
            try:
                response = model(
                    prompt, do_sample=True, max_new_tokens=args.length, temperature=0.7
                )
                response = response[0]["generated_text"]
                fw.write(json.dumps((response, label)))
                fw.write("\n")
            except Exception as ex:
                print("Runtime Error")
                fw.write(json.dumps((text, label)))
                fw.write("\n")