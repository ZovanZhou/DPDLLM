# DPDLLM

[Title] DPDLLM: A Black-box Framework for Detecting Pre-training Data from Large Language Models

[Authors] Baohang Zhou, Zezhong Wang, Lingzhi Wang, Hongru Wang, Ying Zhang, Kehui Song, Xuhui Sui, Kam-Fai Wong

[Findings of ACL 2024]()

## Preparation

1. Clone the repo to your local.
2. Download Python version: 3.7.12
3. Open the shell or cmd in this repo folder. Run this command to install necessary packages.

```cmd
pip install -r requirements.txt
```

4. Change the value **model_path** in "generate_by_llm.py" to your local LLaMA model path.
5. Download the datasets from this [link](https://pan.baidu.com/s/1uFkDO_N2GFhPkyGMrvfLAA), and the extraction code is 1234. Put them into the "dataset" folder.

## Experiments

1. Extraction: Given the prefix texts of the detection texts, you can input the following command to extract the postscripts from LLMs. The meaning of these parameters are shown in the following tables.

|  Parameters | Value | Description|
|  ----  | ----  | ---- |
|  dataset  | string | The dataset name, including: wikimia2, wiki-spgc, wikimia2-spgc, bookmia |
| length | int | The dataset with "length" version, including: 32, 64, 128, 256. |
| llm | string | The target large language models, including: llama-7b, llama-13b, pythia-2.8b |

```cmd
CUDA_VISIBLE_DEVICES=0 python generate_by_llm.py --dataset wikimia2 --length 32 --llm pythia-2.8b
```

2. Memorization: Given the detection texts, you can input the following command to train the reference model GPT-2 for memorizing the them. The meaning of these parameters are shown in the following tables.

|  Parameters | Value | Description|
|  ----  | ----  | ---- |
|  dataset  | string | The dataset name, including: wikimia2, wiki-spgc, wikimia2-spgc, bookmia |
| length | int | The dataset with "length" version, including: 32, 64, 128, 256. |

```cmd
CUDA_VISIBLE_DEVICES=0 python train_detector.py --dataset wikimia2 --length 32
```

3. Detection Feature: You can input the following command to obtain the detection features of the detection texts. The meaning of these parameters are shown in the following tables.

|  Parameters | Value | Description|
|  ----  | ----  | ---- |
|  dataset  | string | The dataset name, including: wikimia2, wiki-spgc, wikimia2-spgc, bookmia |
| length | int | The dataset with "length" version, including: 32, 64, 128, 256. |
| llm | string | The target large language models, including: llama-7b, llama-13b, pythia-2.8b |

```cmd
CUDA_VISIBLE_DEVICES=0 python run_detection.py --dataset wikimia2 --length 32 --llm pythia-2.8b
```

4. Classification: You can input the following command to train the classifier for the detection and evaluation. The meaning of these parameters are shown in the following tables.

|  Parameters | Value | Description|
|  ----  | ----  | ---- |
|  dataset  | string | The dataset name, including: wikimia2, wiki-spgc, wikimia2-spgc, bookmia |
| length | int | The dataset with "length" version, including: 32, 64, 128, 256. |
| llm | string | The target large language models, including: llama-7b, llama-13b, pythia-2.8b |
| func | string | The function to process the detection features, including: original, mintopk, average, minsubstr |
| metric | string | The types of the detection features, including: max_prob, cond_prob, ppl, lowercase_ppl, zlib |
| k | int | The hyper-parameter for the processing function "mintopk" |

```cmd
python predict_result.py --dataset wikimia2 --length 32 --llm pythia-2.8b
```
