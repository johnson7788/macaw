# Macaw

## 简介

(Multi-angle c(q)uestion answering ) Macaw是一个随时可用的模型，能够进行一般的问答，在它被训练的领域之外显示出鲁棒性。
它以 "多角度 "的方式被训练，这意味着它可以处理一组灵活的输入和输出 "槽"（如问题、答案、解释）。

Macaw 是建立在 [T5](https://github.com/google-research/text-to-text-transfer-transformer) 之上， 
有不同的size:  [macaw-11b](https://huggingface.co/allenai/macaw-11b), [macaw-3b](https://huggingface.co/allenai/macaw-3b), 
and [macaw-large](https://huggingface.co/allenai/macaw-large), 
以及各种排行榜上以答案为重点的版本 [macaw-answer-11b](https://huggingface.co/allenai/macaw-answer-11b) (see [below](#training-data)).

### 示例

来自Macaw（11B）模型的一些提示性例子，用于不同角度。

  * (Q→A) <i>Given a question, what's the answer?</i> <br>
  **Q: James went camping in the woods, but forgot to bring a hammer to bang the tent pegs in. What else might he use? <br> 
  → A: rocks**
  
  * (QM→A) <i>Given a question and answer choices, what's the answer?</i> <br>
  **Q: James went camping in the woods, but forgot to bring a hammer to bang the tent pegs in. What else might he use? <br> 
           M: (A) a leaf (B) a log (C) a worm <br>
  → A: a log**

  * (Q→AE) <i>Given a question, what's the answer and an explanation?</i><br>
  **Q: Which force pulls objects to the ground? <br>
  → A: gravity <br>
  → E: Gravitational force causes objects that have mass to be pulled down on a planet.**

  * (A→QE) <i>Given an answer, what's a plausible question and explanation?</i><br>
  **A: elephant <br>
  → Q: Which animal has the largest ears? <br>
  → E: The ears of an elephant are the largest.**

  * (C→QA) <i>Given a context, what's a plausible question and answer?</i><br>
  **C: A car needs a battery to start. <br>
  → Q: What is required for a car to start? <br>
  → A: battery**
  
更多基本的示例 Q→A angle, see [examples.md](examples.md).

## Usage examples

Macaw可以很容易地用在Hugging Face [transformers](https://github.com/huggingface/transformers)库中，如图所示为最小的模型（一般不推荐最小的模型，但体积更小），给定一个问题，我们要返回一个答案和建议的多选答案选项。

模型使用示例：
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("allenai/macaw-large")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/macaw-large")
input_string = "$answer$ ; $mcoptions$ ; $question$ = What is the color of a cloudy sky?"
input_ids = tokenizer.encode(input_string, return_tensors="pt")
output = model.generate(input_ids, max_length=200)

>>> tokenizer.batch_decode(output, skip_special_tokens=True)
['$answer$ = gray ; $mcoptions$ = (A) blue (B) white (C) grey (D) white']
```

(运行`pip install -r requirements.txt`如果缺少任何依赖性）。
注意不能保证不同的槽是完全一致的，就像这里的gray/grey（和重复的 "white"），对于macaw-large的模型与大的模型来说，更应该如此

`macaw/utils.py`中的代码包括一些方便的封装器，如`load_model`和`run_macaw`，这里有一些例子，将macaw-11b模型加载到两个GPU上（需要大约48GB的总GPU内存，最大的模型才能工作）。

使用示例
```
from macaw.utils import load_model, run_macaw
model_dict = load_model("allenai/macaw-11b", cuda_devices=[0,1])
res1 = run_macaw("Q: Which force pulls objects to the ground?\nA\nE", model_dict)
# Alternate input syntax
res2 = run_macaw({"Q:":"Which force causes a compass needle to point north?", "A":""}, model_dict)
# Add sampling options for the output
res3 = run_macaw("Q: Which force pulls objects to the ground?\nA\nE", model_dict, {"do_sample": True, "temperature": 2.0})

>>> [print(res["output_slots_list"][0]) for res in [res1, res2, res3]]
{'answer': 'gravity', 'explanation': 'Gravitational force causes objects that have mass to be pulled down on a planet.'}
{'answer': 'magnetism'}
{'answer': 'gravitional force', 'explanation': 'Gravitational force causes objects that have mass to be pulled down on a planet.'}
```

用于对不同角度的实例进行批次评估。 see [`macaw/batch_eval.py`](macaw/batch_eval.py) for pointers.

## 支持的槽位 slots
下面是Macaw中可用的槽位，一般适用于输入和输出。

| Slot name | Description | Example | 
|---|---|---|
|question (Q) | Question text | What is the color of a cloudy sky? |
|answer (A) | Answer text | The sky is blue |
|mcoptions (M) | Multiple-choice answer options |  (A) blue (B) white (C) grey |
|context (C) | Potentially relevant context (noisy IR) | The sky looks blue to us because... |
|explanation (E) | Sentences explaining the answer | A cloudy sky is usually gray in color... |

一个角度是一组特定的输入/输出槽，例如QM->AE是在给定一个问题和多选题的情况下，产生答案和解释的任务。
Macaw在各种各样的角度上进行训练，也能处理未见过的角度，一个例外是上下文（C）只作为一个输入槽出现在训练数据中。
  
## The Challenge300 dataset of probing questions

由300个不同的探测实例组成的**Challenge300**数据集可以在以下网站找到
[challenge300-probes-v1.jsonl](challenge300-probes-v1.jsonl). The basic Q→A output
from Macaw (at different sizes), as well as outputs from [GPT3](https://arxiv.org/pdf/2005.14165.pdf), 
[Jurassic-1](https://www.ai21.com/blog/announcing-ai21-studio-and-jurassic-1) and 
[alternate T5 models](https://www.aclweb.org/anthology/2020.emnlp-main.437/) trained on NaturalQuestions, can be seen in
[examples.md](examples.md).

## Demo是咧
参见[DEMO.md](DEMO.md)，以获得主持Macaw互动版本的说明和代码。

## 训练数据

Macaw的训练分两步进行，从文本到文本的transformer model [T5](https://github.com/google-research/text-to-text-transfer-transformer):

   1. 通过对以下7个数据集和相关角度的T5进行微调，实现了[UnifiedQA](https://github.com/allenai/unifiedqa)的多角度版本。
       * [BoolQ](https://github.com/google-research-datasets/boolean-questions), 
       [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer), 
       [NarrativeQA](https://github.com/deepmind/narrativeqa): QC→A, AC→Q
       * [ARC](https://allenai.org/data/arc), [OBQA](https://allenai.org/data/open-book-qa): 
       QMC→A, QC→A, QM→A,QAC→M, MAC→Q, AC→QM
       * [RACE](https://www.cs.cmu.edu/~glai1/data/race/), 
       [MCTest](https://mattr1.github.io/mctest/): QMC→A, QC→A, QAC→M,MAC→Q
       
   2. 在选择题和直接回答的初级科学问题上进一步微调多角度统一QA，同时从[WorldTreeV2](http://cognitiveai.org/explanationbank/)获得（多达5个）解释句子。
       * [ARC](https://allenai.org/data/arc): QMC→AE, AQC→M, QMEC→A, QME→A, QE→A, QMC→A, QC→AE, QM→AE, QMAC→E, QMA→E
       * [ARC-DA](https://allenai.org/data/arc-da): QC→AE, Q→AE, QC→A, Q→A, QEC→A, QE→A, AE→Q, AC→Q, QA→E, AQC→E
       
   3. 一个专门的以答案为中心的模型，<b>macaw-answer-11b</b>（在[ARC](https://leaderboard.allenai.org/arc/submissions/public)的排行榜上称为 "UnifiedQA + ARC MC/DA + IR"。
   [ARC-Easy](https://leaderboard.allenai.org/arc_easy/submissions/public), and 
   [ARC-DA](https://leaderboard.allenai.org/genie-arcda/submissions/public))
   是在一个较小的角度集上训练的，不包括解释。
       * ARC: QMC→A, QAC→M, QC→A, QM→A, MAC→Q, AC→QM, M→QA
       * ARC-DA: QC→A, Q→A, AC→Q, C→QA
       
   
## 可用模型
Macaw模型可以从Hugging Face模型中心访问。
   * [macaw-11b](https://huggingface.co/allenai/macaw-11b)  (110亿参数)
   * [macaw-3b](https://huggingface.co/allenai/macaw-3b)  (3亿参数)
   * [macaw-large](https://huggingface.co/allenai/macaw-large)  (770 million parameters)
   * [macaw-answer-11b](https://huggingface.co/allenai/macaw-answer-11b)  (11 billion parameters)

为了了解较小尺寸的性能下降情况，这里是ARC挑战赛和ARC简易多选题<b>发展</b>的基线分数。
包括有和没有IR背景的变体，这些变体来自一个大型科学语料库（分别对应于QMC→A和QM→A角度）。

|Model | ARC Challenge | ARC Challenge (no IR) | ARC Easy | ARC Easy (no IR)|
|---|---|---|---|---|
|Macaw (11B) | 76.9 | 74.6 | 91.2 | 84.9|
|Macaw-3B | 68.2 | 67.9 | 87.9 |  77.7|
|Macaw-large | 57.2 | 50.5 | 82.5 | 63.9|
|Macaw-answer (11B) | 79.9 | 75.2 | 90.5 | 85.8|

## 免责声明

作为一个能够生成自由形式文本的模型，该模型的输出不能保证没有辱骂等材料，因此建议在使用该模型时要适当谨慎。

## Citation

If you use Macaw in your work, please reference the related [paper](https://arxiv.org/abs/2109.02593) using

```
@article{Tafjord2021Macaw,
  title={General-Purpose Question-Answering with {M}acaw},
  author={Oyvind Tafjord and Peter Clark},
  journal={ArXiv},
  year={2021},
  volume={abs/2109.02593}
}
```
