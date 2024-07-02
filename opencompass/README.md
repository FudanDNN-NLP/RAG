<div align="center">
  <img src="docs/en/_static/image/logo.svg" width="500px"/>
  <br />
  <br />

[![][github-release-shield]][github-release-link]
[![][github-releasedate-shield]][github-releasedate-link]
[![][github-contributors-shield]][github-contributors-link]<br>
[![][github-forks-shield]][github-forks-link]
[![][github-stars-shield]][github-stars-link]
[![][github-issues-shield]][github-issues-link]
[![][github-license-shield]][github-license-link]

<!-- [![PyPI](https://badge.fury.io/py/opencompass.svg)](https://pypi.org/project/opencompass/) -->

[🌐Website](https://opencompass.org.cn/) |
[📖CompassHub](https://hub.opencompass.org.cn/home) |
[📊CompassRank](https://rank.opencompass.org.cn/home) |
[📘Documentation](https://opencompass.readthedocs.io/en/latest/) |
[🛠️Installation](https://opencompass.readthedocs.io/en/latest/get_started/installation.html) |
[🤔Reporting Issues](https://github.com/open-compass/opencompass/issues/new/choose)

English | [简体中文](README_zh-CN.md)

[![][github-trending-shield]][github-trending-url]

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/KKwfEbFj7U" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=opencompass" target="_blank">WeChat</a>
</p>

> \[!IMPORTANT\]
>
> **Star Us**, You will receive all release notifications from GitHub without any delay ~ ⭐️

## 📣 OpenCompass 2.0

We are thrilled to introduce OpenCompass 2.0, an advanced suite featuring three key components: [CompassKit](https://github.com/open-compass), [CompassHub](https://hub.opencompass.org.cn/home), and [CompassRank](https://rank.opencompass.org.cn/home).
![oc20](https://github.com/tonysy/opencompass/assets/7881589/90dbe1c0-c323-470a-991e-2b37ab5350b2)

**CompassRank** has been significantly enhanced into the leaderboards that now incorporates both open-source benchmarks and proprietary benchmarks. This upgrade allows for a more comprehensive evaluation of models across the industry.

**CompassHub** presents a pioneering benchmark browser interface, designed to simplify and expedite the exploration and utilization of an extensive array of benchmarks for researchers and practitioners alike. To enhance the visibility of your own benchmark within the community, we warmly invite you to contribute it to CompassHub. You may initiate the submission process by clicking [here](https://hub.opencompass.org.cn/dataset-submit).

**CompassKit** is a powerful collection of evaluation toolkits specifically tailored for Large Language Models and Large Vision-language Models. It provides an extensive set of tools to assess and measure the performance of these complex models effectively. Welcome to try our toolkits for in your research and products.

<details>
  <summary><kbd>Star History</kbd></summary>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=open-compass%2Fopencompass&theme=dark&type=Date">
    <img width="100%" src="https://api.star-history.com/svg?repos=open-compass%2Fopencompass&type=Date">
  </picture>
</details>

## 🧭	Welcome

to **OpenCompass**!

Just like a compass guides us on our journey, OpenCompass will guide you through the complex landscape of evaluating large language models. With its powerful algorithms and intuitive interface, OpenCompass makes it easy to assess the quality and effectiveness of your NLP models.

🚩🚩🚩 Explore opportunities at OpenCompass! We're currently **hiring full-time researchers/engineers and interns**. If you're passionate about LLM and OpenCompass, don't hesitate to reach out to us via [email](mailto:zhangsongyang@pjlab.org.cn). We'd love to hear from you!

🔥🔥🔥 We are delighted to announce that **the OpenCompass has been recommended by the Meta AI**, click [Get Started](https://ai.meta.com/llama/get-started/#validation) of Llama for more information.

> **Attention**<br />
> We launch the OpenCompass Collaboration project, welcome to support diverse evaluation benchmarks into OpenCompass!
> Clike [Issue](https://github.com/open-compass/opencompass/issues/248) for more information.
> Let's work together to build a more powerful OpenCompass toolkit!

## 🚀 What's New <a><img width="35" height="20" src="https://user-images.githubusercontent.com/12782558/212848161-5e783dd6-11e8-4fe0-bbba-39ffb77730be.png"></a>

- **\[2024.05.08\]** We supported the evaluation of 4 MoE models: [Mixtral-8x22B-v0.1](configs/models/mixtral/hf_mixtral_8x22b_v0_1.py), [Mixtral-8x22B-Instruct-v0.1](configs/models/mixtral/hf_mixtral_8x22b_instruct_v0_1.py), [Qwen1.5-MoE-A2.7B](configs/models/qwen/hf_qwen1_5_moe_a2_7b.py), [Qwen1.5-MoE-A2.7B-Chat](configs/models/qwen/hf_qwen1_5_moe_a2_7b_chat.py). Try them out now!
- **\[2024.04.30\]** We supported evaluating a model's compression efficiency by calculating its Bits per Character (BPC) metric on an [external corpora](configs/datasets/llm_compression/README.md) ([official paper](https://github.com/hkust-nlp/llm-compression-intelligence)). Check out the [llm-compression](configs/eval_llm_compression.py) evaluation config now! 🔥🔥🔥
- **\[2024.04.29\]** We report the performance of several famous LLMs on the common benchmarks, welcome to [documentation](https://opencompass.readthedocs.io/en/latest/user_guides/corebench.html) for more information! 🔥🔥🔥.
- **\[2024.04.26\]** We deprecated the multi-madality evaluating function from OpenCompass, related implement has moved to [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), welcome to use! 🔥🔥🔥.
- **\[2024.04.26\]** We supported the evaluation of [ArenaHard](configs/eval_subjective_arena_hard.py)  welcome to try!🔥🔥🔥.
- **\[2024.04.22\]** We supported the evaluation of [LLaMA3](configs/models/hf_llama/hf_llama3_8b.py) 和 [LLaMA3-Instruct](configs/models/hf_llama/hf_llama3_8b_instruct.py), welcome to try! 🔥🔥🔥
- **\[2024.02.29\]** We supported the MT-Bench, AlpacalEval and AlignBench, more information can be found [here](https://opencompass.readthedocs.io/en/latest/advanced_guides/subjective_evaluation.html)
- **\[2024.01.30\]** We release OpenCompass 2.0. Click  [CompassKit](https://github.com/open-compass), [CompassHub](https://hub.opencompass.org.cn/home), and [CompassRank](https://rank.opencompass.org.cn/home) for more information !

> [More](docs/en/notes/news.md)

## ✨ Introduction

![image](https://github.com/open-compass/opencompass/assets/22607038/f45fe125-4aed-4f8c-8fe8-df4efb41a8ea)

OpenCompass is a one-stop platform for large model evaluation, aiming to provide a fair, open, and reproducible benchmark for large model evaluation. Its main features include:

- **Comprehensive support for models and datasets**: Pre-support for 20+ HuggingFace and API models, a model evaluation scheme of 70+ datasets with about 400,000 questions, comprehensively evaluating the capabilities of the models in five dimensions.

- **Efficient distributed evaluation**: One line command to implement task division and distributed evaluation, completing the full evaluation of billion-scale models in just a few hours.

- **Diversified evaluation paradigms**: Support for zero-shot, few-shot, and chain-of-thought evaluations, combined with standard or dialogue-type prompt templates, to easily stimulate the maximum performance of various models.

- **Modular design with high extensibility**: Want to add new models or datasets, customize an advanced task division strategy, or even support a new cluster management system? Everything about OpenCompass can be easily expanded!

- **Experiment management and reporting mechanism**: Use config files to fully record each experiment, and support real-time reporting of results.

## 📊 Leaderboard

We provide [OpenCompass Leaderboard](https://rank.opencompass.org.cn/home) for the community to rank all public models and API models. If you would like to join the evaluation, please provide the model repository URL or a standard API interface to the email address `opencompass@pjlab.org.cn`.

<p align="right"><a href="#top">🔝Back to top</a></p>

## 🛠️ Installation

Below are the steps for quick installation and datasets preparation.

### 💻 Environment Setup

#### Open-source Models with GPU

```bash
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```

#### API Models with CPU-only

```bash
conda create -n opencompass python=3.10 pytorch torchvision torchaudio cpuonly -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
# also please install requirements packages via `pip install -r requirements/api.txt` for API models if needed.
```

### 📂 Data Preparation

```bash
# Download dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
```

Some third-party features, like Humaneval and Llama, may require additional steps to work properly, for detailed steps please refer to the [Installation Guide](https://opencompass.readthedocs.io/en/latest/get_started/installation.html).

<p align="right"><a href="#top">🔝Back to top</a></p>

## 🏗️ ️Evaluation

After ensuring that OpenCompass is installed correctly according to the above steps and the datasets are prepared, you can evaluate the performance of the LLaMA-7b model on the MMLU and C-Eval datasets using the following command:

```bash
python run.py --models hf_llama_7b --datasets mmlu_ppl ceval_ppl
```

OpenCompass has predefined configurations for many models and datasets. You can list all available model and dataset configurations using the [tools](./docs/en/tools.md#list-configs).

```bash
# List all configurations
python tools/list_configs.py
# List all configurations related to llama and mmlu
python tools/list_configs.py llama mmlu
```

You can also evaluate other HuggingFace models via command line. Taking LLaMA-7b as an example:

```bash
python run.py --datasets ceval_ppl mmlu_ppl --hf-type base --hf-path huggyllama/llama-7b
```

> \[!TIP\]
>
> configuration with `_ppl` is designed for base model typically.
> configuration with `_gen` can be used for both base model and chat model.

Through the command line or configuration files, OpenCompass also supports evaluating APIs or custom models, as well as more diversified evaluation strategies. Please read the [Quick Start](https://opencompass.readthedocs.io/en/latest/get_started/quick_start.html) to learn how to run an evaluation task.

<p align="right"><a href="#top">🔝Back to top</a></p>

## 📖 Dataset Support

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Language</b>
      </td>
      <td>
        <b>Knowledge</b>
      </td>
      <td>
        <b>Reasoning</b>
      </td>
      <td>
        <b>Examination</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
<details open>
<summary><b>Word Definition</b></summary>

- WiC
- SummEdits

</details>

<details open>
<summary><b>Idiom Learning</b></summary>

- CHID

</details>

<details open>
<summary><b>Semantic Similarity</b></summary>

- AFQMC
- BUSTM

</details>

<details open>
<summary><b>Coreference Resolution</b></summary>

- CLUEWSC
- WSC
- WinoGrande

</details>

<details open>
<summary><b>Translation</b></summary>

- Flores
- IWSLT2017

</details>

<details open>
<summary><b>Multi-language Question Answering</b></summary>

- TyDi-QA
- XCOPA

</details>

<details open>
<summary><b>Multi-language Summary</b></summary>

- XLSum

</details>
      </td>
      <td>
<details open>
<summary><b>Knowledge Question Answering</b></summary>

- BoolQ
- CommonSenseQA
- NaturalQuestions
- TriviaQA

</details>
      </td>
      <td>
<details open>
<summary><b>Textual Entailment</b></summary>

- CMNLI
- OCNLI
- OCNLI_FC
- AX-b
- AX-g
- CB
- RTE
- ANLI

</details>

<details open>
<summary><b>Commonsense Reasoning</b></summary>

- StoryCloze
- COPA
- ReCoRD
- HellaSwag
- PIQA
- SIQA

</details>

<details open>
<summary><b>Mathematical Reasoning</b></summary>

- MATH
- GSM8K

</details>

<details open>
<summary><b>Theorem Application</b></summary>

- TheoremQA
- StrategyQA
- SciBench

</details>

<details open>
<summary><b>Comprehensive Reasoning</b></summary>

- BBH

</details>
      </td>
      <td>
<details open>
<summary><b>Junior High, High School, University, Professional Examinations</b></summary>

- C-Eval
- AGIEval
- MMLU
- GAOKAO-Bench
- CMMLU
- ARC
- Xiezhi

</details>

<details open>
<summary><b>Medical Examinations</b></summary>

- CMB

</details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Understanding</b>
      </td>
      <td>
        <b>Long Context</b>
      </td>
      <td>
        <b>Safety</b>
      </td>
      <td>
        <b>Code</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
<details open>
<summary><b>Reading Comprehension</b></summary>

- C3
- CMRC
- DRCD
- MultiRC
- RACE
- DROP
- OpenBookQA
- SQuAD2.0

</details>

<details open>
<summary><b>Content Summary</b></summary>

- CSL
- LCSTS
- XSum
- SummScreen

</details>

<details open>
<summary><b>Content Analysis</b></summary>

- EPRSTMT
- LAMBADA
- TNEWS

</details>
      </td>
      <td>
<details open>
<summary><b>Long Context Understanding</b></summary>

- LEval
- LongBench
- GovReports
- NarrativeQA
- Qasper

</details>
      </td>
      <td>
<details open>
<summary><b>Safety</b></summary>

- CivilComments
- CrowsPairs
- CValues
- JigsawMultilingual
- TruthfulQA

</details>
<details open>
<summary><b>Robustness</b></summary>

- AdvGLUE

</details>
      </td>
      <td>
<details open>
<summary><b>Code</b></summary>

- HumanEval
- HumanEvalX
- MBPP
- APPs
- DS1000

</details>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

## 📖 Model Support

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Open-source Models</b>
      </td>
      <td>
        <b>API Models</b>
      </td>
      <!-- <td>
        <b>Custom Models</b>
      </td> -->
    </tr>
    <tr valign="top">
      <td>

- [InternLM](https://github.com/InternLM/InternLM)
- [LLaMA](https://github.com/facebookresearch/llama)
- [LLaMA3](https://github.com/meta-llama/llama3)
- [Vicuna](https://github.com/lm-sys/FastChat)
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Baichuan](https://github.com/baichuan-inc)
- [WizardLM](https://github.com/nlpxucan/WizardLM)
- [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B)
- [ChatGLM3](https://github.com/THUDM/ChatGLM3-6B)
- [TigerBot](https://github.com/TigerResearch/TigerBot)
- [Qwen](https://github.com/QwenLM/Qwen)
- [BlueLM](https://github.com/vivo-ai-lab/BlueLM)
- [Gemma](https://huggingface.co/google/gemma-7b)
- ...

</td>
<td>

- OpenAI
- Gemini
- Claude
- ZhipuAI(ChatGLM)
- Baichuan
- ByteDance(YunQue)
- Huawei(PanGu)
- 360
- Baidu(ERNIEBot)
- MiniMax(ABAB-Chat)
- SenseTime(nova)
- Xunfei(Spark)
- ……

</td>

</tr>
  </tbody>
</table>

<p align="right"><a href="#top">🔝Back to top</a></p>

## 🔜 Roadmap

- [x] Subjective Evaluation
  - [ ] Release CompassAreana
  - [x] Subjective evaluation.
- [x] Long-context
  - [x] Long-context evaluation with extensive datasets.
  - [ ] Long-context leaderboard.
- [x] Coding
  - [ ] Coding evaluation leaderboard.
  - [x] Non-python language evaluation service.
- [x] Agent
  - [ ] Support various agenet framework.
  - [x] Evaluation of tool use of the LLMs.
- [x] Robustness
  - [x] Support various attack method

## 👷‍♂️ Contributing

We appreciate all contributions to improving OpenCompass. Please refer to the [contributing guideline](https://opencompass.readthedocs.io/en/latest/notes/contribution_guide.html) for the best practice.

<!-- Copy-paste in your Readme.md file -->

<!-- Made with [OSS Insight](https://ossinsight.io/) -->

<a href="https://github.com/open-compass/opencompass/graphs/contributors" target="_blank">
  <table>
    <tr>
      <th colspan="2">
        <br><img src="https://contrib.rocks/image?repo=open-compass/opencompass"><br><br>
      </th>
    </tr>
  </table>
</a>

## 🤝 Acknowledgements

Some code in this project is cited and modified from [OpenICL](https://github.com/Shark-NLP/OpenICL).

Some datasets and prompt implementations are modified from [chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub) and [instruct-eval](https://github.com/declare-lab/instruct-eval).

## 🖊️ Citation

```bibtex
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
```

<p align="right"><a href="#top">🔝Back to top</a></p>

[github-contributors-link]: https://github.com/open-compass/opencompass/graphs/contributors
[github-contributors-shield]: https://img.shields.io/github/contributors/open-compass/opencompass?color=c4f042&labelColor=black&style=flat-square
[github-forks-link]: https://github.com/open-compass/opencompass/network/members
[github-forks-shield]: https://img.shields.io/github/forks/open-compass/opencompass?color=8ae8ff&labelColor=black&style=flat-square
[github-issues-link]: https://github.com/open-compass/opencompass/issues
[github-issues-shield]: https://img.shields.io/github/issues/open-compass/opencompass?color=ff80eb&labelColor=black&style=flat-square
[github-license-link]: https://github.com/open-compass/opencompass/blob/main/LICENSE
[github-license-shield]: https://img.shields.io/github/license/open-compass/opencompass?color=white&labelColor=black&style=flat-square
[github-release-link]: https://github.com/open-compass/opencompass/releases
[github-release-shield]: https://img.shields.io/github/v/release/open-compass/opencompass?color=369eff&labelColor=black&logo=github&style=flat-square
[github-releasedate-link]: https://github.com/open-compass/opencompass/releases
[github-releasedate-shield]: https://img.shields.io/github/release-date/open-compass/opencompass?labelColor=black&style=flat-square
[github-stars-link]: https://github.com/open-compass/opencompass/stargazers
[github-stars-shield]: https://img.shields.io/github/stars/open-compass/opencompass?color=ffcb47&labelColor=black&style=flat-square
[github-trending-shield]: https://trendshift.io/api/badge/repositories/6630
[github-trending-url]: https://trendshift.io/repositories/6630
