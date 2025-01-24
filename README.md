# Automatic Dialogues Evaluation

Project of course **Artificial Intelligence** - University of Salerno.

## Contributors
[@raffaele-aurucci](https://github.com/raffaele-aurucci), [@AngeloPalmieri](https://github.com/AngeloPalmieri), [@CSSabino](https://github.com/CSSabino).

## Table of Contents

1. [Introduction](#introduction)
2. [Methodology](#methodology)
   - [Preliminary](#preliminary)
   - [Prompt Engineering](#prompt-engineering)
3. [Experimental Results](#experimental-results)
   - [DSTC9 Dataset](#dstc9_dataset)
   - [Convai2 Dataset](#dstc9_dataset)
   - [FED dataset](#dstc9_dataset)
   - [PC USR dataset](#dstc9_dataset)
   - [TC USR dataset](#dstc9_dataset)
4. [Installation Guide](#installation-guide)
   - [Installing Python](#installing-python)
   - [Cloning the Repository](#cloning-the-repository)
   - [Creating the Virtual Environment](#creating-the-virtual-environment)
   - [Installing Requirements](#installing-requirements)

## Introduction
This study focuses on developing a framework for automatic dialogue evaluation to improve chatbots, virtual assistants and linguistic applications. Human evaluations, while accurate, are costly, hard to reproduce, and not scalable. To address this, we explored automated approaches using advanced Large Language Models (LLMs).  
Building on the framework from *“A Comprehensive Analysis of the Effectiveness of Large Language Models as Automatic Dialogue Evaluators”* ([reference](https://arxiv.org/abs/2312.15407)), we implemented a system to evaluate dialogue quality at both the turn and dialogue levels using state-of-the-art proprietary and open-source LLMs.

## Methodology

### Preliminary
The primary goal of this research is to develop a system capable of automatically evaluating dialogue quality using advanced language models. These models enable scalable and reliable analysis, allowing comparisons with human annotations to assess alignment and performance.

To achieve this, two types of datasets are used:
1. A dataset with human annotations, serving as a reference or "oracle".
2. Internally generated datasets created by interacting with the models themselves.

The aim is to compare model evaluations against human annotations and analyze the correlation between them using metrics as **Pearson**, **Spearman** and **Kendall Tau**.  

Both proprietary and open-source language models are evaluated:
- **Proprietary Model**: GPT4
- **Open-Source Models**: Baichuan2-13B, Chatglm3-6B, Chimera13B, Llama2-13B, Qwen14B, and Vicuna13B.


The evaluation is conducted at both the turn level and dialogue level, as described in the previous paper. Tests were performed on five datasets: 
**DSTC9 Dataset**, **Convai2 Dataset**, **FED Dataset**, **PC USR Dataset**, **TC USR Dataset**.

### Prompt Engineering
In order to interact effectively with the models being tested, it's essential to design prompts that clearly and unambiguously define the task and desired output. Following the approach of the previous paper, distinct prompts were used for **dialogue-level** evaluation depending on whether the model is proprietary or open-source.

#### Proprietary Model Prompt for dialogue-level (GPT-4)
```text
### Dialogues:
[Here is the input dialogue for annotation]

### Instruction:
Rate the coherence, engagingness, diversity, informativeness, and overall quality 
of the input dialogue on a scale of 1 to 5 and just output the corresponding ratings.

### Output Format:
coherence - x  
engagingness - x  
diversity - x  
informativeness - x  
overall - x  

### Your Answer:
[Here is GPT-4’s output]
```

#### Open Source Model Prompt for dialogue-level
```text
### Dialogues:
[Here is the input dialogue for annota-tion]

### Instruction:
Above is a dialogue.

Question: Is the overall quality of the dialogue satisfactory?

### Your Answer:
[Here is LLM’s output in terms of ”Yes” or ”No”]
```


For open-source models, the evaluation of dialogue quality was also performed at the **turn-level**, following the approach outlined in the previous work.
#### Open Source Model Prompt for turn-level
```text
### Context:
[Here is the dialogue context]

### Response:
[Here is the input response for annotation]

### Instruction:
Above is a dialogue context and the corresponding response.

Question: Is the overall quality of the response satisfactory to the context?

### Your Answer:
[Here is LLM’s output in terms of “Yes” or “No”]
```

## Experimental Results
The tables below present the benchmark results across various datasets, providing a comprehensive comparison of performance metrics.

### DSTC9 Dataset
<table>
    <tr>
        <th colspan="4">Dialogue level</th>
    </tr>
    <tr>
        <th>Model</th>
        <th>Pearson</th>
        <th>Spearman</th>
        <th>Kendall Tau</th>
    </tr>
    <tr>
        <td>Baichuan2-13B</td>
        <td>0.116</td>
        <td>0.128</td>
        <td>0.091</td>
    </tr>
    <tr>
        <td>Chatglm3-6B</td>
        <td>0.110</td>
        <td>0.110</td>
        <td>0.078</td>
    </tr>
    <tr>
        <td>Chimera13B</td>
        <td>0.090</td>
        <td>0.131</td>
        <td>0.094</td>
    </tr>
    <tr>
        <td>Llama2-13B</td>
        <td>0.051</td>
        <td>0.052</td>
        <td>0.037</td>
    </tr>
    <tr>
        <td>Qwen14B</td>
        <td>0.195</td>
        <td>0.215</td>
        <td>0.155</td>
    </tr>
    <tr>
        <td>Vicuna13B</td>
        <td><b><u>0.224</u></b></td>
        <td><b><u>0.228</u></b></td>
        <td><b><u>0.164</u></b></td>
    </tr>
    <tr>
        <td colspan="4"></td>
    </tr>
    <tr>
        <td>GPT4</td>
        <td><b><u>0.263</u></b></td>
        <td><b><u>0.252</u></b></td>
        <td><b><u>0.185</u></b></td>
    </tr>
</table>

### Convai2 Dataset
<table>
    <tr>
        <th colspan="4">Dialogue level</th>
    </tr>
    <tr>
        <th>Model</th>
        <th>Pearson</th>
        <th>Spearman</th>
        <th>Kendall Tau</th>
    </tr>
    <tr>
        <td>Baichuan2-13B</td>
        <td><b><u>0.349</u></b></td>
        <td><b><u>0.414</u></b></td>
        <td><b><u>0.308</u></b></td>
    </tr>
    <tr>
        <td>Chatglm3-6B</td>
        <td>0.324</td>
        <td>0.323</td>
        <td>0.238</td>
    </tr>
    <tr>
        <td>Llama2-13B</td>
        <td>0.194</td>
        <td>0.189</td>
        <td>0.141</td>
    </tr>
    <tr>
        <td>Qwen14B</td>
        <td>0.077</td>
        <td>0.027</td>
        <td>0.024</td>
    </tr>
    <tr>
        <td>Vicuna13B</td>
        <td>-0.153</td>
        <td>-0.173</td>
        <td>-0.126</td>
    </tr>
</table>

### FED Dataset
<table>
    <tr>
        <th colspan="4">Dialogue level</th>
    </tr>
    <tr>
        <th>Model</th>
        <th>Pearson</th>
        <th>Spearman</th>
        <th>Kendall Tau</th>
    </tr>
    <tr>
        <td>Baichuan2-13B</td>
        <td>0.469</td>
        <td><b><u>0.574</u></b></td>
        <td><b><u>0.415</u></b></td>
    </tr>
    <tr>
        <td>Chatglm3-6B</td>
        <td>0.017</td>
        <td>-0.004</td>
        <td>0.001</td>
    </tr>
    <tr>
        <td>Llama2-13B</td>
        <td>0.194</td>
        <td>0.189</td>
        <td>0.141</td>
    </tr>
    <tr>
        <td>Qwen14B</td>
        <td>0.077</td>
        <td>0.027</td>
        <td>0.024</td>
    </tr>
    <tr>
        <td>Vicuna13B</td>
        <td><b><u>0.537</u></b></td>
        <td>0.517</td>
        <td>0.354</td>
    </tr>
</table>

<table>
    <tr>
        <th colspan="4">Turn level</th>
    </tr>
    <tr>
        <th>Model</th>
        <th>Pearson</th>
        <th>Spearman</th>
        <th>Kendall Tau</th>
    </tr>
    <tr>
        <td>Baichuan2-13B</td>
        <td>0.416</td>
        <td>0.357</td>
        <td>0.251</td>
    </tr>
    <tr>
        <td>Chatglm3-6B</td>
        <td>0.385</td>
        <td>0.300</td>
        <td>0.210</td>
    </tr>
    <tr>
        <td>Llama2-13B</td>
        <td>0.299</td>
        <td>0.269</td>
        <td>0.190</td>
    </tr>
    <tr>
        <td>Qwen14B</td>
        <td>0.315</td>
        <td>0.261</td>
        <td>0.175</td>
    </tr>
    <tr>
        <td>Vicuna13B</td>
        <td><b><u>0.499</u></b></td>
        <td><b><u>0.491</u></b></td>
        <td><b><u>0.356</u></b></td>
    </tr>
</table>

### TC USR Dataset

<table>
    <tr>
        <th colspan="4">Turn level</th>
    </tr>
    <tr>
        <th>Models</th>
        <th>Pearson</th>
        <th>Spearman</th>
        <th>Kendall Tau</th>
    </tr>
    <tr>
        <td>Baichuan2-13B</td>
        <td>0.171</td>
        <td>0.310</td>
        <td>0.218</td>
    </tr>
    <tr>
        <td>Chatglm3-6B</td>
        <td>0.265</td>
        <td>0.255</td>
        <td>0.178</td>
    </tr>
    <tr>
        <td>Llama2-13B</td>
        <td>0.324</td>
        <td>0.349</td>
        <td>0.246</td>
    </tr>
    <tr>
        <td>Qwen14B</td>
        <td>0.239</td>
        <td>0.238</td>
        <td>0.168</td>
    </tr>
    <tr>
        <td>Vicuna13B</td>
        <td><b><u>0.352</u></b></td>
        <td><b><u>0.384</u></b></td>
        <td><b><u>0.271</u></b></td>
    </tr>
</table>

### PC USR Dataset
<table>
    <tr>
        <th colspan="4">Turn level</th>
    </tr>
    <tr>
        <th>Models</th>
        <th>Pearson</th>
        <th>Spearman</th>
        <th>Kendall Tau</th>
    </tr>
    <tr>
        <td>Baichuan2-13B</td>
        <td>0.341</td>
        <td><b><u>0.391</u></b></td>
        <td><b><u>0.286</u></b></td>
    </tr>
    <tr>
        <td>Chatglm3-6B</td>
        <td><b><u>0.404</u></b></td>
        <td>0.364</td>
        <td>0.254</td>
    </tr>
    <tr>
        <td>Llama2-13B</td>
        <td>0.349</td>
        <td>0.342</td>
        <td>0.244</td>
    </tr>
    <tr>
        <td>Qwen14B</td>
        <td>0.291</td>
        <td>0.281</td>
        <td>0.202</td>
    </tr>
    <tr>
        <td>Vicuna13B</td>
        <td>0.300</td>
        <td>0.307</td>
        <td>0.217</td>
    </tr>
</table>




## Installation Guide
To install the necessary requirements for the project, please follow the steps below.

### Installing Python
Verify you have Python installed on your machine. The project is compatible with Python `3.12.1`.

If you do not have Python installed, please refer to the official [Python Guide](https://www.python.org/downloads/).

### Cloning the Repository 
To clone this repository, download and extract the `.zip` project files using the `<Code>` button on the top-right or run the following command in your terminal:
```shell 
git clone https://github.com/raffaele-aurucci/Ludo_Game_AI.git
```

### Creating the Virtual Environment 
It's strongly recommended to create a virtual environment for the project and activate it before proceeding. 
Feel free to use any Python package manager to create the virtual environment. However, for a smooth installation of the requirements we recommend you use `pip`. Please refer to [Creating a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).

You may skip this step, but please keep in mind that doing so could potentially lead to conflicts if you have other projects on your machine. 
### Installing Requirements
To install the requirements, please: 
1. Make sure you have **activated the virtual environment where you installed the project's requirements**. If activated, your terminal, assuming you are using **bash**, should look like the following: ``(name-of-your-virtual-environment) user@user path``

2. Install the project requirements using `pip`:
```shell 
pip install -r requirements.txt
```