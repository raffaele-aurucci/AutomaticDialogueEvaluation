# Automatic Dialogues Evaluation

Project of course **Artificial Intelligence** - University of Salerno.

## Contributors
[@raffaele-aurucci](https://github.com/raffaele-aurucci), [@AngeloPalmieri](https://github.com/AngeloPalmieri), [@CSSabino](https://github.com/CSSabino).

## Table of Contents

1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Experimental Results](#experimental-results)
   - [Preliminary](#preliminary)
   - [Prompt Engineering](#prompt-engineering)
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
**FED Dataset**, **Convai2 Dataset**, **DSTC9 Dataset**, **PC USR Dataset**, **TC USR Dataset**.

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