import json
import math
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os


# Process the desired output.
def process_list(dialogue_id: int, output_tokens_prob: list):
    return {
        "id_dialogue": dialogue_id,
        "yes": output_tokens_prob[0]['Yes'],
        "no": output_tokens_prob[1]['No']
    }

# Create prompt for turn.
def create_prompt(context: list, response: list):

    prompt = f"""
    ### Context:
    {context}
    
    ### Response:
    {response}
    
    ## Instruction:
    Above is a dialogue context and the corresponding response.
    
    Question: Is the overall quality of the response satisfactory to the context?

    ### Your Answer:
    """
    return prompt


def make_inferences():

    # Suppress warnings.
    sys.stderr = open(os.devnull, 'w')

    # Load model.
    checkpoint = "Qwen/Qwen-14B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)

    # Reset warnings.
    sys.stderr = sys.__stderr__

    # Load PC_USR dataset.
    with open('../../../_datasets/pc_usr_data.json', 'r') as file:
        pc_usr_data = json.load(file)

    # File to save dialogue ratings.
    file_path = 'qwen14b_dialogue_ratings.json'

    # Check if file exists.
    if os.path.exists(file_path):
        # Empty file.
        if os.stat(file_path).st_size == 0:
            formatted_dialogues = []
        else:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                formatted_dialogues = data.get('dialogues', [])
    else:
        with open(file_path, 'w') as json_file:
            json.dump({"dialogues": []}, json_file)
        formatted_dialogues = []


    # Iterate over PC_USR dataset.
    for dialog_id, example in enumerate(tqdm(pc_usr_data, desc="Dialogue ratings progress")):
        context = example["context"].split("\n")
        responses = example["responses"]

        for response_data in responses:
            response = response_data["response"].split("\n")[0]

            if response is None:
                # this is a conversation data point, not a turn data point
                continue

            response = [response]
            prompt = create_prompt(context, response)

            input_tokens = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
            output_token_yes = tokenizer.encode("Yes", add_special_tokens=False, return_tensors="pt")[0]
            output_token_no = tokenizer.encode("No", add_special_tokens=False, return_tensors="pt")[0]

            output_tokens = [output_token_yes, output_token_no]

            output_tokens_prob = []

            # Predict with the given model
            with torch.no_grad():
                outputs = model.generate(input_tokens, max_new_tokens=1, output_logits=True, return_dict_in_generate=True)
                logit_predictions = outputs.logits[0]

            for output_token in output_tokens:
                # Decode output token
                token = tokenizer.decode(output_token)

                # logit_predictions is a tensor of dimensions [1, vocab_size], where each value represents an unnormalized logit for a token in the vocabulary.
                log_probs = torch.nn.functional.log_softmax(logit_predictions, dim=-1)

                # Extract the log probability of the output token
                out_token_log_prob = log_probs[0, output_token]

                # Extract the probability of the output token
                out_token_prob = math.exp(out_token_log_prob)

                output_tokens_prob.append({token: out_token_prob})

                # print("============")
                # print(f"Token: {token}", "log prob: ", out_token_log_prob, 'prob: ', out_token_prob)
                # print("============")


            # Normalized probability as in the paper ("Yes" is our score to considering)
            output_tokens_prob[0]['Yes'] = output_tokens_prob[0]['Yes'] / (output_tokens_prob[0]['Yes'] + output_tokens_prob[1]['No'])
            output_tokens_prob[1]['No'] = 1 - output_tokens_prob[0]['Yes']

            formatted_data = process_list(dialog_id, output_tokens_prob)
            formatted_dialogues.append(formatted_data)

            with open(file_path, 'w') as json_file:
                json.dump({"dialogues": formatted_dialogues}, json_file, indent=4)

            # print(f"Dialogue {i} ratings write successfully!")


if __name__ == '__main__':
    make_inferences()
