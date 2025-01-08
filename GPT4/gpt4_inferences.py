import os
from openai import OpenAI
import pandas as pd
import json

from config import API_TOKEN

# Load DSTC9 dataset.
df = pd.read_json('../dstc9_data.json')

# Process the desired output.
def process_list(output_split: list, dialogue_id: int):
    return {
        "id_dialogue": dialogue_id,
        "coherence": int(output_split[0].split('-')[1].strip()),
        "engagingness": int(output_split[1].split('-')[1].strip()),
        "diversity": int(output_split[2].split('-')[1].strip()),
        "informativeness": int(output_split[3].split('-')[1].strip()),
        "overall": int(output_split[4].split('-')[1].strip())
    }


# Dialogue to send to API.
def create_dialogue(context_response: list):
    dialogue = f"""
    ### Dialogues:
    {context_response}

    ## Instruction:
    Rate the coherence, engagingness, diversity, informativeness, and overall quality of the input dialogue on a scale of 1 to 5 and just output the corresponding ratings.

    ### Output Format:
    coherence - x
    engagingness - x
    diversity - x
    informativeness - x
    overall - x

    ### Your Response:
    """
    return dialogue


client = OpenAI(api_key=API_TOKEN, base_url="https://api.gpt4-all.xyz/v1")

# File to save dialogue ratings.
file_path = 'gpt4_dialogue_ratings.json'

# Check if file exists.
if not os.path.exists(file_path):
    with open(file_path, 'w') as json_file:
        json.dump([], json_file)

# Read json file as list of dicts.
formatted_dialogues = pd.read_json(file_path).to_dict(orient='records')

# Iterate over dataset DSTC9.
for i in range(0, 2):

    # Read context and response to DSTC9 dataset.
    context = df['contexts'][i]
    response = df['responses'][i]
    context.append(response)

    # Create formatted dialogue to send to GPT4.
    dialogue = create_dialogue(context)

    # Request to API.
    api_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": dialogue}],
        stream=False,
    )

    # Output of model.
    output = api_response.choices[0].message.content

    # Process output.
    output_split = output.split('\n')
    formatted_data = process_list(output_split, i)
    formatted_dialogues.append(formatted_data)

    with open(file_path, 'w') as json_file:
        json.dump(formatted_dialogues, json_file, indent=4)

    print(f"Dialogue {i} ratings write successfully!")

