import os
from openai import OpenAI
import pandas as pd
import json
from tqdm import tqdm

from config import API_TOKEN_GPT


# Process the desired output.
def process_list(dialogue_id: int, output_split: list):
    return {
        "id_dialogue": dialogue_id,
        "coherence": float(output_split[0].split('-')[1].strip()),
        "engagingness": float(output_split[1].split('-')[1].strip()),
        "diversity": float(output_split[2].split('-')[1].strip()),
        "informativeness": float(output_split[3].split('-')[1].strip()),
        "overall": float(output_split[4].split('-')[1].strip())
    }


# Create prompt for dialogue.
def create_prompt(context_response: list):
    prompt = f"""
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
    return prompt


def make_inferences():

    # Load DSTC9 dataset.
    df = pd.read_json('../../../_datasets/dstc9_data.json')

    client = OpenAI(api_key=API_TOKEN_GPT, base_url="https://api.gpt4-all.xyz/v1")

    for i in range(0, 5):
        # File to save dialogue ratings.
        file_path = f'gpt4_dialogue_ratings{i+1}.json'

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

        # Iterate over dataset DSTC9.
        for i in tqdm(range(0, len(df)), desc="Dialogue ratings progress"):

            # Read context and response to DSTC9 dataset.
            context = df['contexts'][i]
            response = df['responses'][i]
            context.append(response)

            # Create formatted dialogue to send to GPT4.
            dialogue = create_prompt(context)

            # Request to API.
            api_response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": dialogue}],
                temperature=0.7,
                top_p=0.95,
                stream=False,
            )

            # Output of model.
            output = api_response.choices[0].message.content

            # Process output.
            output_split = output.split('\n')
            formatted_data = process_list(dialogue_id=i, output_split=output_split)
            formatted_dialogues.append(formatted_data)

            with open(file_path, 'w') as json_file:
                json.dump({"dialogues": formatted_dialogues}, json_file, indent=4)

            # print(f"Dialogue {i} ratings write successfully!")


def make_mean_inferences():
    # Json paths.
    file_paths = [
        './test/gpt4_dialogue_ratings1.json',
        './test/gpt4_dialogue_ratings2.json',
        './test/gpt4_dialogue_ratings3.json',
        './test/gpt4_dialogue_ratings4.json',
        './test/gpt4_dialogue_ratings5.json',
    ]

    aggregated_data = {}

    for file_path in file_paths:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            dialogues = data.get("dialogues", [])
            for dialogue in dialogues:
                dialogue_id = dialogue["id_dialogue"]
                if dialogue_id not in aggregated_data:
                    aggregated_data[dialogue_id] = {"coherence": 0, "engagingness": 0, "diversity": 0, "informativeness": 0, "overall": 0}
                aggregated_data[dialogue_id]["coherence"] += dialogue["coherence"]
                aggregated_data[dialogue_id]["engagingness"] += dialogue["engagingness"]
                aggregated_data[dialogue_id]["diversity"] += dialogue["diversity"]
                aggregated_data[dialogue_id]["informativeness"] += dialogue["informativeness"]
                aggregated_data[dialogue_id]["overall"] += dialogue["overall"]

    # Compute mean of dialogues.
    mean_dialogues = []
    for dialogue_id, values in aggregated_data.items():
        mean_dialogues.append({
            "id_dialogue": dialogue_id,
            "mean_coherence": values["coherence"] / 5,
            "mean_engagingness": values["engagingness"] / 5,
            "mean_diversity": values["diversity"] / 5,
            "mean_informativeness": values["informativeness"] / 5,
            "mean_overall": values["overall"] / 5,
        })

    # Save results.
    output_path = 'gpt4_dialogue_ratings_mean.json'

    with open(output_path, 'w') as output_file:
        json.dump({"dialogues": mean_dialogues}, output_file, indent=4)


if __name__ == '__main__':
    # make_inferences()
    make_mean_inferences()