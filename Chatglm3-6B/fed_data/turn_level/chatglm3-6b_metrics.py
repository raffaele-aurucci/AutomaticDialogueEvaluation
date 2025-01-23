import pandas as pd
import json
from scipy.stats import pearsonr, spearmanr, kendalltau

# Read datasets.
with open('../../../_datasets/fed_data.json', 'r') as file:
    df_human = json.load(file)

df_chatglm3 = pd.read_json('chatglm3-6b_dialogue_ratings.json')

# Annotations.
human_annotations = [
    sum(example["annotations"]["Overall"]) / len(example["annotations"]["Overall"])
    for example in df_human
    if example.get("response") is not None
]

predicted_annotations = [dialogue['yes'] for dialogue in df_chatglm3['dialogues']]

# Metrics.
pearson_correlation, _ = pearsonr(human_annotations, predicted_annotations)
spearman_correlation, _ = spearmanr(human_annotations, predicted_annotations)
kendall_tau_correlation, _ = kendalltau(human_annotations, predicted_annotations)

print(f'pearson_correlation: {pearson_correlation}')
print(f'spearman_correlation: {spearman_correlation}')
print(f'kendall_tau_correlation: {kendall_tau_correlation}')

# Write metrics in JSON file.
dialogues = []
for i in range(len(human_annotations)):
    dialogues.append({
        'dialogue_id': i,
        'human_annotation': human_annotations[i],
        'predicted_annotation': predicted_annotations[i],
    })

metrics = {
    'pearson_correlation': pearson_correlation,
    'spearman_correlation': spearman_correlation,
    'kendall_tau_correlation': kendall_tau_correlation,
}

final_data = {
    "dialogues": dialogues,
    "metrics": metrics
}

file_path = 'chatglm3-6b_dialogue_metrics.json'

with open(file_path, 'w') as json_file:
    json.dump(final_data, json_file, indent=4)
