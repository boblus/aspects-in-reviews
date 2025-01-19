import json
import time
import torch
import random
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import *

device = 'cuda'
seed = 2266

random.seed(seed)
torch.cuda.manual_seed_all(seed)

for type_of_labels in ['coarse', 'fine']:

    column_name = 'COARSE'
    if type_of_labels == 'fine':
        column_name = 'FINE'

    category_to_aspect = pd.read_csv(f'aspects - {type_of_labels}.csv')
    aspect_to_category = defaultdict(set)
    for i in range(len(category_to_aspect)):
        aspect_to_category[category_to_aspect['LLM annotation'].to_list()[i]].add(category_to_aspect[column_name].to_list()[i])
    
    categories = ['-']
    categories.extend(category_to_aspect[column_name].to_list())
    categories = sorted(list(set(categories)))

    for source in ['ReviewCritique', 'llm_generated_reviews_emnlp23_liang24', 'llm_generated_reviews_emnlp23_ours', 'llm_generated_reviews_iclr24_liang24', 'llm_generated_reviews_iclr24_ours']:
        with open(f'preprocessed/preprocessed-{source}.json') as file:
            data = json.loads(file.read())

        number_of_data = len(data)
    
        run_id = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        model_id = {'coarse': '20250102_125421', 'fine': '20250102_135556'}[type_of_labels]
        model = torch.load(f'model_checkpoints/{model_id}.pth').to(device)
        tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
        
        model.eval()
        outputs = defaultdict(dict)
        with tqdm(total=len(data)) as t:
            for paper_id in data:
                for reviewer_id in data[paper_id]:
                    output = defaultdict()
                    for _, sentence in data[paper_id][reviewer_id].items():
                        input = tokenizer(sentence, max_length=512, return_tensors='pt')
                        input = {k: v.to(device) for k, v in input.items()}
        
                        with torch.no_grad():
                            model_output = model(**input)
                        logits = model_output.logits
                        prediction = torch.sigmoid(logits) > 0.5
                        prediction = prediction.cpu().numpy().squeeze()
                        
                        result = []
                        for i in range(len(prediction)):
                            if prediction[i] == 1:
                                result.append(categories[i])
                        output[len(output)] = result
                    
                    outputs[paper_id][reviewer_id] = output
                t.update(1)
        
                with open(f'results/agr_detection-{run_id}.json', 'w') as file:
                    json.dump(outputs, file, ensure_ascii=False, indent=4)

        with open('config-inference.txt', 'a') as file:
            file.write(f'{run_id}\t{source}\t{number_of_data}\t{type_of_labels}\n')

        del model
        torch.cuda.empty_cache()
