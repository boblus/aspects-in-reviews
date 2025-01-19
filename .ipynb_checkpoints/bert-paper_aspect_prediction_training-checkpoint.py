import json
import time
import torch
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from utils import *

batch_size = 16
epochs = 10
learning_rate = 3e-5
device = 'cuda'
seed = 2266

random.seed(seed)
torch.cuda.manual_seed_all(seed)

note = 'focal'
if note == 'focal':
    alphas = [0.1, 0.2, 0.3]
    gammas = [1.5, 2.0, 2.5]
else:
    alphas = [None]
    gammas = [None]

for type_of_labels in ['coarse', 'fine']:

    column_name = 'COARSE'
    if type_of_labels == 'fine':
        column_name = 'FINE'
    
    category_to_aspect = pd.read_csv(f'aspects - {type_of_labels}.csv')
    aspect_to_category = defaultdict(set)
    for i in range(len(category_to_aspect)):
        if (type_of_labels == 'fine' and category_to_aspect[column_name].to_list()[i] not in ['Contribution', 'Definition', 'Description', 'Detail', 'Discussion', 'Explanation', 'Interpretation', 'Intuition', 'Justification', 'Motivation', 'Validation', 'Novelty', 'Clarity', 'Confusion', 'Figure', 'Grammar', 'Notation', 'Presentation', 'Table', 'Terminology', 'Typo', 'Related Work', 'Impact', 'Importance', 'Significance']) or (type_of_labels == 'coarse' and category_to_aspect[column_name].to_list()[i] not in ['Contribution', 'Definition/Description/Detail/Discussion/Explanation/Interpretation', 'Intuition/Justification/Motivation/Validation', 'Novelty', 'Presentation', 'Related Work', 'Significance']):
            aspect_to_category[category_to_aspect['LLM annotation'].to_list()[i]].add(category_to_aspect[column_name].to_list()[i])
    
    data = defaultdict()
    for venue in ['iclr20', 'iclr21', 'iclr22', 'iclr23', 'iclr24']:
        with open(f'data/{venue}.json') as file:
            data[venue] = json.loads(file.read())
    
    annotation = pd.read_csv('annotation - llm.csv')
    result = defaultdict(list)
    for venue in ['iclr20', 'iclr21', 'iclr22', 'iclr23', 'iclr24']:
        with open(f'preprocessed/preprocessed-{venue}.json') as file:
            preprocessed = json.loads(file.read())
        for paper_id in preprocessed:
            with open(f'data/papers/{paper_id}.txt') as file:
                paper = file.read()
            aspects = []
            for item in annotation['annotation_1'][(annotation['venue'] == venue) & (annotation['paper_id'] == paper_id)].tolist():
                aspects.extend(merge_synonyms(str(item).replace(' and ', ', ').split(', ')))
            result['venue'].append(venue)
            result['paper_id'].append(paper_id)
            result['abstract'].append(data[venue][paper_id]['Abstract'])
            result['keywords'].append(', '.join(data[venue][paper_id]['Keywords']))
            result['title'].append(data[venue][paper_id]['Title'])
            result['paper'].append(paper)
            result['aspects'].append(list(set(aspects)))
    
    for part in ['title', 'keywords', 'abstract']:
        texts = result[part]
        labels = result['aspects']
        
        categories = []
        for aspect, category in aspect_to_category.items():
            categories.extend(list(category))
        categories = sorted(list(set(categories)))
        labels_one_hot = []
        for item in labels:
            output = np.zeros(len(categories))
            for aspect in item:
                if aspect in aspect_to_category:
                    for category in aspect_to_category[aspect]:
                        output[categories.index(category)] = 1
            labels_one_hot.append(output)
        
        number_of_data = int(len(texts)*0.9)
        train_texts = texts[:number_of_data]
        train_labels = labels_one_hot[:number_of_data]
        eval_texts = texts[number_of_data:]
        eval_labels = labels_one_hot[number_of_data:]
        
        number_of_aspects = len(categories)

        for alpha in alphas:
            for gamma in gammas:
                for pretrained in ['FacebookAI/roberta-base']:
                    experiment_id = time.strftime('%Y%m%d_%H%M%S', time.localtime())
                    train_dataset = MyDataset(train_texts, train_labels)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    eval_dataset = MyDataset(eval_texts, train_labels)
                    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
                    
                    tokenizer = AutoTokenizer.from_pretrained(pretrained)
                    model = AutoModelForSequenceClassification.from_pretrained(pretrained, num_labels=len(categories)).to(device)
                    
                    optimizer = AdamW(model.parameters(), lr=learning_rate, no_deprecation_warning=True)
                    if note == 'focal':
                        criterion = FocalLoss(alpha=alpha, gamma=gamma)
                    else:
                        criterion = torch.nn.BCEWithLogitsLoss()

                    for epoch in range(epochs):
                        model.train()
                        with tqdm(total=len(train_dataloader)) as t:
                            for inputs, targets in train_dataloader:
                                inputs = tokenizer(list(inputs), return_tensors='pt', padding=True, truncation=True, max_length=max_lengths[pretrained])
                                inputs = {k: v.to(device) for k, v in inputs.items()}
                                targets = targets.to(device)
                                
                                outputs = model(**inputs)
                                logits = outputs.logits
                                
                                loss = criterion(logits, targets.float())
                                loss.backward()
                                
                                optimizer.step()
                                optimizer.zero_grad()
                                
                                t.set_postfix({'loss': loss.detach().cpu().numpy()})
                                t.update(1)
                
                        model.eval()
                        actuals_one_hot = []
                        predictions_one_hot = []
                        with torch.no_grad():
                            for inputs, targets in eval_dataloader:
                                inputs = tokenizer(list(inputs), return_tensors='pt', padding=True, truncation=True, max_length=512)
                                inputs = {k: v.to(device) for k, v in inputs.items()}
                                targets = targets.to(device)
                                
                                outputs = model(**inputs)
                                logits = outputs.logits
                                actual = targets > 0.5
                                prediction = torch.sigmoid(logits) > 0.5
                        
                                actuals_one_hot.extend(targets.cpu().numpy())
                                predictions_one_hot.extend(prediction.cpu().numpy())
            
                        actuals, predictions = [], []
                        for item in actuals_one_hot:
                            actual = []
                            for i in range(len(item)):
                                if item[i] == 1:
                                    actual.append(categories[i])
                            actuals.append(actual)
                        
                        for item in predictions_one_hot:
                            prediction = []
                            for i in range(len(item)):
                                if item[i] == 1:
                                    prediction.append(categories[i])
                            predictions.append(prediction)
                        
                        accuracy = accuracy_score(actuals_one_hot, predictions_one_hot)
                        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(actuals_one_hot, predictions_one_hot, average='macro', zero_division=0)
                        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(actuals_one_hot, predictions_one_hot, average='micro', zero_division=0)
                        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(actuals_one_hot, predictions_one_hot, average='weighted', zero_division=0)
                        similarities = calculate_jaccard_similarity_for_lists(actuals, predictions)
                        jaccard = sum(similarities) / len(similarities)
        
                        with open('evaluation_scores-paper_aspect_prediction.txt', 'a') as file:
                            file.write(f'{experiment_id}\t{number_of_aspects}\t{part}\t{batch_size}\t{epoch}\t{learning_rate}\t{pretrained}\t{alpha}\t{gamma}\t{device}\t{seed}\t{accuracy}\t{precision_macro}\t{recall_macro}\t{f1_macro}\t{precision_micro}\t{recall_micro}\t{f1_micro}\t{precision_weighted}\t{recall_weighted}\t{f1_weighted}\t{jaccard}\n')
                    
                    torch.save(model, f'model_checkpoints/{experiment_id}.pth')
                    del model
                    torch.cuda.empty_cache()
                    