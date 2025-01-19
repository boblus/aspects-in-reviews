# Identifying Aspects in Paper Reviews

> **Abstract:** Peer review is central to scientific research, yet it is burdened by the increasing volume of publications. Ensuring the quality of review writing, a crucial component of peer review, is important. One of the key desiderata of high quality reviews is comprehensiveness, which reflects the diversity of aspects considered by reviewers. Understanding these aspects is critical to assessing comprehensiveness. Existing NLP research on aspect analysis largely operates with aspects outlined in review guidelines for major NLP venues, which are usually coarse-grained and lack comprehensiveness. To develop a more comprehensive set of aspects, this work leverages a state-of-the-art large language model to identify aspects from reviews of 350 NLP papers across various venues and time periods. We develop a taxonomy of review aspects with different granularity, and introduce a new dataset of reviews augmented with aspects. Our dataset supports two tasks: predicting the aspects of a paper that should be focused on during a review, and identifying the aspects that a review covers. We perform detailed aspect analysis which provides new community-wide insights into the reviewing process. Furthermore, we show that our proposed aspect set helps LLM-generated review detection. Our work advances the analysis of paper reviews in NLP and contributes to the development of better tools to improve review quality.

## Aspect set construction

### Aspect identification
* Implementation: [notebook-azure.ipynb](notebook-azure.ipynb)
* Pre- and postprocessing: [notebook-processs.ipynb](notebook-processs.ipynb)
* Raw GPT-4o identification results: [annotation - llm.csv](annotation-llm.csv)

### Aspect taxonomy
* Coarsed-grained: [aspects - coarse.csv](aspects-coarse.csv)
* Fine-grained: [aspects - fine.csv](aspects-fine.csv)

### Validity Check
* Validation: [notebook-validity_check_&_aspect_analysis.ipynb](notebook-validity_check_&_aspect_analysis.ipynb)
* Human annotations: [human_annotations/](human_annotations)

### Data sources
* Raw review data from Openreview: [data/](./data)
* Selected papers: [selected_papers/](./selected_papers)
* Preprocessed data (segmented into sentences): [preprocessed/](preprocessed)

## Aspect prediction

### Bag-of-Words
* Paper aspect prediction: [notebook-bow-paper_aspect_prediction.ipynb](notebook-bow-paper_aspect_prediction.ipynb)
* Review aspect prediction: [notebook-bow-review_aspect_prediction.ipynb](notebook-bow-review_aspect_prediction.ipynb)

### BERT-based models
* Paper aspect prediction: [bert-paper_aspect_prediction_training.py](bert-paper_aspect_prediction_training.py)
* Review aspect prediction: [bert-review_aspect_prediction_training.py](bert-review_aspect_prediction_training.py)

### GPT-4o
* Implementation: [notebook-azure.ipynb](notebook-azure.ipynb)

### Evaluation scores
* Paper aspect prediction: [evaluation_scores-paper_aspect_prediction.txt](evaluation_scores-paper_aspect_prediction.txt)
* Review aspect prediction: [evaluation_scores-review_aspect_prediction.txt](evaluation_scores-review_aspect_prediction.txt)

## Practical applications

### Aspect analysis
* Implementation: [notebook-validity_check_&_aspect_analysis.ipynb](notebook-validity_check_&_aspect_analysis.ipynb)

### Review comparison and LLM-generated review detection
* Implementation: [notebook-review_comparison_&_detection.ipynb](notebook-review_comparison_&_detection.ipynb)

## Disclaimer
This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
