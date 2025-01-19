# Identifying Aspects in Paper Reviews

> **Abstract:** Peer review is central to scientific research, yet it is burdened by the increasing volume of publications. Ensuring the quality of review writing, a crucial component of peer review, is important. One of the key desiderata of high quality reviews is comprehensiveness, which reflects the diversity of aspects considered by reviewers. Understanding these aspects is critical to assessing comprehensiveness. Existing NLP research on aspect analysis largely operates with aspects outlined in review guidelines for major NLP venues, which are usually coarse-grained and lack comprehensiveness. To develop a more comprehensive set of aspects, this work leverages a state-of-the-art large language model to identify aspects from reviews of 350 NLP papers across various venues and time periods. We develop a taxonomy of review aspects with different granularity, and introduce a new dataset of reviews augmented with aspects. Our dataset supports two tasks: predicting the aspects of a paper that should be focused on during a review, and identifying the aspects that a review covers. We perform detailed aspect analysis which provides new community-wide insights into the reviewing process. Furthermore, we show that our proposed aspect set helps LLM-generated review detection. Our work advances the analysis of paper reviews in NLP and contributes to the development of better tools to improve review quality.

## Aspect set construction
See [notebook-azure.ipynb](notebook-azure.ipynb) for the identification of aspects using GPT-4o and [notebook-processs.ipynb](notebook-processs.ipynb) for the pre- and postprocessing of the data. See [annotation - llm.csv](annotation-llm.csv) for the GPT-4o identification. See [aspects - coarse.csv](aspects-coarse.csv) and [aspects - fine.csv](aspects-fine.csv) for the resulting aspect sets.

See [notebook-validity_check_&_aspect_analysis.ipynb](notebook-validity_check_&_aspect_analysis.ipynb) for the validity check. See [human_annotations/](human_annotations) for the human annotations.

See [data/](./data) for the raw review data obtained via OpenReview. See [selected_papers/](./selected_papers) for the lists of papers we selected from each venue. See [preprocessed/](preprocessed) for the segmented data.

## Aspect prediction
See [notebook-bow-paper_aspect_prediction.ipynb](notebook-bow-paper_aspect_prediction.ipynb) and [notebook-bow-review_aspect_prediction.ipynb](notebook-bow-review_aspect_prediction.ipynb) for the implementation of BoW. See [bert-paper_aspect_prediction_training.py](bert-paper_aspect_prediction_training.py) and [bert-review_aspect_prediction_training.py](bert-review_aspect_prediction_training.py) for the implementation of BERT. See [notebook-azure.ipynb](notebook-azure.ipynb) for the implementation of GPT-4o.

The evaluation scores are stored in [evaluation_scores-paper_aspect_prediction.txt](evaluation_scores-paper_aspect_prediction.txt) and [evaluation_scores-review_aspect_prediction.txt](evaluation_scores-review_aspect_prediction.txt).

## Practical applications
See [notebook-validity_check_&_aspect_analysis.ipynb](notebook-validity_check_&_aspect_analysis.ipynb) for the aspect analysis. See [notebook-review_comparison_&_detection.ipynb](notebook-review_comparison_&_detection.ipynb) for the review comparison and LLM-generated review detection.

## Disclaimer
This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
