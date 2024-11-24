# Sentiment Analysis on Reddit Data (Hurricane Helene)

This project implements a sentiment analysis model for Reddit data, specifically related to Hurricane Helene. It utilizes various machine learning techniques, including deep learning and traditional models like logistic regression, random forests, and gradient boosting. The project also uses the Hugging Face `distilbert-base-uncased-emotion` model for emotion classification.

## Table of Contents

- [Overview](#overview)
- [Files and Their Descriptions](#files-and-their-descriptions)
- [Process](#process)
- [Model Citation](#model-citation)
- [Requirements](#requirements)

## Overview

The goal of this project is to analyze and predict sentiments and emotional responses to Reddit posts related to Hurricane Helene. This involves:
1. Cleaning and preprocessing data.
2. Implementing sentiment analysis using machine learning models.
3. Fine-tuning a transformer model.
4. Evaluating and visualizing results.

The project includes both traditional machine learning models and a deep learning-based transformer model for emotion classification.

## Files and Their Descriptions

### Data
- **data/**: This folder contains raw datasets (Reddit posts, comments, and other text data) used for training, testing, and evaluation.

### Scripts
- **combine.py**: Script for combining multiple datasets and cleaning the data (e.g., removing duplicates and irrelevant text).
- **analysistransformers.py**: Script for applying sentiment analysis and emotion classification using transformer models (e.g., fine-tuned `distilbert`).
- **deep_learning.py**: Script for applying deep learning-based models to sentiment analysis (including training and evaluation).
- **deep_learning_balanced.py**: A variation of the `deep_learning.py` script that applies class balancing techniques to improve model performance on imbalanced datasets.
- **gradient_boosting.py**: Script that applies gradient boosting techniques (e.g., XGBoost) to predict sentiment.
- **logistic_regression.py**: Script for sentiment analysis using logistic regression.
- **naive_bayes.py**: Script that applies the Naive Bayes classifier for sentiment analysis.
- **random_forest.py**: Script for training a Random Forest classifier for sentiment analysis.
- **clean_manually_labeled_further.py**: Script to further clean and preprocess manually labeled data for sentiment analysis.

### Results and Visualizations
- **combined.csv**: A CSV file containing the combined dataset after merging multiple raw datasets.
- **combined_clean.csv**: A cleaned version of the `combined.csv` file after removing irrelevant or noisy data.
- **Figure_1.png**, **Figure_2.png**, **Figure_3.png**: Images containing visualizations and results from the analysis (e.g., bar plots of emotion scores).

### Model
- **tokenizer_fine_tuned_distilbert/**: Contains the tokenizer for the fine-tuned `distilbert` model that processes text before feeding it to the model.
- **model_fine_tuned_distilbert/**: Directory containing the fine-tuned `distilbert-base-uncased-emotion` model, specifically trained for emotion classification.

### Other
- **requirements.txt**: A text file listing all the required Python dependencies for the project.

## Process

1. **Data Preprocessing**:
    - Combine multiple raw datasets (`combine.py`).
    - Clean and preprocess text data (`clean_manually_labeled_further.py`).
    - Tokenize the text data (`analysistransformers.py`).

2. **Modeling**:
    - **Traditional Machine Learning Models**: Implement models like Logistic Regression, Naive Bayes, Random Forest, and Gradient Boosting (`logistic_regression.py`, `naive_bayes.py`, `random_forest.py`, `gradient_boosting.py`).
    - **Deep Learning Models**: Implement and train deep learning-based models (`deep_learning.py` and `deep_learning_balanced.py`).
    - **Emotion Classification with Transformer Model**: Fine-tune the pre-trained `distilbert-base-uncased-emotion` model for emotion classification and apply it to the data (`analysistransformers.py`).

3. **Evaluation and Results**:
    - Evaluate model performance and visualize emotion scores (`analysistransformers.py`).
    - Save and analyze results in the `results/` folder.

## Model Citation

The `analysistransformers.py` file used emotion classification by utilizing the fine-tuned version of the `distilbert-base-uncased-emotion` model. This model was developed by
 [Bhadresh Savani](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) and can be accessed at:

```plaintext
@article{savani2020distilbert,
  title={distilbert-base-uncased-emotion},
  author={Bhadresh Savani},
  journal={Hugging Face},
  year={2020},
  url={https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion}
}
```

## Requirements

- Python 3.6 or higher
- Install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Ensure that you have all the required libraries, such as `pandas`, `sklearn`, `transformers`, `torch`, and `seaborn`.

---

By following the instructions above, you can replicate this analysis and sentiment classification on any text data. The process is flexible and can be adjusted for different types of datasets or analysis needs.
