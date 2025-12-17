# hate-speech-detection-using-bert

# Hate Speech and Toxic Comment Classification using BERT

This repository contains the implementation of a deep learning model for detecting toxic comments in text using a transformer-based architecture. The project focuses on binary toxic comment classification and is developed as part of a Deep Learning Applications coursework.

## Problem Description
Online platforms contain large volumes of user-generated content, including toxic, abusive, and hateful comments. Manual moderation is not scalable, motivating the need for automated systems that can detect harmful language reliably.

## Dataset
The model is trained and evaluated using the **Jigsaw Toxic Comment Classification dataset**, which consists of Wikipedia discussion comments annotated for multiple types of toxicity. For this project, the original multi-label annotations are converted into a binary classification task:
- 1: Toxic
- 0: Non-toxic

## Model
The project uses **BERT-base-uncased**, a pretrained transformer model. The model is fine-tuned using PyTorch and the Hugging Face Transformers library for binary sequence classification.

## Requirements
All dependencies are listed in the `requirements.txt` file. The environment can be set up using:

```bash
pip install -r requirements.txt```
