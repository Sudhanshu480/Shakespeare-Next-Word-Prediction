# 🎭 Shakespeare Next Word Prediction using LSTM and GRU
An end-to-end NLP project comparing LSTM and GRU deep learning architectures for next-word prediction on Shakespeare's Hamlet. Includes an interactive Streamlit web app for real-time text generation.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.15-orange" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Streamlit-App-red" alt="Streamlit" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

## Table of Contents
- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [LSTM vs GRU Architecture](#lstm-vs-gru-architecture)
- [Dataset](#dataset)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Model Architectures](#model-architectures)
- [Model Training](#model-training)
- [Model Comparison](#model-comparison)
- [Streamlit Application](#streamlit-application)
- [Example Output](#example-output)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Acknowledgement](#acknowledgement)

## Project Overview
This project implements a Deep Learning-based Language Model capable of predicting the next word in a sentence using Recurrent Neural Networks (RNNs). The models are trained on Shakespeare's *Hamlet* text corpus and learn contextual word relationships to generate Shakespeare-style text.

**The main goals of this project were to:**
* Implement sequence modelling for Natural Language Processing (NLP).
* Compare the performance, complexity, and efficiency of LSTM vs. GRU architectures.
* Build an interactive Streamlit application to demonstrate predictions dynamically.

## Motivation
Traditional RNNs struggle with long-term dependencies due to the vanishing gradient problem. To overcome this limitation, advanced gated architectures were introduced:

| Architecture | Description                                                      |
| ------------ | ---------------------------------------------------------------- |
| LSTM         | Uses memory cells and multiple gates to retain long-term context |
| GRU          | A simplified gated architecture with fewer parameters            |

This project experimentally compares:
* Prediction performance
* Training efficiency
* Model complexity

## LSTM vs GRU Architecture

### LSTM (Long Short-Term Memory)
LSTM networks contain three primary gates to manage the flow of information:

| Gate        | Function                              |
| ----------- | ------------------------------------- |
| Forget Gate | Decides what information to discard   |
| Input Gate  | Decides what new information to store |
| Output Gate | Controls output from memory           |

Advantages:
* Strong long-term memory
* Powerful for language modeling

Limitations:
* Computationally expensive
* More parameters

### GRU (Gated Recurrent Unit)
GRU simplifies the LSTM structure by merging the cell and hidden states and using only two gates:

| Gate        | Function                                     |
| ----------- | -------------------------------------------- |
| Update Gate | Controls how much past information to retain |
| Reset Gate  | Controls how much past information to forget |

Advantages:
* Faster training
* Fewer parameters
* Comparable performance in many NLP tasks

## Dataset
* **Dataset Source:** NLTK Gutenberg Corpus
* **Text Used:** Shakespeare - *Hamlet*

**Dataset Preparation Steps:**
1. Load text using NLTK.
2. Clean and normalize text (lowercasing, punctuation removal).
3. Tokenize words.
4. Generate n-gram sequences.
5. Pad sequences to a fixed length.
6. Convert labels to categorical format (one-hot encoding).

## Data Processing Pipeline

### Step 1 — Tokenization
Text is converted into integer tokens representing the vocabulary index.
* *Example:* `To be or not to be` → `[5, 10, 3, 7, 5, 10]`

### Step 2 — N-Gram Sequence Generation
Sequences are built progressively to help the model learn context.
* `To be`
* `To be or`
* `To be or not`
* `To be or not to`

### Step 3 — Sequence Padding
All sequences are pre-padded to a uniform maximum length to ensure consistent input shapes for the neural network.

## Model Architectures
Both models were built and trained using the `TensorFlow/Keras` Sequential API.

### LSTM Model Architecture
| Layer     | Configuration                |
| --------- | ---------------------------- |
| Embedding | 150 dimensions               |
| LSTM      | 150 units (return sequences) |
| Dropout   | 0.3                          |
| LSTM      | 100 units                    |
| Dropout   | 0.3                          |
| Dense     | Softmax output               |


### GRU Model Architecture
| Layer     | Configuration                |
| --------- | ---------------------------- |
| Embedding | 150 dimensions               |
| GRU       | 150 units (return sequences) |
| Dropout   | 0.2                          |
| GRU       | 100 units                    |
| Dropout   | 0.2                          |
| Dense     | Softmax output               |

## Model Training
**Training Configuration:**
| Parameter  | Value                    |
| ---------- | ------------------------ |
| Loss       | Categorical Crossentropy |
| Optimizer  | Adam                     |
| Epochs     | 100                      |
| Batch Size | 64                       |

The models are optimized to learn the conditional probability: `P(next_word | previous_words)`

## Model Comparison
| Metric                 | LSTM   | GRU        |
| ---------------------- | ------ | ---------- |
| Training Speed         | Slower | Faster     |
| Parameter Count        | Higher | Lower      |
| Memory Capability      | Strong | Moderate   |
| Prediction Performance | Good   | **Better** |

<img width="1274" height="583" alt="image" src="https://github.com/user-attachments/assets/e0b8e91f-15cc-4819-a0df-777b4f609b43" />

*Conclusion:* Based on our experimental results, the **GRU outperformed the LSTM**. Not only did the GRU train faster and require fewer parameters, but it also achieved a higher validation accuracy and maintained a significantly lower validation loss, proving to be the more efficient and effective architecture for this specific dataset.

## Streamlit Application
An interactive Streamlit web application was built to demonstrate the models in real-time.

**Features:**
* Toggle seamlessly between LSTM and GRU models.
* Predict the next word based on a custom seed phrase.
* Display the Top 3 predictions with probability distributions.
* Generate continuous, short Shakespeare-style text sequences.

## Example LSTM (1 word):
<img width="1478" height="743" alt="image" src="https://github.com/user-attachments/assets/fcb6a849-b61b-4909-b720-ef029ff12a6b" />

## Example LSTM (multiple words):
<img width="1583" height="753" alt="image" src="https://github.com/user-attachments/assets/fe8cdf1d-9d24-4692-9727-2d3ea2a4bcdd" />

## Example GRU (1 word):
<img width="1525" height="767" alt="image" src="https://github.com/user-attachments/assets/c647a521-b3ab-4c1f-91f9-e1edae6eaeea" />

## Example GRU (multiple words):
<img width="1564" height="731" alt="image" src="https://github.com/user-attachments/assets/d2159dc3-9e50-4cd1-8622-59d0f7aca592" />


## Project Structure
```text
Shakespeare-Next-Word-Prediction/
│
├── my_app.py                     # Streamlit application script
├── lstm_gru.ipynb                # Jupyter notebook for training and evaluation
│
├── final_next_word_lstm.keras    # Trained LSTM model weights
├── final_next_word_gru.keras     # Trained GRU model weights
├── tokenizer_final.pickle        # Keras tokenizer mapping
│
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

# Installation

Clone the repository
```
git clone https://github.com/yourusername/Shakespeare-Next-Word-Prediction.git
```

Install dependencies
```
pip install -r requirements.txt
```

Run the Streamlit application
```
streamlit run app.py
```


# Technologies Used
* Python
* TensorFlow / Keras
* NLTK
* NumPy
* Streamlit
* Matplotlib


# Future Improvements
Possible extensions include:
* Training on larger Shakespeare corpus
* Implementing **Transformer-based models**
* Adding **temperature sampling**
* Implementing **beam search decoding**
* Deploying on **cloud platforms**


# Acknowledgement
This project was developed with learning references from the course:
**Complete Generative AI Course with LangChain and HuggingFace — by Krish Naik (Udemy)**


# Author
**Sudhanshu Kumar**
