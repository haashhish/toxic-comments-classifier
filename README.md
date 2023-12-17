# toxic-comments-classifier
This project focuses on developing and evaluating deep machine learning models for detecting toxic comments in online discussions.

Here's a brief summary of what the code does:

1- Data Loading and Exploration:
The code starts by loading the training and testing datasets from CSV files.
It explores the structure and content of the datasets, including checking the shape and displaying some examples.

2- Text Preprocessing:
Text cleaning is performed on the comments, including lowercasing, removing patterns, and removing repeating characters.
NLTK is used for tokenization, stopword removal, and lemmatization.

3- Exploratory Data Analysis (EDA):
Word clouds are created for toxic and threat comments to visualize common words.
Class distribution and positive label distribution in the datasets are analyzed and visualized.

4- Tokenization and Padding:
Tokenization is applied to convert text into sequences of numbers.
Padding is performed to ensure that all sequences have the same length.

5- Model Building:
The code defines and trains several models, including Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM) networks, Bidirectional GRU, and combinations of LSTM with CNN.
Different architectures and configurations are experimented with, such as varying filter sizes in CNN layers and using GloVe embeddings.

6-Model Evaluation:
The F1 score is used as the evaluation metric.
Training history is plotted to visualize the model's performance over epochs.

7- BERT Model Training:
The code loads a BERT model using TensorFlow Hub and builds a classifier on top of it.
The BERT-based model is trained and evaluated.

8- Model Comparison:
The F1 scores of different models (baseline, GloVe, BERT) are compared and visualized.

9- Model Prediction and Evaluation:
The trained models are used to predict labels on the test set, and classification reports are generated.

10- Visualization:
The code includes commented-out sections for visualizing the training progress and comparing different models using a bar chart.

Conclusion:
The code provides a comprehensive approach to text classification, including preprocessing, model building, training, and evaluation.
