# !pip install tensorflow_text
# !pip install tf-models-official
# !pip install tensorflow-addons

# from google.colab import drive
# drive.mount('/content/drive')
# path = 'drive/MyDrive/toxicity/'

import os
import re
import shutil
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow_hub as hub
import tensorflow_text as text
import nltk
import json

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GRU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.text import Tokenizer
from wordcloud import WordCloud, STOPWORDS
from official.nlp import optimization
from matplotlib.pyplot import show, plot

tf.get_logger().setLevel('ERROR')

df_train = pd.read_csv('train.csv')

df_train.head()

df_train.shape

test_samples = pd.read_csv('test.csv')
test_labels = pd.read_csv('test_labels.csv')
df_test = pd.merge(test_samples, test_labels, on="id")

df_test.head()

df_test.shape

# removing sample with labels equal to -1
df_test = df_test.loc[df_test['toxic'] >= 0]
df_test.reset_index(inplace=True)
df_test = df_test.drop(columns=['index'])

df_test.head()

df_test.shape

df_train[df_train.columns[2:]].iloc[0]

#NON-TOXIC comment example
df_train.iloc[0]['comment_text']

df_train[df_train.columns[2:]].iloc[6]

#TOXIC comment example
df_train.iloc[6]['comment_text']

toxic_corpus = df_train.loc[df_train['toxic'] == 1]
toxic_corpus = toxic_corpus["comment_text"].tolist()

threat_corpus = df_train.loc[df_train['threat'] == 1]
threat_corpus = threat_corpus["comment_text"].tolist()


print("Toxic comment:")
print()
wordcloud1 = WordCloud(width = 3000, height = 2000, collocations=False, stopwords = STOPWORDS).generate(" ".join(toxic_corpus))
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud1)
plt.axis("off");
plt.show(block=True)

print()
print("threat comment:")
print()
wordcloud1 = WordCloud(width = 3000, height = 2000, collocations=False, stopwords = STOPWORDS).generate(" ".join(threat_corpus))
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud1)
plt.axis("off");
plt.show(block=True)

for label in df_train.columns[2:]:
    print(df_train[label].value_counts(), '\n')

# Get the class distribution for each column
class_distributions = []
for i in range(2, 8):
    class_distributions.append(df_train.iloc[:, i].value_counts())

# Create a combined bar chart
labels = class_distributions[0].index
num_columns = len(class_distributions)
width = 1 / (num_columns + 1)

fig, ax = plt.subplots(figsize=(10, 5))

for i, class_dist in enumerate(class_distributions):
    x = np.arange(len(labels)) + (i + 1) * width
    bars = ax.bar(x, class_dist, width, label=df_train.columns[i+2])

ax.set_ylabel('Number of Examples')
ax.set_xlabel('Classes')
ax.set_title('Class Distribution of Train Set')
ax.set_xticks(x - width * (num_columns / 2))
ax.set_xticklabels(labels)
ax.legend()
plt.show(block=True)

labels = df_train.columns[2:]
# Compute the class distribution for the train set
train_class_distribution = df_train.iloc[:, 2:].sum()

# Compute the class distribution for the test set
test_class_distribution = df_test.iloc[:, 2:].sum()

print('Positive labels distribution in train set in percentage (%)')
print(round(train_class_distribution/df_train.shape[0]*100,2).sort_values(ascending = False))
print()
print(print('Positive labels distribution in test set in percentage (%)'))
print(round(test_class_distribution/df_test.shape[0]*100,2).sort_values(ascending = False))

train_data = [train_class_distribution[label] for label in labels]
test_data = [test_class_distribution[label] for label in labels]

# plot the bar chart
x = np.arange(len(labels))
width = 0.3

fig, ax = plt.subplots(figsize=(10, 5))
train_bars = ax.bar(x - width/2, train_data, width, label='Train')
test_bars = ax.bar(x + width/2, test_data, width, label='Test')

# add labels, title and legend
ax.set_ylabel('Number of examples')
ax.set_title('Label distribution across train and test sets')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# display the plot
plt.show(block=True)

train_class_distribution.sort_values(ascending = False)

print('Distribution among only positive labels in train set in percentage (%)')
print(round(train_class_distribution/train_class_distribution.sum()*100,2).sort_values(ascending = False))
print()
print('Distribution among only positive labels in test set in percentage (%)')
print(round(test_class_distribution/test_class_distribution.sum()*100,2).sort_values(ascending = False))

RE_PATTERNS = {
    ' american ':
        [
            'amerikan'
        ],

    ' adolf ':
        [
            'adolf'
        ],


    ' hitler ':
        [
            'hitler'
        ],

    ' fuck':
        [
            '(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*',
            '(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)',
            ' f[!@#\$%\^\&\*]*u[!@#\$%\^&\*]*k', 'f u u c',
            '(f)(c|[^a-z ])(u|[^a-z ])(k)', r'f\*',
            'feck ', ' fux ', 'f\*\*', 'f**k','fu*k',
            'f\-ing', 'f\.u\.', 'f###', ' fu ', 'f@ck', 'f u c k', 'f uck', 'f ck'
        ],

    ' ass ':
        [
            '[^a-z]ass ', '[^a-z]azz ', 'arrse', ' arse ', '@\$\$',
            '[^a-z]anus', ' a\*s\*s', '[^a-z]ass[^a-z ]',
            'a[@#\$%\^&\*][@#\$%\^&\*]', '[^a-z]anal ', 'a s s','a55', '@$$'
        ],

    ' ass hole ':
        [
            ' a[s|z]*wipe', 'a[s|z]*[w]*h[o|0]+[l]*e', '@\$\$hole', 'a**hole'
        ],

    ' bitch ':
        [
            'b[w]*i[t]*ch', 'b!tch',
            'bi\+ch', 'b!\+ch', '(b)([^a-z]*)(i)([^a-z]*)(t)([^a-z]*)(c)([^a-z]*)(h)',
            'biatch', 'bi\*\*h', 'bytch', 'b i t c h', 'b!tch', 'bi+ch', 'l3itch'
        ],

    ' bastard ':
        [
            'ba[s|z]+t[e|a]+rd'
        ],

    ' trans gender':
        [
            'transgender'
        ],

    ' gay ':
        [
            'gay'
        ],

    ' cock ':
        [
            '[^a-z]cock', 'c0ck', '[^a-z]cok ', 'c0k', '[^a-z]cok[^aeiou]', ' cawk',
            '(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)', 'c o c k'
        ],

    ' dick ':
        [
            ' dick[^aeiou]', 'deek', 'd i c k', 'dik'
        ],

    ' suck ':
        [
            'sucker', '(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)', 'sucks', '5uck', 's u c k'
        ],

    ' cunt ':
        [
            'cunt', 'c u n t'
        ],

    ' bull shit ':
        [
            'bullsh\*t', 'bull\$hit'
        ],

    ' homo sex ual':
        [
            'homosexual'
        ],

    ' jerk ':
        [
            'jerk'
        ],

    ' idiot ':
        [
            'i[d]+io[t]+', '(i)([^a-z ]*)(d)([^a-z ]*)(i)([^a-z ]*)(o)([^a-z ]*)(t)', 'idiots'
                                                                                      'i d i o t'
        ],

    ' dumb ':
        [
            '(d)([^a-z ]*)(u)([^a-z ]*)(m)([^a-z ]*)(b)'
        ],

    ' shit ':
        [
            'shitty', '(s)([^a-z ]*)(h)([^a-z ]*)(i)([^a-z ]*)(t)', 'shite', '\$hit', 's h i t', '$h1t'
        ],

    ' shit hole ':
        [
            'shythole'
        ],

    ' retard ':
        [
            'returd', 'retad', 'retard', 'wiktard', 'wikitud'
        ],

    ' rape ':
        [
            ' raped'
        ],

    ' dumb ass':
        [
            'dumbass', 'dubass'
        ],

    ' ass head':
        [
            'butthead'
        ],

    ' sex ':
        [
            'sexy', 's3x', 'sexuality'
        ],


    ' nigger ':
        [
            'nigger', 'ni[g]+a', ' nigr ', 'negrito', 'niguh', 'n3gr', 'n i g g e r'
        ],

    ' shut the fuck up':
        [
            'stfu', 'st*u'
        ],

    ' pussy ':
        [
            'pussy[^c]', 'pusy', 'pussi[^l]', 'pusses', 'p*ssy'
        ],

    ' faggot ':
        [
            'faggot', ' fa[g]+[s]*[^a-z ]', 'fagot', 'f a g g o t', 'faggit',
            '(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)', 'fau[g]+ot', 'fae[g]+ot',
        ],

    ' mother fucker':
        [
            ' motha ', ' motha f', ' mother f', 'motherucker',
        ],

    ' whore ':
        [
            'wh\*\*\*', 'w h o r e'
        ],
    ' fucking ':
        [
            'f*$%-ing'
        ],
}

def clean_text(text,remove_repeat_text=True, remove_patterns_text=True, is_lower=True):

  if is_lower:
    text=text.lower()

  if remove_patterns_text:
    for target, patterns in RE_PATTERNS.items():
      for pat in patterns:
        text=str(text).replace(pat, target)

  if remove_repeat_text:
    text = re.sub(r'(.)\1{2,}', r'\1', text)

  # Replacing newline characters with spaces
  text = str(text).replace("\n", " ")

  # Removing any non-alphanumeric characters (except spaces)
  text = re.sub(r'[^\w\s]',' ',text)

  # Removing any numbers
  text = re.sub('[0-9]',"",text)

  # Removing any extra spaces
  text = re.sub(" +", " ", text)

  # Removing any non-ASCII characters
  text = re.sub("([^\x00-\x7F])+"," ",text)

  return text

df2_train = df_train.copy()
df2_test = df_test.copy()
df2_train['comment_text']= df_train['comment_text'].apply(lambda x: clean_text(x))
df2_test['comment_text'] = df_test['comment_text'].apply(lambda x: clean_text(x))

df_train.comment_text[0]

df2_train.comment_text[0]

df_test['comment_text'][3]

df2_test['comment_text'][3]

df3_train = df2_train.copy()
df3_test = df2_test.copy()

# Initialize NLTK objects
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define a function to preprocess the text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]

    # Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    preprocessed_text = " ".join(lemmatized_tokens)

    return preprocessed_text

# Apply the preprocessing function to the 'comment_text' column
df3_train['comment_text'] = df2_train['comment_text'].apply(preprocess_text)
df3_test['comment_text'] = df2_test['comment_text'].apply(preprocess_text)

df2_train['comment_text'][3]

df3_train['comment_text'][3]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df3_train['comment_text'].values)

word_index = tokenizer.word_index
NUM_FEATURES = len(word_index)
print("Words in Vocabulary: ",len(word_index))

word_index

list_tokenized_train = tokenizer.texts_to_sequences(df3_train['comment_text'].values)
list_tokenized_test = tokenizer.texts_to_sequences(df3_test['comment_text'].values)

print(list_tokenized_train[:3])

print(list_tokenized_test[:3])

import matplotlib.pyplot as plt

# Count the number of words in each comment
lengths = df3_train['comment_text'].str.split().apply(len)

# Plot the distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(lengths, bins=100)
ax.set_xlabel('Number of words')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of number of words in comments')
plt.show(block=True)

lengths = df3_train['comment_text'].str.split().apply(len)
percentile_98 = np.percentile(lengths, 98)
percentile_98

MAX_LENGTH = 200

X_train = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_train, maxlen=MAX_LENGTH, padding = 'post')
X_test  = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_test, maxlen=MAX_LENGTH, padding = 'post')

X_train

print("Shape train set:", X_train.shape)

X_test

print("Shape test set:", X_test.shape)

y_train = df_train[df_train.columns[2:]].values
y_test = df_test[df_test.columns[2:]].values

y_train

y_train.shape

y_test

y_test.shape

ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(160000)
ds_train = ds_train.batch(32)
ds_train = ds_train.prefetch(16) # helps bottlenecks

# let's see how a batch looks like
batch_X, batch_y = ds_train.as_numpy_iterator().next()

print(batch_X)
print("\n", batch_X.shape)

print(batch_y)
print("\n", batch_y.shape)

train = ds_train.take(int(len(ds_train)*.8))
val = ds_train.skip(int(len(ds_train)*.8)).take(int(len(ds_train)*.2))

test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test = test.cache()
test = test.batch(32)
test = test.prefetch(16) # helps bottlenecks

print("Number of batches in train set:", len(train))
print("Number of batches in validatiom set:", len(val))
print("Number of batches in test set:", len(test))

# Set up EarlyStopping callback
earlystop_callback = EarlyStopping(
    monitor='val_loss',
    patience=0.1,
    verbose=1,
    restore_best_weights=True
)

# Set up ReduceLROnPlateau callback
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=1,
    verbose=1,
)

callbacks = [earlystop_callback, reduce_lr_callback]


#cnn model created , con1D and maxpooling1D since the data is just text not images

tf.keras.backend.clear_session()
model = Sequential()
model.add(Embedding(NUM_FEATURES+1, 128, input_length = MAX_LENGTH))

model.add(Conv1D(filters = 128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size = 3))

model.add(Conv1D(filters = 128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size = 3))

model.add(Conv1D(filters = 128, kernel_size=7, activation='relu'))
model.add(MaxPooling1D(pool_size = 3))
model.add(GlobalMaxPooling1D())

model.add(Dense(6, activation='sigmoid'))

# model.compile(loss='BinaryCrossentropy', optimizer='Adam', metrics = [tfa.metrics.F1Score(num_classes=6, average='macro', threshold=0.5)]) //alternative for F1 metric
# optimizer = tf.keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='BinaryCrossentropy', optimizer = tf.keras.optimizers.Adam(lr = 0.01), metrics = [tfa.metrics.F1Score(num_classes=6, average='micro', threshold=0.5)])

history = model.fit(train, epochs=8, validation_data=val, callbacks = callbacks)
# Plot the loss and validation loss
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show(block=True)

plt.plot(history.history['f1_score'], label='Training F1 Score')
plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
plt.title("CNN Model - Activation: Tanh - Loss: BinaryCrossentropy - Optimizer: RMSProp(lr = 0.1) - Metric: F1 Score (average = micro) - Epoch = 8")
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.show(block=True)
# Save the entire model
model.save("cnn_model.h5")

# Save only the architecture and weights (not recommended for model deployment)
model.save_weights("cnn_model_weights.h5")

# Save the tokenizer configuration
tokenizer_config = tokenizer.to_json()
with open("cnn_model_tokenizer_config.json", "w", encoding="utf-8") as f:
    f.write(tokenizer_config)

'''
tf.keras.backend.clear_session()
model = Sequential()
model.add(Embedding(NUM_FEATURES + 1, 128, input_length=MAX_LENGTH))

model.add(Bidirectional(GRU(128, return_sequences=True, activation='tanh')))
model.add(Bidirectional(GRU(128, return_sequences=True, activation='tanh')))
model.add(Bidirectional(GRU(128, return_sequences=True, activation='tanh')))
model.add(GlobalMaxPooling1D())

model.add(Dense(6, activation='sigmoid'))

model.compile(loss='BinaryCrossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.1),
              metrics=[tfa.metrics.F1Score(num_classes=6, average='macro', threshold=0.5)])

history = model.fit(train, epochs=8, validation_data=val, callbacks=callbacks)

plt.plot(history.history['f1_score'], label='Training F1 Score')
plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
plt.title("Bidirectional GRU Model - Activation: Tanh - Loss: BinaryCrossentropy - Optimizer: RMSProp(lr = 0.1) - Metric: F1 Score (average = macro) - Epoch = 8 - Early Stopping Patience: 0.5")
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.show(block=True)
'''


'''
# #LSTM Model
# #-----------

tf.keras.backend.clear_session()
model = Sequential()
# Create the embedding layer
model.add(Embedding(NUM_FEATURES+1, 128))
# Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(128, activation='tanh', dropout = 0.2)))
# Final layer
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='BinaryCrossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), metrics = [tfa.metrics.F1Score(num_classes=6, average='macro', threshold=0.5)])

model.summary()

history = model.fit(train, epochs=5, validation_data=val, callbacks = callbacks)

# Plot the loss and validation loss
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show(block=True)

# Plot the F1 macro score on the training and validation sets
plt.plot(history.history['f1_score'], label='Training F1')
plt.plot(history.history['val_f1_score'], label='Validation F1')
plt.title("LSTM Baseline Model - Activation: Tanh - Loss: BinaryCrossentropy - Oprimizer: Adam(lr=0.001) - Droupout: 0.2 - Early Stop Patience: 0.2 - Metric: F1 Score (average = macro) - Epoch = 5")
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.show(block=True)

# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show(block=True)




# restore the best weights for the baseline model
# model.load_weights('drive/MyDrive/toxicity_model.h5')

from sklearn.metrics import classification_report
# Predict the labels for all batches in your test dataset
y_pred = []
y_true = []
label_names = df_train.columns[2:]

for X_batch, y_batch in test:
    y_pred_batch = model.predict(X_batch)
    y_pred.extend(y_pred_batch)
    y_true.extend(y_batch)

# Convert the predicted and true labels into numpy arrays
y_pred = np.array(y_pred)
y_true = np.array(y_true)

threshold = 0.35
y_pred_thresh = (y_pred >= threshold).astype(int)

# Compute the classification report
report = classification_report(y_true, y_pred_thresh, target_names=label_names, zero_division = 1)
print(report)
'''

'''
# LSTM with CNN models combined
model = Sequential()
model.add(Embedding(NUM_FEATURES + 1, 128, input_length=MAX_LENGTH))

# Adding CNN layers
model.add(Conv1D(128, 5, activation='tanh'))
model.add(MaxPooling1D(5))

# Adding Bidirectional LSTM layers
model.add(Bidirectional(LSTM(128, return_sequences=True, activation='tanh')))
model.add(Bidirectional(LSTM(128, return_sequences=True, activation='tanh')))

# Global max pooling to reduce the dimensionality
model.add(GlobalMaxPooling1D())

# Dense layers for classification
model.add(Dense(128, activation='tanh'))
model.add(Dense(6, activation='sigmoid'))

# Compile the model with an appropriate optimizer, loss, and metrics
model.compile(loss='Poisson', optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), metrics = [tfa.metrics.F1Score(num_classes=6, average='macro', threshold=0.5)])

# Print the model summary
model.summary()

history = model.fit(train, epochs=5, validation_data=val, callbacks = callbacks)

# Plot the F1 macro score on the training and validation sets
plt.plot(history.history['f1_score'], label='Training F1')
plt.plot(history.history['val_f1_score'], label='Validation F1')
plt.title("LSTM-CNN Model - Activation: Tanh - Loss: Poisson - Oprimizer: Adam(lr=0.001) - Droupout: 0.2 - Early Stop Patience: 0.1 - Metric: F1 Score (average = macro) - Epoch = 5")
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.show(block=True)
'''

'''
# GRU with CNN models combined
model = Sequential()
model.add(Embedding(NUM_FEATURES + 1, 128, input_length=MAX_LENGTH))

# Adding CNN layers
model.add(Conv1D(128, 5, activation='tanh'))
model.add(MaxPooling1D(5))

# Adding Bidirectional GRU layers
model.add(Bidirectional(GRU(128, return_sequences=True, activation='tanh')))
model.add(Bidirectional(GRU(128, return_sequences=True, activation='tanh')))

# Global max pooling to reduce the dimensionality
model.add(GlobalMaxPooling1D())

# Dense layers for classification
model.add(Dense(128, activation='tanh'))
model.add(Dense(6, activation='sigmoid'))

# Compile the model with an appropriate optimizer, loss, and metrics
model.compile(loss='BinaryCrossentropy', optimizer=tf.keras.optimizers.RMSprop(learning_rate = 0.001), metrics = [tfa.metrics.F1Score(num_classes=6, average='macro', threshold=0.5)])

# Print the model summary
model.summary()

history = model.fit(train, epochs=5, validation_data=val, callbacks = callbacks)

# Plot the F1 macro score on the training and validation sets
plt.plot(history.history['f1_score'], label='Training F1')
plt.plot(history.history['val_f1_score'], label='Validation F1')
plt.title("GRU-CNN Model - Activation: Tanh - Loss: BinaryCrossentropy - Oprimizer: RMSprop(lr=0.001) - Droupout: 0.2 - Early Stop Patience: 0.1 - Metric: F1 Score (average = macro) - Epoch = 5")
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.show(block=True)
'''


'''
import gensim.downloader as api
glove = api.load('glove-wiki-gigaword-300')
print("Found %s word vectors." % len(glove.vectors))

def get_embedding_matrix(length_voc, word_index, embedding_dim = 300 ):
    num_tokens = length_voc + 1
    hits = 0
    misses = 0
    oov_embedded = []
    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        if not(word in oov_embedded):
            try:
                embedding_vector = glove.get_vector(word)
                embedding_matrix[i] = embedding_vector
                hits += 1
            except:
                # handling oovs with random embeddings
                embedding_matrix[i] = np.random.uniform(low=-0.05, high=0.05, size=embedding_dim)
                misses += 1
                oov_embedded.append(word)

    return embedding_matrix, hits, misses, oov_embedded

# creation of vocabulary for the training set and extraction of his OOVs.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df3_train.comment_text)
word_index_train = tokenizer.word_index

print(len(word_index_train.keys()))
embedding_matrix_train, hits, misses, oov_embedded = get_embedding_matrix(len(word_index_train.keys()), word_index_train)
#print(oov_embedded)
print("Embedding matrix shape: {}".format(embedding_matrix_train.shape))
print("Converted %d words (%d misses)" % (hits, misses))

from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, TimeDistributed, GRU
import keras

embedding_layer = Embedding(
    input_dim = embedding_matrix_train.shape[0],
    output_dim = 300,
    embeddings_initializer= keras.initializers.Constant(embedding_matrix_train),
    trainable = False,
    mask_zero = True
)



#LSTM Model with Glove Embeddings
#--------------------------------

tf.keras.backend.clear_session()
model_glove = Sequential()
# Create the embedding layer with Glove's embeddings
model_glove.add(embedding_layer)
# Bidirectional LSTM Layer
model_glove.add(Bidirectional(LSTM(128, dropout=0.2, activation='tanh')))
# Final layer
model_glove.add(Dense(6, activation='sigmoid'))

model_glove.compile(loss='BinaryCrossentropy', optimizer='Adam', metrics = [tfa.metrics.F1Score(num_classes=6, average='macro', threshold=0.5)])

model_glove.summary()

history = model_glove.fit(train, epochs=5, validation_data=val, callbacks = callbacks)

# Plot the loss and validation loss over the epochs
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.show(block=True)

# Plot the F1 macro score on the training and validation sets
plt.plot(history.history['f1_score'], label='Training F1')
plt.plot(history.history['val_f1_score'], label='Validation F1')
plt.title("LSTM - Glove Baseline Model - Activation: Tanh - Loss: BinaryCrossentropy - Oprimizer: Adam(lr=0.001) - Droupout: 0.2 - Early Stop Patience: 0.2 - Metric: F1 Score (average = macro) - Epoch = 5")
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.show(block=True)
'''

# restoring the best weights for Glove model
# model_glove.load_weights('drive/MyDrive/toxicity_glove.h5')
'''
from sklearn.metrics import classification_report
# Predict the labels for all batches in your test dataset
y_pred = []
y_true = []
label_names = df_train.columns[2:]

for X_batch, y_batch in test:
    y_pred_batch = model.predict(X_batch)
    y_pred.extend(y_pred_batch)
    y_true.extend(y_batch)

# Convert the predicted and true labels into numpy arrays
y_pred = np.array(y_pred)
y_true = np.array(y_true)

threshold = 0.5
y_pred_thresh = (y_pred >= threshold).astype(int)

# Compute the classification report
report = classification_report(y_true, y_pred_thresh, target_names=label_names, zero_division = 1)
print(report)
model.save('CNN Best one.h5')
'''


'''
bert_model_name = 'bert_en_uncased_L-12_H-768_A-12'

map_name_to_handle = {
    'bert_en_uncased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
}

map_model_to_preprocess = {
    'bert_en_uncased_L-12_H-768_A-12': 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
}

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

text_test = ['this is such an amazing movie!']
text_preprocessed = bert_preprocess_model(text_test)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

bert_model = hub.KerasLayer(tfhub_handle_encoder)

bert_results = bert_model(text_preprocessed)

print(f'Loaded BERT: {tfhub_handle_encoder}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')

def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.5)(net)
  net = tf.keras.layers.Dense(6, activation='sigmoid', name='classifier')(net)
  return tf.keras.Model(text_input, net)

tf.keras.backend.clear_session()
classifier_model = build_classifier_model()

tf.keras.utils.plot_model(classifier_model)

y = df_train[df_train.columns[2:]]
ds= tf.data.Dataset.from_tensor_slices((df_train['comment_text'], y))
ds = ds.cache()
ds = ds.shuffle(160000)
ds = ds.batch(8) #originally 16
ds = ds.prefetch(8) #originally 16

train_ds = ds.take(int(len(ds)*.8))
val_ds = ds.skip(int(len(ds)*.8)).take(int(len(ds)*.2))

y_test = df_test[df_test.columns[2:]]
test= tf.data.Dataset.from_tensor_slices((df_test['comment_text'], y_test))
test = test.cache()
test = test.batch(8)
test = test.prefetch(8)

epochs = 4
steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics = [tfa.metrics.F1Score(num_classes=6, average='macro', threshold=0.5)])

print(f'Training model with {tfhub_handle_encoder}')
checkpoint_filepath = 'drive/MyDrive/tmp_weights.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=epochs,
                               callbacks = [model_checkpoint_callback])

# Plot the loss and validation loss over the epochs
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper right')
# plt.show(block=True)

# Plot the F1 macro score on the training and validation sets
plt.plot(history.history['f1_score'], label='Training F1')
plt.plot(history.history['val_f1_score'], label='Validation F1')
plt.title("BERT Baseline Model - Activation: Sigmoid - Loss: BinaryCrossentropy - Oprimizer: Adamw(lr=3e-5) - Droupout: 0.5 - Metric: F1 Score (average = macro) - Epoch = 4")
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.show(block=True)
classifier_model.save('Bert Baseline.h5')
'''

# restoring the best weights for BERT model
# model_glove.load_weights('drive/MyDrive/toxicity_bert.h5')

from sklearn.metrics import classification_report
# Predict the labels for all batches in your test dataset
y_pred = []
y_true = []
label_names = df_train.columns[2:]

for X_batch, y_batch in test:
    y_pred_batch = model.predict(X_batch)
    y_pred.extend(y_pred_batch)
    y_true.extend(y_batch)

# Convert the predicted and true labels into numpy arrays
y_pred = np.array(y_pred)
y_true = np.array(y_true)

threshold = 0.5
y_pred_thresh = (y_pred >= threshold).astype(int)

# Compute the classification report
report = classification_report(y_true, y_pred_thresh, target_names=label_names, zero_division = 1)
print(report)

'''


# import matplotlib.pyplot as plt

# # define the labels and f1 scores for each model
# labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# f1_scores_model1 = [0.65, 0.40, 0.68, 0.45, 0.64, 0.56]
# f1_scores_model2 = [0.67, 0.38, 0.68, 0.42, 0.64, 0.51]
# f1_scores_model3 = [0.67, 0.42, 0.70, 0.59, 0.70, 0.62]

# # create a list of x values for each label
# x_values = list(range(len(labels)))

# plt.figure(figsize=(12,5))
# # plot the f1 scores for each model
# plt.plot(x_values, f1_scores_model1, 'o-', label='Baseline')
# plt.plot(x_values, f1_scores_model2, 'o-', label='Glove')
# plt.plot(x_values, f1_scores_model3, 'o-', label='BERT')

# # set the x-axis labels to the label names
# plt.xticks(x_values, labels)

# # add a legend
# plt.legend()

# # set the title and axis labels
# plt.title('F1 Score Performance by Label and Model')
# plt.xlabel('Label')
# plt.ylabel('F1 Score')
# plt.legend(bbox_to_anchor=(1.14, 1), loc='upper right')
# # display the plot
# plt.show(block=True)
'''