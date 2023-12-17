import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

loaded_model = load_model('cnn_model.h5')

loaded_model.load_weights('cnn_model_weights.h5')

with open("cnn_model_tokenizer_config.json", "r", encoding="utf-8") as f:
    tokenizer_config = f.read()

loaded_tokenizer = tokenizer_from_json(tokenizer_config)

new_sentence = ["FUCKKKK"]

max_length = 200 
preprocessed_sentence = loaded_tokenizer.texts_to_sequences(new_sentence)
preprocessed_sentence = pad_sequences(preprocessed_sentence, maxlen=max_length)

# make predictions
predictions = loaded_model.predict(preprocessed_sentence)

print(predictions)