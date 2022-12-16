import tensorflow.keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentence = ["I am happy to meet my friends. We are planning to go a party.", 
            "I had a bad day at school. i got hurt while playing football"]

# Tokenization
tokenizer=Tokenizer(num_words=10000,oov_token='<OOV>')
tokenizer.fit_on_texts(sentence)
word_index=tokenizer.word_index
sequence=tokenizer.texts_to_sequences(sentence)
padded=pad_sequences(sequence,maxlen=100,padding='post',truncating='post')
model=tensorflow.keras.models.load_model('textemotion.h5')
result=model.perdict(padded)
perdict_class=np.argmax(result,axis=1)
print(perdict_class)   
# Create a word_index dictionary

# Padding the sequence

# Define the model using .h5 file

# Test the model

# Print the result

