import json
import tensorflow as tf
import numpy as np


with open('travel_conversations.json', 'r') as file:
    data = json.load(file)
    conversations = data['conversations']

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts([message for conversation in conversations for message in conversation])

input_sequences = []
output_sequences = []

for conversation in conversations:
    input_seq = tokenizer.texts_to_sequences([conversation[0]])[0]
    output_seq = tokenizer.texts_to_sequences([conversation[1]])[0]
    
    input_sequences.append(input_seq)
    output_sequences.append(output_seq)

max_sequence_length = max([len(seq) for seq in input_sequences + output_sequences])

input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_sequence_length, padding='post')


vocab_size = len(tokenizer.word_index) + 1

encoder_inputs = tf.keras.Input(shape=(max_sequence_length,))
encoder_embedding = tf.keras.layers.Embedding(vocab_size, 256, input_length=max_sequence_length)(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.Input(shape=(max_sequence_length,))
decoder_embedding = tf.keras.layers.Embedding(vocab_size, 256, input_length=max_sequence_length)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

decoder_target_sequences = [seq[1:] + [0] for seq in output_sequences]

model.fit([input_sequences, output_sequences], np.array(decoder_target_sequences), epochs=100, batch_size=64)

encoder_model = tf.keras.Model(encoder_inputs, encoder_states)

decoder_state_input_h = tf.keras.Input(shape=(256,))
decoder_state_input_c = tf.keras.Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = tf.keras.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


def generate_response(input_text):
    input_seq = tokenizer.texts_to_sequences([input_text])[0]
    input_seq = pad_sequences([input_seq], maxlen=max_sequence_length, padding='post')
    
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    
    target_seq[0, 0] = tokenizer.word_index['start']

    response = []
    stop_condition = False

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = tokenizer.index_word[sampled_token_index]

        if sampled_word == 'end' or len(response) >= max_sequence_length - 1:
            stop_condition = True
        else:
            response.append(sampled_word)
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]

    return ' '.join(response)

print("Welcome to TravelBuddy! I am here to help you with travel-related information and suggestions. Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("TravelBuddy: Goodbye! Have a great day!")
        break
    response = generate_response(user_input)
    print("TravelBuddy:", response)
