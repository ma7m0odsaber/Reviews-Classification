from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
import time

# Set hyperparameters
max_features = 5000  # the most commonly occuring words in the dataset
max_len = 400  # Maximum length of each review
embedding_dim = 50  # Dimensionality of the word embeddings
num_filters = 64  # Number of filters in each convolutional layer
filter_sizes = [3, 4, 5]  # Sizes of the filters in each convolutional layer
hidden_dims = 250  # Number of neurons in the hidden layer
batch_size = 32  # Batch size for training
epochs = 8  # Number of training epochs

# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad the sequences so that they are all the same length
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Define the model
model = Sequential()
model.add(Embedding(max_features, embedding_dim, input_length=max_len))
model.add(Conv1D(num_filters, filter_sizes[0], activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
start_time = time.time()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
end_time = time.time()
# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:',accuracy)
print("training time: %.2f seconds" % (end_time - start_time))
