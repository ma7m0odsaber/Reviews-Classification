from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import time

# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Pad the sequences so that they are all the same length
maxlen = 100
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Flatten the input sequences
x_train_flat = x_train.reshape((x_train.shape[0], maxlen * 1))
x_test_flat = x_test.reshape((x_test.shape[0], maxlen * 1))

# Define and train the Logistic Regression model

lr_model = LogisticRegression(solver='lbfgs')
start_time = time.time()
lr_model.fit(x_train_flat, y_train)
end_time = time.time()

# Evaluate the Logistic Regression model on the test set
lr_pred = lr_model.predict(x_test_flat)
lr_acc = accuracy_score(y_test, lr_pred)
print(" accuracy: %.2f%%" % (lr_acc * 100))
print(" training time: %.2f seconds" % (end_time - start_time))
