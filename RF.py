from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from keras.datasets import imdb
import time

# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

# Convert the sequences back to text
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
x_train_text = [' '.join([reverse_word_index.get(i - 3, '?') for i in sequence]) for sequence in x_train]
x_test_text = [' '.join([reverse_word_index.get(i - 3, '?') for i in sequence]) for sequence in x_test]

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
x_train_vec = vectorizer.fit_transform(x_train_text)
x_test_vec = vectorizer.transform(x_test_text)

# Define and train the decision tree model
dt_model = RandomForestClassifier()
start_time = time.time()
dt_model.fit(x_train_vec, y_train)
end_time = time.time()

# Evaluate the decision tree model on the test set
dt_pred = dt_model.predict(x_test_vec)
dt_acc = accuracy_score(y_test, dt_pred)
print("Decision Tree accuracy: %.2f%%" % (dt_acc * 100))
print(" training time: %.2f seconds" % (end_time - start_time))


