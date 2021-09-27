import pandas as pd
import nltk
from nltk.stem import SnowballStemmer  
from nltk.stem import WordNetLemmatizer

#might be needed depending if nltk installed on local machine or not
#nltk.download()

csv = pd.read_csv("Musical_instruments_reviews.csv", index_col=False)
csv.info()
summary = csv['summary']

print("Printing summary column:\n")
print(summary)


word_tokens = nltk.word_tokenize(str(summary))


print("\nPrinting word tokens:\n")
print(word_tokens)


stemming_array = []
for token in word_tokens:
    stemming_array.append(SnowballStemmer('english').stem(token))

print("\nPrinting stemming array:\n")
print(stemming_array)

lemmatizer = WordNetLemmatizer()

lem_array = []
for token in word_tokens:
    lem_array.append(lemmatizer.lemmatize(token))

print("\nPrinting lemmatizer array:\n")
print(lem_array)

