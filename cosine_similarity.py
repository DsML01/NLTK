import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Frases a serem comparadas
frase1 = "Ele eu correr"
frase2 = "Ele eu fizer alongamentos"

# Tokenização e remoção de stopwords
stop_words = set(stopwords.words('portuguese'))
tokens1 = word_tokenize(frase1)
tokens2 = word_tokenize(frase2)
filtered_tokens1 = [w for w in tokens1 if not w in stop_words]
filtered_tokens2 = [w for w in tokens2 if not w in stop_words]

# Join tokens into strings
sentence1 = " ".join(filtered_tokens1)
sentence2 = " ".join(filtered_tokens2)

# Vetorização
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([sentence1, sentence2])

# Cálculo da similaridade
similarity = cosine_similarity(vectors)[0][1]
print("Similaridade:", similarity)