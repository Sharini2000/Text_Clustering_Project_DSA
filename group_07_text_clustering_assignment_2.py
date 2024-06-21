from IPython.display import clear_output

!pip install gutenbergpy nltk contractions
!pip install gensim
clear_output()

"""# Import necessary libraries"""

# Commented out IPython magic to ensure Python compatibility.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.mixture import GaussianMixture
from gutenbergpy.textget import get_text_by_id
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import cohen_kappa_score
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.text import UMAPVisualizer, TSNEVisualizer
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from collections import defaultdict
import gensim
# %matplotlib inline

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

Campian_model=pd.DataFrame({
    'Algorithm Name': [],
    'kappa': [],
    'coherence': [],
    'silhouette_avg': [],
    'Cluster' : [],
})

Campian_model

"""# Function to clean and preprocess the text"""

def preprocess(text):
    # Initialize lemmatizer, stemmer, and stop words
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Convert to lowercase
    text = text.lower()

    # Remove garbage characters, punctuation, and non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize
    words = nltk.word_tokenize(text)

    # Remove numbers, but not words that contain numbers
    words = [word for word in words if not word.isnumeric()]

    # Remove words that are only one character
    words = [word for word in words if len(word) > 1]

    # Remove stopwords, lemmatize, and stem
    words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words if word not in stop_words]

    return ' '.join(words)

"""# Dictionary of book IDs, labels, and authors"""

books = {
    "The Adventures of Sherlock Holmes": (1661, 'a', 'Arthur Conan Doyle'), #Mystery
    "Pride and Prejudice": (1342, 'b', 'Jane Austen'), #Romance
    "Moby-Dick": (2701, 'c', 'Herman Melville'), #Adventure
    "The Republic": (1497, 'd', 'Plato'), #Philosophy
    "The Jungle Book": (35997, 'e', 'Rudyard Kipling') #Children
}

data = []
for title, (gutenberg_id, label, author) in books.items():
    raw_text = get_text_by_id(gutenberg_id)
    text = raw_text.decode('utf-8')
    #text = preprocess(text)

    words = text.split()
    chunks = [' '.join(words[i:i + 150]) for i in range(0, len(words), 150)]
    sampled_chunks = random.sample(chunks, min(200, len(chunks)))

    for chunk in sampled_chunks:
        data.append({'author': author,'label': label,'text': chunk })

"""# Create a DataFrame"""

df = pd.DataFrame(data)
print("Display dhead of the data frame")
print(df.head())
print("--------------------------------------------")
print("Display shape of the data frame")
print(df.shape)

"""# preprocess the text"""

df["text_clean"] = df["text"].apply(lambda x: preprocess(x))
df.head()

"""# Add semantic Column
TextBlob sentiment analysis is a tool that evaluates the sentiment of a piece of text, determining whether it is positive, negative, or neutral. It uses a lexicon of words to determine the sentiment, assigning a polarity score between -1 (negative) and 1 (positive) to the text. It can be useful for understanding the general sentiment of a large amount of text, such as customer reviews or social media posts.
"""

df["sentiment"] = df["text_clean"].apply(lambda x:TextBlob(x).sentiment.polarity)
df.head()

sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)

"""# Length Analysis

## Length Analysis before cleaning
"""

df['word_count'] = df["text"].apply(lambda x: len(str(x).split(" ")))
df['char_count'] = df["text"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
df['sentence_count'] = df["text"].apply(lambda x: len(str(x).split(".")))
df['avg_word_length'] = df['char_count'] / df['word_count']
df['avg_sentence_lenght'] = df['word_count'] / df['sentence_count']
df.head()

"""## Length Analysis After cleaning"""

df['word_count'] = df["text_clean"].apply(lambda x: len(str(x).split(" ")))
df['char_count'] = df["text_clean"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
df['sentence_count'] = df["text_clean"].apply(lambda x: len(str(x).split(".")))
df['avg_word_length'] = df['char_count'] / df['word_count']
df['avg_sentence_lenght'] = df['word_count'] / df['sentence_count']
df.head()

"""# Create a list of words

A list of words is created by using the text_clean.
"""

docs = []
for text in df['text_clean']:
    doc_words = nltk.word_tokenize(text)
    docs.append(doc_words)

from gensim.models import Phrases
# Create bigram model
bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

"""Remove rare and common tokens."""

from gensim.corpora import Dictionary

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5)

"""# Transform Data"""

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]

"""# LDA"""

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""## Train LDA model."""

from gensim.models import LdaModel

# Set training parameters.
num_topics = 10
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

"""## Evaluate Model"""

top_topics = model.top_topics(corpus) #, num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics

new_row = {'Algorithm Name': 'LDA', 'kappa': '-','coherence': avg_topic_coherence,'silhouette_avg':'-', 'Cluster' :num_topics  }
# Create a DataFrame from the new row
new_row_df = pd.DataFrame([new_row])

# Concatenate the DataFrame with the new row
Campian_model = pd.concat([Campian_model, new_row_df], ignore_index=True)

print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)

!pip install pyLDAvis
clear_output()

import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(model, corpus, dictionary=dictionary)
vis

"""# Word-Embedding"""

from gensim.models import Word2Vec

# Tokenize the sentences in the data frame
list_of_sent_train = []

for text in df['text_clean']:
    doc_words = nltk.word_tokenize(text)
    list_of_sent_train.append(doc_words)

# Train Word2Vec model (you can also load a pre-trained model)
word2vec_model = Word2Vec(sentences=list_of_sent_train, vector_size=100, workers=4)

# Function to compute average word vector for a sentence
def compute_avg_word_vector(sentence, model, vector_size):
    sent_vec = np.zeros(vector_size)  # Initialize a vector of zeros with the same size as word vectors
    cnt_words = 0  # Counter for words with a valid vector in the sentence
    for word in sentence:
        try:
            vec = model.wv[word]  # Get the word vector
            sent_vec += vec  # Add the word vector to the sentence vector
            cnt_words += 1  # Increment the counter
        except KeyError:
            continue  # If the word is not in the vocabulary, skip it
    if cnt_words != 0:
        sent_vec /= cnt_words  # Average the sum of word vectors
    return sent_vec

# Initialize a list to hold sentence vectors
sent_vectors = []
# Compute the average word2vec for each sentence in the DataFrame
for text in df['text_clean']:

    sentence = nltk.word_tokenize(text)
    avg_vector = compute_avg_word_vector(sentence, word2vec_model, 100)
    sent_vectors.append(avg_vector)

# Convert the list of sentence vectors to a numpy array
sent_vectors = np.array(sent_vectors)

# Output the shape of the resulting document embeddings
print(sent_vectors.shape)

# Assign the document embeddings to a variable
doc_embeddings = sent_vectors

doc_embeddings.shape

"""## Create BoW CountVectorizer"""

# Create BoW CountVectorizer with English stop words removed
bow_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the training data
df_bow = bow_vectorizer.fit_transform(df['text_clean'])

# Print BoW DataFrame shapes
print("BoW Training DataFrame shape:", df_bow.shape)

bow_vectorizer.vocabulary_.get(u'algorithm')

"""## Create TF-IDF vectorizer"""

# Create TF-IDF vectorizer with English stop words removed
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the training data
df_tfidf = tfidf_vectorizer.fit_transform(df['text_clean'])


# Print TF-IDF DataFrame shapes
print("TF-IDF Training DataFrame shape:", df_tfidf.shape)

"""### Elbow Method"""

def compute_silhouette(source, labels):
    # Only compute the silhouette score if there is more than 1 cluster
    if len(set(labels)) > 1 and len(set(labels)) != len(source):
        return silhouette_score(source, labels)
    else:
        return None

def compute_wcss(source, labels):
    # Initialize the WCSS to 0
    wcss = 0.0
    # Get the unique cluster labels
    unique_labels = np.unique(labels)

    # Iterate over each unique label and compute the WCSS for that cluster
    for label in unique_labels:
        if label == -1:
            continue  # Ignore noise points
        # Get the data points corresponding to the current cluster
        cluster_points = source[labels == label]
        # Compute the centroid of the cluster
        centroid = np.mean(cluster_points, axis=0)
        # Compute the sum of squared distances from each point to the centroid
        wcss += np.sum((cluster_points - centroid) ** 2)

    return wcss

def ElbowMethod(source, name):
    if name == 'KMeans':
        # Using KElbowVisualizer to find the optimal number of clusters
        model = KMeans(init='k-means++', max_iter=300, n_init=10, random_state=0)
        visualizer = KElbowVisualizer(model, k=(1, 11), timings=False)
        visualizer.fit(source)
        visualizer.show()

    elif name == 'DBSCAN':
        eps_values = np.linspace(0.1, 1.0, 10)
        min_samples_values = [5, 10, 15, 20, 25]
        silhouette_scores = []
        x_values = []
        index = 1
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(source)
                score = compute_silhouette(source, labels)
                if score is not None:  # Avoid cases where all points are considered noise
                    silhouette_scores.append(score)
                    x_values.append(index)
                index += 1
        if len(silhouette_scores) > 0 and len(x_values) > 0:
          plt.plot(x_values, silhouette_scores)
          plt.title('Silhouette Scores for DBSCAN')
          plt.xlabel('Index')
          plt.ylabel('Silhouette Score')
          plt.show()
        else:
          print("No valid silhouette scores to plot.")

        return

    elif name == 'hierarchy':
        wcss = []
        for n_clusters in range(2, 11):
            agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
            labels = agglomerative.fit_predict(source)
            wcss.append(compute_wcss(source, labels))
        x_values = list(range(2, 11))

        # Plotting the dendrogram
        Z = linkage(source, method='ward')
        plt.figure(figsize=(12, 8))
        dendrogram(Z)
        plt.axhline(y=35, color='r', linestyle='--')
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Data points')
        plt.ylabel('Distance')
        plt.grid()
        plt.show()

        # Plotting the WCSS to use the elbow method
        plt.plot(x_values, wcss)
        plt.title('Elbow Method for Hierarchical Clustering')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

"""## EM for Word_Embedding"""

def EM(Source, max_components=10):
    n_components = np.arange(1, max_components + 1)
    bics = []
    aics = []

    for n in n_components:
        gmm = GaussianMixture(n, covariance_type='full', random_state=0)
        gmm.fit(Source)
        bics.append(gmm.bic(Source))
        aics.append(gmm.aic(Source))

    best_n_components = n_components[np.argmin(bics)]

    plt.plot(n_components, bics, label='BIC', marker='o')
    plt.plot(n_components, aics, label='AIC', marker='o')
    plt.axvline(x=best_n_components, color='red', linestyle='--', label=f'Optimal n_components={best_n_components}')
    plt.legend(loc='best')
    plt.xlabel("Number of Components")
    plt.ylabel("Score")
    plt.title("BIC and AIC for Gaussian Mixture Model")
    plt.grid(True)
    plt.show()

EM(doc_embeddings)

def compute_coherence(source, labels, cluster_number):
    coherence_scores = []
    for cluster in range(cluster_number):
        cluster_points = source[labels == cluster]
        if len(cluster_points) > 1:
            similarity_matrix = cosine_similarity(cluster_points)
            upper_triangular_values = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            coherence_scores.append(np.mean(upper_triangular_values))
        else:
            # If there's only one point in the cluster, coherence is not well-defined, assign 0
            coherence_scores.append(0)
    return np.mean(coherence_scores)

def EM_function(source, Cluster_number, name):
    gmm = GaussianMixture(n_components=Cluster_number)
    gmm.fit(source)
    labels = gmm.predict(source)

    # Add cluster labels to the DataFrame
    df['EM_' + name + '_cluster'] = labels
    print(df[['text_clean', 'label', 'EM_' + name + '_cluster']])

    # Compute silhouette score
    silhouette_avg = silhouette_score(source, labels)

    # Calculate coherence for each cluster
    coherence_avg = compute_coherence(source, labels, Cluster_number)

    print(f'Silhouette Score for {name}: {silhouette_avg}')
    print(f'Coherence Score for {name}: {coherence_avg}')

    # Plotting
    plt.scatter(source[:, 0], source[:, 1], c=labels, cmap='viridis')
    plt.title(f'EM Clustering: {name}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.show()

    return coherence_avg, silhouette_avg

coherence_avg, silhouette_avg  = 0,0
coherence_avg, silhouette_avg = EM_function (doc_embeddings,6,'Word_Embedding')

kMeanEvaluation = np.asanyarray( df[['label','EM_Word_Embedding_cluster']])
mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
kMeanEvaluation[:, 0] = np.vectorize(mapping.get)(kMeanEvaluation[:, 0])
labels_rater1 = kMeanEvaluation[:, 0]
labels_rater2 = kMeanEvaluation[:, 1]

labels_rater1 = labels_rater1.astype(int)
labels_rater2 = labels_rater2.astype(int)


# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(labels_rater1, labels_rater2)
# New row as a dictionary
new_row = {'Algorithm Name': 'EM_Word_Embedding_cluster', 'kappa': kappa_score, 'coherence': coherence_avg,    'silhouette_avg' : silhouette_avg, 'Cluster': 6}

# Create a DataFrame from the new row
new_row_df = pd.DataFrame([new_row])
Campian_model = pd.concat([Campian_model, new_row_df], ignore_index=True)

print(f"Cohen's Kappa Score: {kappa_score:.4f}")

Campian_model

"""## K-Mean Clustering for Word_Embedding


"""

def Kmeans_Fuction (Source,Cluster_number,name):
  tsne = TSNE(n_components=3, random_state=0)
  doc_embeddings_3d = tsne.fit_transform(Source)
  # Perform KMeans clustering
  num_clusters = Cluster_number

  kmeans = KMeans(n_clusters=Cluster_number, random_state=0)
  kmeans.fit(doc_embeddings_3d)
  labels = kmeans.labels_
  print(len(labels))

  # Calculate silhouette score
  silhouette_avg = silhouette_score(doc_embeddings_3d, labels)
  print(f"Silhouette Score: {silhouette_avg}")


  # Add cluster labels to the original DataFrame
  df['K_mean_'+name+'_cluster'] = labels
  print(df[['text_clean', 'label', 'K_mean_'+name+'_cluster']])
  # Ensure you clear the current figure
  plt.clf()

  # Calculate coherence for each cluster
  coherence_scores = []
  for cluster in range(num_clusters):
      cluster_texts = df[df['K_mean_'+name+'_cluster'] == cluster]['text_clean'].tolist()
      if len(cluster_texts) > 0:
          # Tokenize the text
          texts = [text.split() for text in cluster_texts]
          # Create a dictionary and corpus required for coherence calculation
          dictionary = Dictionary(texts)
          corpus = [dictionary.doc2bow(text) for text in texts]
          # Train LDA model
          lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=1, random_state=0, passes=10)
          # Compute Coherence Score using 'c_v'
          coherence_model = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
          coherence_score = coherence_model.get_coherence()
          coherence_scores.append(coherence_score)
      else:
          coherence_scores.append(None)  # If the cluster is empty

  avg_coherence_score = sum(filter(None, coherence_scores)) / len(coherence_scores)
  print(f"Average Coherence Score: {avg_coherence_score}")

    # Create a silhouette plot
  fig, ax = plt.subplots(figsize=(10, 8))
  y_lower = 10
  for i in range(num_clusters):
      cluster_silhouette_values = silhouette_samples(Source, labels)
      cluster_silhouette_values.sort()
      cluster_size = cluster_silhouette_values.shape[0]
      y_upper = y_lower + cluster_size
      color = plt.cm.nipy_spectral(float(i) / num_clusters)
      ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
      ax.text(-0.1, y_lower + 0.5 * cluster_size, f'Cluster {i}', color=color)
      y_lower = y_upper + 10

  ax.set_title("Silhouette Plot for K-means Clustering")
  ax.set_xlabel("Silhouette coefficient values")
  ax.set_ylabel("Cluster label")
  ax.axvline(x=silhouette_avg, color="red", linestyle="--")
  ax.set_yticks([])
  ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

  # 3D Scatter plot of clusters
  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')
  colors = ['red', 'black', 'blue', 'cyan', 'green', 'magenta', 'yellow', 'purple', 'orange', 'brown']

  for cluster in range(Cluster_number):
      ax.scatter(doc_embeddings_3d[labels == cluster, 0],
                doc_embeddings_3d[labels == cluster, 1],
                doc_embeddings_3d[labels == cluster, 2],
                  s=100,
                  c=colors[cluster % len(colors)],
                  label=f'Cluster {cluster}')
  ax.set_title('TSNE 3D Visualization of Clusters')
  ax.set_xlabel('TSNE Component 1')
  ax.set_ylabel('TSNE Component 2')
  ax.set_zlabel('TSNE Component 3')
  ax.legend()
  plt.show()
  return avg_coherence_score, silhouette_avg

"""### K_mean_Word_Embedding_cluster"""

ElbowMethod(doc_embeddings,'KMeans')

avg_coherence_score, silhouette_avg =0,0
avg_coherence_score, silhouette_avg = Kmeans_Fuction(doc_embeddings,3,'Word_Embedding')

df.groupby(['K_mean_Word_Embedding_cluster'])['text'].count()

"""### Kappa"""

kMeanEvaluation = np.asanyarray( df[['label','K_mean_Word_Embedding_cluster']])
mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
kMeanEvaluation[:, 0] = np.vectorize(mapping.get)(kMeanEvaluation[:, 0])
labels_rater1 = kMeanEvaluation[:, 0]
labels_rater2 = kMeanEvaluation[:, 1]

labels_rater1 = labels_rater1.astype(int)
labels_rater2 = labels_rater2.astype(int)


# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(labels_rater1, labels_rater2)

new_row = {'Algorithm Name': 'K_mean_Word_Embedding_cluster', 'kappa': kappa_score, 'coherence': avg_coherence_score,    'silhouette_avg' : silhouette_avg, 'Cluster': 3}
# Create a DataFrame from the new row
new_row_df = pd.DataFrame([new_row])
Campian_model = pd.concat([Campian_model, new_row_df], ignore_index=True)



print(f"Cohen's Kappa Score: {kappa_score:.4f}")

"""# Hierarchical Clustering Algorithm for Word_Embedding"""

def compute_coherence(source, labels, cluster_number):
    coherence_scores = []
    for cluster in range(cluster_number):
        cluster_points = source[labels == cluster]
        if len(cluster_points) > 1:
            similarity_matrix = cosine_similarity(cluster_points)
            upper_triangular_values = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            coherence_scores.append(np.mean(upper_triangular_values))
        else:
            # If there's only one point in the cluster, coherence is not well-defined, assign 0
            coherence_scores.append(0)
    return np.mean(coherence_scores)

def Hierarchical_Function (Source, Cluster_number, name):
    # TSNE for dimensionality reduction to 3D
    tsne = TSNE(n_components=3, random_state=0)
    doc_embeddings_3d = tsne.fit_transform(Source)

    hierarchy = AgglomerativeClustering(n_clusters=Cluster_number, affinity='euclidean', linkage='ward')
    labels = hierarchy.fit_predict(Source)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(doc_embeddings_3d, labels)
    print(f"Silhouette Score: {silhouette_avg}")

    # Add cluster labels to the original DataFrame
    df['Hierarchical_'+name+'_cluster'] = labels
    print(df[['text_clean', 'label', 'Hierarchical_'+name+'_cluster']])

    # Calculate coherence for each cluster
    coherence_avg = compute_coherence(Source, labels, Cluster_number)
    print(f"Average Coherence Score: {coherence_avg}")

    # Create a silhouette plot
    fig, ax = plt.subplots(figsize=(10, 8))
    y_lower = 10
    for i in range(Cluster_number):
        cluster_silhouette_values = silhouette_samples(doc_embeddings_3d, labels)
        cluster_silhouette_values.sort()
        cluster_size = cluster_silhouette_values.shape[0]
        y_upper = y_lower + cluster_size
        color = plt.cm.nipy_spectral(float(i) / Cluster_number)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.1, y_lower + 0.5 * cluster_size, f'Cluster {i}', color=color)
        y_lower = y_upper + 10

    ax.set_title("Silhouette Plot for Hierarchical Clustering")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 3D Scatter plot of clusters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'black', 'blue', 'cyan', 'green', 'magenta', 'yellow', 'purple', 'orange', 'brown']

    for cluster in range(Cluster_number):
        ax.scatter(doc_embeddings_3d[labels == cluster, 0],
                   doc_embeddings_3d[labels == cluster, 1],
                   doc_embeddings_3d[labels == cluster, 2],
                   s=100,
                   c=colors[cluster % len(colors)],
                   label=f'Cluster {cluster}')
    ax.set_title('TSNE 3D Visualization of Clusters')
    ax.set_xlabel('TSNE Component 1')
    ax.set_ylabel('TSNE Component 2')
    ax.set_zlabel('TSNE Component 3')
    ax.legend()
    plt.show()

    return coherence_avg, silhouette_avg

ElbowMethod(doc_embeddings,'hierarchy')

coherence_avg, silhouette_avg =0,0
coherence_avg, silhouette_avg= Hierarchical_Function(doc_embeddings,5,'Word_Embedding')

"""### kappa"""

kMeanEvaluation = np.asanyarray( df[['label','Hierarchical_Word_Embedding_cluster']])
mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
kMeanEvaluation[:, 0] = np.vectorize(mapping.get)(kMeanEvaluation[:, 0])
labels_rater1 = kMeanEvaluation[:, 0]
labels_rater2 = kMeanEvaluation[:, 1]

labels_rater1 = labels_rater1.astype(int)
labels_rater2 = labels_rater2.astype(int)


# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(labels_rater1, labels_rater2)

new_row = {'Algorithm Name': 'Hierarchical_Word_Embedding_cluster', 'kappa': kappa_score, 'coherence': coherence_avg,    'silhouette_avg' : silhouette_avg, 'Cluster': 6}
# Create a DataFrame from the new row
new_row_df = pd.DataFrame([new_row])
Campian_model = pd.concat([Campian_model, new_row_df], ignore_index=True)


print(f"Cohen's Kappa Score: {kappa_score:.4f}")

"""# DBSCAN Clustering for Word_Embedding"""

ElbowMethod(doc_embeddings,'DBSCAN')

def plot_2D_clusters(X, labels, core_samples_mask, n_clusters_):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"Estimated number of clusters: {n_clusters_}")
    plt.show()

def plot_3D_clusters(doc_embeddings_3d, labels):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'black', 'blue', 'cyan', 'green', 'magenta', 'yellow', 'purple', 'orange', 'brown']

    unique_labels = set(labels)
    for cluster in unique_labels:
        ax.scatter(doc_embeddings_3d[labels == cluster, 0],
                   doc_embeddings_3d[labels == cluster, 1],
                   doc_embeddings_3d[labels == cluster, 2],
                   s=100,
                   c=colors[cluster % len(colors)],
                   label=f'Cluster {cluster}')

    ax.set_title('TSNE 3D Visualization of Clusters')
    ax.set_xlabel('TSNE Component 1')
    ax.set_ylabel('TSNE Component 2')
    ax.set_zlabel('TSNE Component 3')
    ax.legend()
    plt.show()

def plot_silhouette(source, labels, num_clusters, silhouette_avg):
    fig, ax = plt.subplots(figsize=(10, 8))
    y_lower = 10
    cluster_silhouette_values = silhouette_samples(source, labels)

    for i in range(num_clusters):
        ith_cluster_silhouette_values = cluster_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        cluster_size = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + cluster_size
        color = plt.cm.nipy_spectral(float(i) / num_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.1, y_lower + 0.5 * cluster_size, f'Cluster {i}', color=color)
        y_lower = y_upper + 10

    ax.set_title("Silhouette Plot for DBSCAN Clustering")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()

def DBSCAN_Function(Source, eps, min_samples, name):
    tsne = TSNE(n_components=2, random_state=0)
    doc_embeddings_2d = tsne.fit_transform(Source)

    tsne = TSNE(n_components=3, random_state=0)
    doc_embeddings_3d = tsne.fit_transform(Source)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(Source)

    unique_labels = set(labels)
    n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f"Estimated number of clusters: {n_clusters_}")
    print(f"Estimated number of noise points: {n_noise_}")

    if n_clusters_ > 1:
        silhouette_avg = silhouette_score(Source, labels)
        print(f"Silhouette Score: {silhouette_avg}")
    else:
        silhouette_avg = -1  # Invalid value to indicate silhouette score is not applicable
        print("Silhouette Score: Not applicable (only one cluster)")

    df['DBSCAN_' + name + '_cluster'] = labels
    print(df[['text_clean', 'label', 'DBSCAN_' + name + '_cluster']])

    core_samples_mask = np.zeros_like(labels, dtype=bool)
    if hasattr(dbscan, 'core_sample_indices_'):
        core_samples_mask[dbscan.core_sample_indices_] = True

    # Coherence Calculation
    if n_clusters_ > 1:
        processed_texts = df['text_clean'].tolist()

        # Create a dictionary and corpus for coherence calculation
        dictionary = Dictionary(processed_texts)
        corpus = [dictionary.doc2bow(text) for text in processed_texts]

        # Group texts by clusters
        cluster_texts = defaultdict(list)
        for text, label in zip(processed_texts, labels):
            if label != -1:  # Exclude noise points
                cluster_texts[label].append(text)

        # Compute coherence for each cluster
        coherences = []
        for cluster, texts in cluster_texts.items():
            if len(texts) > 1:  # Coherence requires at least 2 documents per cluster
                cm = CoherenceModel(texts=texts, dictionary=dictionary, coherence='c_v')
                coherences.append(cm.get_coherence())

        avg_coherence_score = np.mean(coherences) if coherences else -1
        print(f"Average Coherence Score: {avg_coherence_score}")
    else:
        avg_coherence_score = -1  # Invalid value to indicate coherence score is not applicable
        print("Average Coherence Score: Not applicable (only one cluster)")

    plot_2D_clusters(doc_embeddings_2d, labels, core_samples_mask, n_clusters_)
    plot_3D_clusters(doc_embeddings_3d, labels)
    if n_clusters_ > 1:
        plot_silhouette(Source, labels, n_clusters_, silhouette_avg)
        return avg_coherence_score, silhouette_avg
    else:
        print("Silhouette Plot: Not applicable (only one cluster)")
        return 0,0

avg_coherence_score, silhouette_avg =0,0
avg_coherence_score, silhouette_avg  = DBSCAN_Function(doc_embeddings, 0.58, 5, 'Word_Embedding')

kMeanEvaluation = np.asanyarray( df[['label','DBSCAN_Word_Embedding_cluster']])
mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
kMeanEvaluation[:, 0] = np.vectorize(mapping.get)(kMeanEvaluation[:, 0])
labels_rater1 = kMeanEvaluation[:, 0]
labels_rater2 = kMeanEvaluation[:, 1]

labels_rater1 = labels_rater1.astype(int)
labels_rater2 = labels_rater2.astype(int)


# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(labels_rater1, labels_rater2)

new_row = {'Algorithm Name': 'DBSCAN_Word_Embedding_cluster', 'kappa': kappa_score, 'coherence': avg_coherence_score,    'silhouette_avg' : silhouette_avg, 'Cluster': 1}
# Create a DataFrame from the new row
new_row_df = pd.DataFrame([new_row])
Campian_model = pd.concat([Campian_model, new_row_df], ignore_index=True)

print(f"Cohen's Kappa Score: {kappa_score:.4f}")

"""## EM for bow"""

df_bow_array = df_bow.toarray()

"""## K-means clustering for BoW"""

unique_values = np.unique(df_bow_array)
print("Unique values in df_bow_array:")
print(unique_values)

ElbowMethod(df_bow_array,'KMeans')

avg_coherence_score, silhouette_avg =0,0
avg_coherence_score, silhouette_avg = Kmeans_Fuction(df_bow_array,5,'bow')

"""### kappa"""

kMeanEvaluation = np.asanyarray( df[['label','K_mean_bow_cluster']])
mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
kMeanEvaluation[:, 0] = np.vectorize(mapping.get)(kMeanEvaluation[:, 0])
labels_rater1 = kMeanEvaluation[:, 0]
labels_rater2 = kMeanEvaluation[:, 1]

labels_rater1 = labels_rater1.astype(int)
labels_rater2 = labels_rater2.astype(int)


# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(labels_rater1, labels_rater2)
new_row = {'Algorithm Name': 'K_mean_bow_cluster', 'kappa': kappa_score, 'coherence': avg_coherence_score,    'silhouette_avg' : silhouette_avg, 'Cluster': 4}
# Create a DataFrame from the new row
new_row_df = pd.DataFrame([new_row])
Campian_model = pd.concat([Campian_model, new_row_df], ignore_index=True)

print(f"Cohen's Kappa Score: {kappa_score:.4f}")

"""## HIERARCHICAL clustering for BoW"""

ElbowMethod(df_bow_array,'hierarchy')

avg_coherence_score, silhouette_avg =0,0
avg_coherence_score, silhouette_avg = Hierarchical_Function(df_bow_array,6,'bow')

df.groupby(['Hierarchical_bow_cluster'])['text'].count()

"""## kappa"""

kMeanEvaluation = np.asanyarray( df[['label','Hierarchical_bow_cluster']])
mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
kMeanEvaluation[:, 0] = np.vectorize(mapping.get)(kMeanEvaluation[:, 0])
labels_rater1 = kMeanEvaluation[:, 0]
labels_rater2 = kMeanEvaluation[:, 1]

labels_rater1 = labels_rater1.astype(int)
labels_rater2 = labels_rater2.astype(int)


# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(labels_rater1, labels_rater2)

# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(labels_rater1, labels_rater2)
new_row = {'Algorithm Name': 'Hierarchical_bow_cluster', 'kappa': kappa_score, 'coherence': avg_coherence_score,    'silhouette_avg' : silhouette_avg, 'Cluster': 6}
# Create a DataFrame from the new row
new_row_df = pd.DataFrame([new_row])
Campian_model = pd.concat([Campian_model, new_row_df], ignore_index=True)

print(f"Cohen's Kappa Score: {kappa_score:.4f}")

"""## DBSCAN clustering for BoW"""

ElbowMethod(df_bow,'DBSCAN')

avg_coherence_score, silhouette_avg =0,0
avg_coherence_score, silhouette_avg = DBSCAN_Function(df_bow_array, 0.42, 5, 'bow')

kMeanEvaluation = np.asanyarray( df[['label','DBSCAN_bow_cluster']])
mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
kMeanEvaluation[:, 0] = np.vectorize(mapping.get)(kMeanEvaluation[:, 0])
labels_rater1 = kMeanEvaluation[:, 0]
labels_rater2 = kMeanEvaluation[:, 1]

labels_rater1 = labels_rater1.astype(int)
labels_rater2 = labels_rater2.astype(int)


# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(labels_rater1, labels_rater2)

new_row = {'Algorithm Name': 'DBSCAN_bow_cluster', 'kappa': kappa_score, 'coherence': avg_coherence_score,    'silhouette_avg' : silhouette_avg, 'Cluster': -1}
# Create a DataFrame from the new row
new_row_df = pd.DataFrame([new_row])
Campian_model = pd.concat([Campian_model, new_row_df], ignore_index=True)



print(f"Cohen's Kappa Score: {kappa_score:.4f}")

"""## K-means clustering for TFIDF"""

df_tfidf_array = df_tfidf.toarray()

ElbowMethod(df_tfidf_array,'KMeans')

avg_coherence_score, silhouette_avg =0,0
avg_coherence_score, silhouette_avg = Kmeans_Fuction(df_tfidf_array,6,'TFIDF')

"""### Kappa"""

kMeanEvaluation = np.asanyarray( df[['label','K_mean_TFIDF_cluster']])
mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
kMeanEvaluation[:, 0] = np.vectorize(mapping.get)(kMeanEvaluation[:, 0])
labels_rater1 = kMeanEvaluation[:, 0]
labels_rater2 = kMeanEvaluation[:, 1]

labels_rater1 = labels_rater1.astype(int)
labels_rater2 = labels_rater2.astype(int)


# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(labels_rater1, labels_rater2)
# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(labels_rater1, labels_rater2)
new_row = {'Algorithm Name': 'K_mean_TFIDF_cluster', 'kappa': kappa_score, 'coherence': avg_coherence_score,    'silhouette_avg' : silhouette_avg, 'Cluster': 5}
# Create a DataFrame from the new row
new_row_df = pd.DataFrame([new_row])
Campian_model = pd.concat([Campian_model, new_row_df], ignore_index=True)

print(f"Cohen's Kappa Score: {kappa_score:.4f}")

"""## HIERARCHICAL clustering for TF-IDF"""

ElbowMethod(df_tfidf_array,'hierarchy')

avg_coherence_score, silhouette_avg =0,0
avg_coherence_score, silhouette_avg = Hierarchical_Function(df_tfidf_array,5,'TFIDF')

"""### kappa"""

kMeanEvaluation = np.asanyarray( df[['label','Hierarchical_TFIDF_cluster']])
mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
kMeanEvaluation[:, 0] = np.vectorize(mapping.get)(kMeanEvaluation[:, 0])
labels_rater1 = kMeanEvaluation[:, 0]
labels_rater2 = kMeanEvaluation[:, 1]

labels_rater1 = labels_rater1.astype(int)
labels_rater2 = labels_rater2.astype(int)


# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(labels_rater1, labels_rater2)

# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(labels_rater1, labels_rater2)
new_row = {'Algorithm Name': 'Hierarchical_TFIDF_cluster', 'kappa': kappa_score, 'coherence': avg_coherence_score,    'silhouette_avg' : silhouette_avg, 'Cluster': 5}
# Create a DataFrame from the new row
new_row_df = pd.DataFrame([new_row])
Campian_model = pd.concat([Campian_model, new_row_df], ignore_index=True)
print(f"Cohen's Kappa Score: {kappa_score:.4f}")

"""## DBSCAN clustering for TFIDF"""

avg_coherence_score, silhouette_avg =0,0
avg_coherence_score, silhouette_avg = DBSCAN_Function(df_tfidf_array, 0.42, 5, 'TFIDF')

kMeanEvaluation = np.asanyarray( df[['label','DBSCAN_TFIDF_cluster']])
mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
kMeanEvaluation[:, 0] = np.vectorize(mapping.get)(kMeanEvaluation[:, 0])
labels_rater1 = kMeanEvaluation[:, 0]
labels_rater2 = kMeanEvaluation[:, 1]

labels_rater1 = labels_rater1.astype(int)
labels_rater2 = labels_rater2.astype(int)


# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(labels_rater1, labels_rater2)

# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(labels_rater1, labels_rater2)
new_row = {'Algorithm Name': 'DBSCAN_TFIDF_cluster', 'kappa': kappa_score, 'coherence': avg_coherence_score,    'silhouette_avg' : silhouette_avg, 'Cluster': -1}
# Create a DataFrame from the new row
new_row_df = pd.DataFrame([new_row])
Campian_model = pd.concat([Campian_model, new_row_df], ignore_index=True)
print(f"Cohen's Kappa Score: {kappa_score:.4f}")

"""# Bert"""

pip install transformers[torch]

!pip install scikit-learn
!pip install umap-learn
clear_output()

from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling
import transformers
from torch.utils.data import Dataset
import torch
from transformers import BertModel
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import seaborn as sns

from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to generate BERT embeddings
def get_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return embeddings

# Generate embeddings for your data
df['embeddings'] = df['text_clean'].apply(lambda x: get_bert_embeddings(x, tokenizer, model))

# Prepare embeddings for clustering
embeddings = np.vstack(df['embeddings'].values)

# Reduce dimensions with UMAP
umap_embeddings = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine').fit_transform(embeddings)

# Cluster with AgglomerativeClustering
clusterer = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster_labels = clusterer.fit_predict(umap_embeddings)

# Add cluster labels to the DataFrame
df['Bert_cluster'] = cluster_labels

df['Bert_cluster']

import umap
# Plot the clusters
plt.figure(figsize=(15, 15))  # Increase the figure size
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=df['Bert_cluster'], cmap='Spectral', s=20)  # Increase the point size
plt.colorbar(boundaries=np.arange(df['Bert_cluster'].max() + 2) - 0.5).set_ticks(np.arange(df['Bert_cluster'].max() + 1))
plt.title('UMAP projection of the clusters', fontsize=20)  # Increase the title font size
plt.xlabel('UMAP 1', fontsize=15)  # Add x-axis label
plt.ylabel('UMAP 2', fontsize=15)  # Add y-axis label
plt.show()

from sklearn.metrics import silhouette_score, cohen_kappa_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Silhouette Score
valid_clusters = [label for label in np.unique(cluster_labels) if list(cluster_labels).count(label) > 1]
valid_labels = [label for label in cluster_labels if label in valid_clusters]

if len(valid_labels) > 1:
    silhouette_avg = silhouette_score(umap_embeddings, cluster_labels)
else:
    silhouette_avg = -1  # If valid labels are not sufficient for silhouette score
print(f'Silhouette Score: {silhouette_avg}')

# Assuming df has a 'label' column with true labels for Cohen's Kappa
# Replace 'Bert_cluster' with 'cluster' after clustering
if 'label' in df.columns:
    bertEvaluation = np.vstack((df['label'].values, df['Bert_cluster'].values)).T
    mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    labels_rater1 = np.vectorize(mapping.get)(bertEvaluation[:, 0])
    labels_rater2 = bertEvaluation[:, 1]

    labels_rater1 = labels_rater1.astype(int)
    labels_rater2 = labels_rater2.astype(int)

    # Calculate Cohen's Kappa
    kappa_score = cohen_kappa_score(labels_rater1, labels_rater2)
    print(f'kappa Score: {kappa_score}')
else:
    kappa_score = None  # If true labels are not available
# Topic Coherence
def calculate_coherence(texts, labels):
    vectorizer = CountVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    similarity_matrix = cosine_similarity(vectors)
    label_set = np.unique(labels)
    coherence = 0.0
    for label in label_set:
        indices = [i for i, l in enumerate(labels) if l == label]
        if len(indices) > 1:
            sub_similarity_matrix = similarity_matrix[indices][:, indices]
            coherence += np.mean(sub_similarity_matrix)
    return coherence / len(label_set)

coherence_score = calculate_coherence(df['text_clean'].values, df['Bert_cluster'])

new_row = {'Algorithm Name': 'Bert_Embedding_Historical_cluster', 'kappa': kappa_score, 'coherence': coherence_score, 'silhouette_avg' : silhouette_avg, 'Cluster': '5'}

# Create a DataFrame from the new row
new_row_df = pd.DataFrame([new_row])

# Concatenate the DataFrame with the new row
Campian_model = pd.concat([Campian_model, new_row_df], ignore_index=True)

print(f'Coherence Score: {coherence_score}')

"""# Evaluation"""

Campian_model

selected_columns = [
    'author', 'text', 'label', 'EM_Word_Embedding_cluster',
    'K_mean_Word_Embedding_cluster', 'Hierarchical_Word_Embedding_cluster',
    'DBSCAN_Word_Embedding_cluster', 'K_mean_bow_cluster',
    'Hierarchical_bow_cluster', 'DBSCAN_bow_cluster',
    'K_mean_TFIDF_cluster', 'Hierarchical_TFIDF_cluster',
    'DBSCAN_TFIDF_cluster',
    'Bert_cluster',
]

# Displaying the selected columns
selected_df = df[selected_columns]
selected_df. head(5)

"""# Graph knowledge"""

# Commented out IPython magic to ensure Python compatibility.
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import Matcher
from spacy.tokens import Span
import networkx as nx
from tqdm import tqdm
pd.set_option('display.max_colwidth', 200)
# %matplotlib inline

# POS tag would not be sufficient for long span relations. We need dependency parsing as well!

def get_entities(sent):
  ## chunk 1
  ent1 = ""
  ent2 = ""

  prv_tok_dep = ""    # dependency tag of previous token in the sentence
  prv_tok_text = ""   # previous token in the sentence

  prefix = ""
  modifier = ""

  #############################################################

  for tok in nlp(sent):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
    if tok.dep_ != "punct":
      # check: token is a compound word or not
      if tok.dep_ == "compound":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text

      # check: token is a modifier or not
      if tok.dep_.endswith("mod") == True:
        modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " "+ tok.text

      ## chunk 3
      if tok.dep_.find("subj") == True:
        ent1 = modifier +" "+ prefix + " "+ tok.text
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""

      ## chunk 4
      if tok.dep_.find("obj") == True:
        ent2 = modifier +" "+ prefix +" "+ tok.text

      ## chunk 5
      # update variables
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text
  #############################################################

  return [ent1.strip(), ent2.strip()]

entity_pairs = []

for i in tqdm(df['text_clean']):
  entity_pairs.append(get_entities(i))

def get_relation(sent):
    doc = nlp(sent)

    # Matcher class object
    matcher = Matcher(nlp.vocab)

    # Define the pattern
    pattern = [{'DEP':'ROOT'},
               {'DEP':'prep', 'OP':"?"},
               {'DEP':'agent', 'OP':"?"},
               {'POS':'ADJ', 'OP':"?"}]

    matcher.add("matching_1", [pattern])  # Update this line

    matches = matcher(doc)
    k = len(matches) - 1

    if k >= 0:  # Ensure there is at least one match
        span = doc[matches[k][1]:matches[k][2]]
        return span.text
    else:
        return ""

relations = [get_relation(i) for i in tqdm(df['text_clean'])]

# Let's build the KG
# extract subject
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]

kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})

# create a directed-graph from a dataframe
G=nx.from_pandas_edgelist(kg_df, "source", "target",
                          edge_attr=True, create_using=nx.MultiDiGraph())

pos = nx.spring_layout(G)  # Positioning the nodes using the spring layout
plt.figure(figsize=(10, 8))

# Draw the nodes and edges
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='k', font_size=15, font_weight='bold', arrows=True)

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12)

plt.title('Directed Graph from DataFrame')
plt.show()

# Let's filter by only important relations: written by
G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="see"], "source", "target",
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5)
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()