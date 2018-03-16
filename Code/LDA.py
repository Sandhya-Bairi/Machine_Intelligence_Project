from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pandas

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')
#print(en_stop)
en_stop.append('rt')
#print(en_stop)
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

colnames = ['text', 'sentiment']
data = pandas.read_csv('feeds.csv', names=colnames, header=0)
tweets = data.text.tolist()

for tweet in tweets:
	tweet = str(tweet)
	#print tweet
#print("------")
# list for tokenized documents in loop
texts = []

# loop through document list
for i in tweets:
    print(i)
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=50, id2word = dictionary, passes=100)

#print("\n".join(ldamodel.print_topics(num_topics=10, num_words=5)))
print(ldamodel.show_topics(num_topics=30, num_words=8))