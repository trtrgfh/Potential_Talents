
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from gensim.models import Word2Vec as w2v
from sklearn.decomposition import PCA

sw = stopwords.words('english')
# Tokenize each job title
word2vec_tokens = [word_tokenize(job_title) for job_title in job_titles]

# Remove stop words and puntuations
word2vec_clean = []
for line in word2vec_tokens:
  tokens = []
  for word in line:
    if (word not in sw and word not in string.punctuation):
        tokens.append(word)
  word2vec_clean.append(tokens)

w = w2v(
    word2vec_clean,
    min_count=1,
    sg = 1,
    window=5
)

# Create word embedding
word2vec_df = (
    pd.DataFrame(
        [w.wv.get_vector(str(n)) for n in w.wv.key_to_index],
        index = w.wv.key_to_index
    )
)

# Get document embedding
word2vec_keyword_vec1 = np.zeros((1, 100))
word2vec_keyword_vec2 = np.zeros((1, 100))
word2vec_title_vec = np.zeros((104, 100))

for keyword in ["aspiring", "human", "resources"]:
  word2vec_keyword_vec1 += w.wv.get_vector(keyword)

for keyword in ["seeking", "human", "resources"]:
  word2vec_keyword_vec2 += w.wv.get_vector(keyword)

for i in range(len(word2vec_clean)):
  for word in word2vec_clean[i]:
    word2vec_title_vec[i] += w.wv.get_vector(word)

word2vec_cs1 = cosine_similarity(word2vec_keyword_vec1, word2vec_title_vec)
word2vec_cs2 = cosine_similarity(word2vec_keyword_vec2, word2vec_title_vec)
word2vec_df1, word2vec_df2 = df_fit(df, word2vec_cs1[0], word2vec_cs2[0])

# Re-ranking with starred candidate Approach 1
# Star the 7th candiate
star_key1 = keywords[0] + " " + word2vec_df1.iloc[6]["job_title"].lower()
star_key2 = keywords[1] + " " + word2vec_df2.iloc[6]["job_title"].lower()
star_key1 = star_key1.split()
star_key2 = star_key2.split()

# Get document embedding
star_keyword_vec1 = np.zeros((1, 100))
star_keyword_vec2 = np.zeros((1, 100))

for keyword in star_key1:
  if keyword not in sw:
    star_keyword_vec1 += w.wv.get_vector(keyword)

for keyword in star_key2:
  if keyword not in sw:
    star_keyword_vec2 += w.wv.get_vector(keyword)

star_cs1 = cosine_similarity(star_keyword_vec1, word2vec_title_vec)
star_cs2 = cosine_similarity(star_keyword_vec2, word2vec_title_vec)
star_df1, star_df2 = df_fit(df, star_cs1[0], star_cs2[0])

# Re-ranking with starred candidate Approach 1
star_key1 = word2vec_df1.iloc[6]["job_title"].lower()
star_key2 = word2vec_df2.iloc[6]["job_title"].lower()
star_key1 = star_key1.split()
star_key2 = star_key2.split()

# Get document embedding
star_keyword_vec1 = np.zeros((1, 100))
star_keyword_vec2 = np.zeros((1, 100))

for keyword in star_key1:
  if keyword not in sw:
    star_keyword_vec1 += w.wv.get_vector(keyword)

for keyword in star_key2:
  if keyword not in sw:
    star_keyword_vec2 += w.wv.get_vector(keyword)

star_cs1 = cosine_similarity(star_keyword_vec1, word2vec_title_vec)
star_cs2 = cosine_similarity(star_keyword_vec2, word2vec_title_vec)

# Take the average of the starred keywords and original keywords
star_cs1 = (star_cs1 + word2vec_cs1) / 2
star_cs2 = (star_cs2 + word2vec_cs2) / 2
star_df1, star_df2 = df_fit(df, star_cs1[0], star_cs2[0])
