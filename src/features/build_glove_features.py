
glove_embeddings = {}
# Download embeddings from https://nlp.stanford.edu/projects/glove/
# Load the pre-trained GloVe word vectors
with open("/content/glove.6B.100d.txt", 'r', encoding="utf-8") as f:
  for line in f:
      values = line.split()
      word = values[0]
      vector = np.asarray(values[1:], "float32")
      glove_embeddings[word] = vector

sw = stopwords.words('english')
# Tokenize each job title
glove_tokens = [word_tokenize(job_title) for job_title in job_titles]
# Remove stop words and puntuations
glove_clean = []

for line in word2vec_tokens:
  tokens = []
  for word in line:
    if (word not in sw and word not in string.punctuation):
        tokens.append(word)
  glove_clean.append(tokens)

# Get document embedding
glove_keyword_vec1 = np.zeros((1, 100))
glove_keyword_vec2 = np.zeros((1, 100))
glove_title_vec = np.zeros((104, 100))

for keyword in ["aspiring", "human", "resources"]:
  if keyword in glove_embeddings:
    glove_keyword_vec1 += glove_embeddings[keyword]

for keyword in ["seeking", "human", "resources"]:
  if keyword in glove_embeddings:
    glove_keyword_vec2 += glove_embeddings[keyword]

for i in range(len(glove_clean)):
  for word in glove_clean[i]:
    if word in glove_embeddings:
      glove_title_vec[i] += glove_embeddings[word]
  
glove_cs1 = cosine_similarity(glove_keyword_vec1, glove_title_vec)
glove_cs2 = cosine_similarity(glove_keyword_vec2, glove_title_vec)
glove_df1, glove_df2 = df_fit(df, glove_cs1[0], glove_cs2[0])

# Re-ranking with starred candidate Approach 1
star_key1 = keywords[0] + " " + glove_df1.iloc[6]["job_title"].lower()
star_key2 = keywords[1] + " " + glove_df2.iloc[6]["job_title"].lower()
star_key1 = star_key1.split()
star_key2 = star_key2.split()

# Get document embedding
star_keyword_vec1 = np.zeros((1, 100))
star_keyword_vec2 = np.zeros((1, 100))

for keyword in star_key1:
  if keyword in glove_embeddings:
    star_keyword_vec1 += glove_embeddings[keyword]

for keyword in star_key2:
  if keyword in glove_embeddings:
    star_keyword_vec2 += glove_embeddings[keyword]

star_cs1 = cosine_similarity(star_keyword_vec1, glove_title_vec)
star_cs2 = cosine_similarity(star_keyword_vec2, glove_title_vec)
star_df1, star_df2 = df_fit(df, star_cs1[0], star_cs2[0])

# Re-ranking with starred candidate Approach 2
star_key1 = word2vec_df1.iloc[6]["job_title"].lower()
star_key2 = word2vec_df2.iloc[6]["job_title"].lower()
star_key1 = star_key1.split()
star_key2 = star_key2.split()

# Get document embedding
star_keyword_vec1 = np.zeros((1, 100))
star_keyword_vec2 = np.zeros((1, 100))

for keyword in star_key1:
  if keyword in glove_embeddings:
    star_keyword_vec1 += glove_embeddings[keyword]

for keyword in star_key2:
  if keyword in glove_embeddings:
    star_keyword_vec2 += glove_embeddings[keyword]

star_cs1 = cosine_similarity(star_keyword_vec1, glove_title_vec)
star_cs2 = cosine_similarity(star_keyword_vec2, glove_title_vec)

# Take the average of the starred keywords and original keywords
star_cs1 = (star_cs1 + glove_cs1) / 2
star_cs2 = (star_cs2 + glove_cs2) / 2
star_df1, star_df2 = df_fit(df, star_cs1[0], star_cs2[0])
