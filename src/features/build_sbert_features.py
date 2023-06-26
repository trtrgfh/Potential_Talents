
# !pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# Load the SBERT model
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

# Encode the sentences to obtain embeddings
sbert_embeddings = sbert_model.encode(job_titles)

sbert_keyword_vec1 = sbert_model.encode("aspiring human resources").reshape(1, -1)
sbert_keyword_vec2 = sbert_model.encode("seeking human resources").reshape(1, -1)
sbert_cs1 = cosine_similarity(sbert_keyword_vec1, sbert_embeddings)
sbert_cs2 = cosine_similarity(sbert_keyword_vec2, sbert_embeddings)
sbert_df1, sbert_df2 = df_fit(df, sbert_cs1[0], sbert_cs2[0])

# Re-ranking with starred candidate Approach 1
star_key1 = keywords[0] + " " + sbert_df1.iloc[6]["job_title"].lower()
star_key2 = keywords[1] + " " + sbert_df2.iloc[6]["job_title"].lower()

# Get keyword embeddings
star_keyword_vec1 = sbert_model.encode(star_key1).reshape(1, -1)
star_keyword_vec2 = sbert_model.encode(star_key2).reshape(1, -1)

star_cs1 = cosine_similarity(sbert_keyword_vec1, sbert_embeddings)
star_cs2 = cosine_similarity(sbert_keyword_vec2, sbert_embeddings)
star_df1, star_df2 = df_fit(df, star_cs1[0], star_cs2[0])

# Re-ranking with starred candidate Approach 2
# Star the 7th candiate
star_key1 = sbert_df1.iloc[6]["job_title"].lower()
star_key2 = sbert_df2.iloc[6]["job_title"].lower()

# Get keyword embeddings
star_keyword_vec1 = sbert_model.encode(star_key1).reshape(1, -1)
star_keyword_vec2 = sbert_model.encode(star_key2).reshape(1, -1)

star_cs1 = cosine_similarity(sbert_keyword_vec1, sbert_embeddings)
star_cs2 = cosine_similarity(sbert_keyword_vec2, sbert_embeddings)

# Take the average of the starred keywords and original keywords
star_cs1 = (star_cs1 + sbert_cs1) / 2
star_cs2 = (star_cs2 + sbert_cs2) / 2
star_df1, star_df2 = df_fit(df, star_cs1[0], star_cs2[0])
