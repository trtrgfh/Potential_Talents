from make_dataset import *

tfidf_vec = TfidfVectorizer(use_idf=True,
                        smooth_idf=True,
                        ngram_range=(1,1),stop_words='english')

tfidf_data = tfidf_vec.fit_transform(combined_list)

tfidf_keyword_vec = tfidf_data[:len(keywords)].toarray()
tfidf_title_vec = tfidf_data[len(keywords):].toarray()
print("tfidf_keyword_vec shape: {}, tfidf_title_vec shape: {}".format(cv_keyword_vec.shape, cv_title_vec.shape))

tfidf_cs = cosine_similarity(tfidf_keyword_vec, tfidf_title_vec)
tfidf_df1, tfidf_df2 = df_fit(df, tfidf_cs[0], tfidf_cs[1])

# Re-ranking with starred candidate Approach 1
# Star the 7th candiate
star_key1 = keywords[0] + " " + tfidf_df1.iloc[6]["job_title"].lower()
star_key2 = keywords[1] + " " + tfidf_df2.iloc[6]["job_title"].lower()
star_keywords = [star_key1, star_key2]

# Get the vector of new keywords
star_keyword_vec = tfidf_vec.transform(star_keywords)
star_cs = cosine_similarity(star_keyword_vec, cv_title_vec)
star_df1, star_df2 = df_fit(df, star_cs[0], star_cs[1])

# Re-ranking with starred candidate Approach 1
# Star the 7th candiate
star_key1 = tfidf_df1.iloc[6]["job_title"].lower()
star_key2 = tfidf_df2.iloc[6]["job_title"].lower()
star_keywords = [star_key1, star_key2]

# Get the vector of new keywords
star_keyword_vec = CountVec.transform(star_keywords)
star_cs = cosine_similarity(star_keyword_vec, cv_title_vec)
# Take the average of the starred keywords and original keywords
star_cs = (star_cs + tfidf_cs) / 2
star_df1, star_df2 = df_fit(df, star_cs[0], star_cs[1])
