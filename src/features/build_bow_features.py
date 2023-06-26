from make_dataset import *

CountVec = CountVectorizer(ngram_range=(1,1), # to use bigrams ngram_range=(2,2)
                           stop_words='english')
# fit and transform
Count_data = CountVec.fit_transform(combined_list)

cv_keyword_vec = Count_data[:len(keywords)].toarray()
cv_title_vec = Count_data[len(keywords):].toarray()

# Compute cosine similarity between the keyword vectors and job title vectors
cv_cs = cosine_similarity(cv_keyword_vec, cv_title_vec)
cv_df1, cv_df2 = df_fit(df, cv_cs[0], cv_cs[1])

# Re-ranking with starred candidate Approach 1
# Star the 7th candiate
star_key1 = keywords[0] + " " + cv_df1.iloc[6]["job_title"].lower()
star_key2 = keywords[1] + " " + cv_df2.iloc[6]["job_title"].lower()
star_keywords = [star_key1, star_key2]

# Get the vector of new keywords
star_keyword_vec = CountVec.transform(star_keywords)

# Get cos similarity
star_cs = cosine_similarity(star_keyword_vec, cv_title_vec)
star_df1, star_df2 = df_fit(df, star_cs[0], star_cs[1])

# Re-ranking with starred candidate Approach 2
star_key1 = cv_df1.iloc[6]["job_title"].lower()
star_key2 = cv_df2.iloc[6]["job_title"].lower()
star_keywords = [star_key1, star_key2]

# Get the vector of new keywords
star_keyword_vec = CountVec.transform(star_keywords)

# Take the average of the starred keywords and original keywords
star_cs = cosine_similarity(star_keyword_vec, cv_title_vec)
star_cs = (star_cs + cv_cs) / 2
star_df1, star_df2 = df_fit(df, star_cs[0], star_cs[1])
