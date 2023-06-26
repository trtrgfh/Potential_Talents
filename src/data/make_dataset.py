
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("/content/potential-talents.csv")

job_titles = df["job_title"].to_list()
job_titles = [string.lower() for string in job_titles]
keywords = ["aspiring human resources", "seeking human resources"]

combined_list = keywords + job_titles

def df_fit(df, cos_similarity1, cos_similarity2):
  res_df1 = df.copy()
  res_df2 = df.copy()

  res_df1['fit'] = res_df1['fit'].fillna(pd.Series(cos_similarity1))
  res_df2['fit'] = res_df2['fit'].fillna(pd.Series(cos_similarity2))
  res_df1 = res_df1.sort_values(by=['fit'], ascending=False)
  res_df2 = res_df2.sort_values(by=['fit'], ascending=False)

  return res_df1, res_df2
