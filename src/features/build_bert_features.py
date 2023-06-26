# !pip install transformers
from transformers import BertTokenizer, BertModel
import torch
from make_dataset import *

def get_bert_embeddings(text):
    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Convert tokens to token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Convert token IDs to tensors
    input_tensor = torch.tensor([token_ids])

    # Get the BERT model outputs
    with torch.no_grad():
        outputs = model(input_tensor)

    # Get the word embeddings
    embeddings = outputs.last_hidden_state.squeeze()

    return embeddings
  
# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Get keyword embeddings
keyword1 = "aspiring human resources"
keyword2 = "seeking human resources"

bert_key_embeddings1 = get_bert_embeddings(keyword1).sum(axis=0).unsqueeze(0)
bert_key_embeddings2 = get_bert_embeddings(keyword2).sum(axis=0).unsqueeze(0)

# Get document embeddings
bert_embeddings = []

# Iterate over each string
for title in job_titles:
  title_emb = get_bert_embeddings(title)
  if title_emb.ndim != 2:
    title_emb = title_emb.reshape(1, -1)
    # Append word embeddings
  bert_embeddings.append(np.array(title_emb.sum(axis=0)))

bert_cs1 = cosine_similarity(np.array(bert_key_embeddings1), np.array(bert_embeddings))
bert_cs2 = cosine_similarity(np.array(bert_key_embeddings2), np.array(bert_embeddings))
bert_df1, bert_df2 = df_fit(df, bert_cs1[0], bert_cs2[0])

# Re-ranking with starred candidate Approach 1
star_key1 = keywords[0] + " " + bert_df1.iloc[6]["job_title"].lower()
star_key2 = keywords[1] + " " + bert_df2.iloc[6]["job_title"].lower()

# Get keyword embeddings
star_keyword_vec1 = get_bert_embeddings(star_key1).sum(axis=0).unsqueeze(0)
star_keyword_vec2 = get_bert_embeddings(star_key2).sum(axis=0).unsqueeze(0)

star_cs1 = cosine_similarity(np.array(star_keyword_vec1), bert_embeddings)
star_cs2 = cosine_similarity(np.array(star_keyword_vec2), bert_embeddings)
star_df1, star_df2 = df_fit(df, star_cs1[0], star_cs2[0])

# Re-ranking with starred candidate Approach 2
star_key1 = bert_df1.iloc[6]["job_title"].lower()
star_key2 = bert_df2.iloc[6]["job_title"].lower()

# Get keyword embeddings
star_keyword_vec1 = get_bert_embeddings(star_key1).sum(axis=0).unsqueeze(0)
star_keyword_vec2 = get_bert_embeddings(star_key2).sum(axis=0).unsqueeze(0)

star_cs1 = cosine_similarity(np.array(star_keyword_vec1), bert_embeddings)
star_cs2 = cosine_similarity(np.array(star_keyword_vec2), bert_embeddings)

# Take the average of the starred keywords and original keywords
star_cs1 = (star_cs1 + bert_cs1) / 2
star_cs2 = (star_cs2 + bert_cs2) / 2
star_df1, star_df2 = df_fit(df, star_cs1[0], star_cs2[0])
