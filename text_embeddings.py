import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2Model
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np 
import pandas as pd 
import json
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict 
import pickle 


model_label = "mpnet"
model_name = "sentence-transformers/all-mpnet-base-v2"
document_embedding = False

sentence_transformers = ["princeton-nlp/sup-simcse-roberta-large", "bert-base-nli-mean-tokens", "sentence-transformers/gtr-t5-xl", "sentence-transformers/gtr-t5-large", "sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2", "sentence-transformers/all-mpnet-base-v2"]

if "gpt2" in model_name:
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
elif model_name in sentence_transformers:
    model = SentenceTransformer(model_name)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

embedding_list = []

# Load annotations to embed as doclist
for docs in doclist:
    embeddings = None
    
    for doc in tqdm(docs):
        if document_embedding:
            # large context length allows document-embeddings
            trimmed_doc = [x.strip() for x in doc]
            input_doc = " ".join(trimmed_doc)
            if "gte" in model_name:
                batch_dict = tokenizer(input_doc, max_length=8192, padding=True, truncation=True, return_tensors='pt')
                outputs = model(**batch_dict)
                embeddings = outputs.last_hidden_state[:, 0]
            else:
                inputs = tokenizer(input_doc, return_tensors='pt', padding=True, truncation=True)
                with torch.no_grad():
                     outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings.cpu().detach().numpy()
        else:
            sentence_embeddings = []
            # embed individual sentences 
            for sentence in doc:
                if model_name in sentence_transformers:
                    embedding_tensor = model.encode(sentence.strip(), convert_to_tensor=True)
                    sentence_embeddings.append(embedding_tensor.cpu().detach().numpy())
                else:
                    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        output = model(**inputs)
                        sentence_embeddings.append(output.last_hidden_state.mean(dim=1).cpu().detach().numpy())  
            embeddings = np.mean(sentence_embeddings,axis=0)
        embedding_list.append(embeddings)

print(np.array(embedding_list).shape)
