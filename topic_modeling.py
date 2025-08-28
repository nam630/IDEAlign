import os
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
import pickle 
from scipy.stats import spearmanr
from scipy.stats import rankdata
from itertools import combinations
import little_mallet_wrapper as lmw


def hellinger(p, q):
    return (1 / np.sqrt(2)) * np.linalg.norm(np.sqrt(p) - np.sqrt(q))

# Topic modeling implemented with mallet LDA
# Set the number of topics to enumerate as a hyperparameter.
for num_topics in range(50, 175, 25):
    path_to_mallet = "mallet-2.0.8/bin/mallet"
    docs = []
    task_idxs = []
    doc_idxs = []
    for (t, combined_doc) in enumerate(doclist):
        for (i, doc) in enumerate(combined_doc):
            for sentence in doc:
                docs.append(sentence)
                doc_idxs.append(i)
                task_idxs.append(t)

    training_data = [lmw.process_string(t) for t in docs]
    training_data = [d for d in training_data if d.strip()]
    lmw.print_dataset_stats(training_data)
    output_directory_path = "lmw-output"
    topic_keys, topic_distributions = lmw.quick_train_topic_model(path_to_mallet,
                                                              output_directory_path,
                                                              num_topics,
                                                              training_data)
    representation = {} # key: doc_id, value: probs across topics
    for (i, distribution) in enumerate(topic_distributions):
        docid = doc_idxs[i]
        if docid in representation.keys():
            representation[docid] += np.array(distribution)
        else:
            representation[docid] = np.array(distribution)

    # normalize distributions
    normalized = {}
    for key in representation.keys():
        normalized[key] = representation[key] / np.sum(representation[key])
    
    # use hellinger distance because hellinger is symmetric for (i, j) input
    pairwise_heatmap = np.ones((15, 15))
    keys = list(normalized.keys())
    keys = sorted(keys)
    for (i, key_i) in enumerate(keys):
        for (j, key_j) in enumerate(keys):
            # smaller distance means closer
            if i != j:
                dist = hellinger(normalized[key_i], normalized[key_j])
                pairwise_heatmap[i, j] = 1-dist
                pairwise_heatmap[j, i] = 1-dist
    human_eval = pickle.load(open("reasoning_human-eval_scores.pkl","rb"))
    avg_human_eval = human_eval.mean(axis=0)

    i_lower = np.tril_indices(15, k=-1)
    coeff = np.corrcoef(avg_human_eval[i_lower].reshape(-1), pairwise_heatmap[i_lower].reshape(-1))
    print("Avg correlation coeff: ", coeff[0,1],"\n")
    human_scores = avg_human_eval[i_lower].reshape(-1)
    model_scores = pairwise_heatmap[i_lower].reshape(-1)
    # turn into ranks
    human_ranks = rankdata(human_scores, method='average')
    model_ranks = rankdata(model_scores, method='average')
    rho, p_value = spearmanr(human_ranks, model_ranks)
    print("spearman ", rho, p_value)

# Topic modeling implemented with BERTopic using UMPA and HDBSCAN
for min_cluster_size in range(11, 16, 1):
    docs = []
    task_idxs = []
    doc_idxs = []
    for (t, combined_doc) in enumerate(doclist):
        for (i, doc) in enumerate(combined_doc):
            for sentence in doc:
                docs.append(sentence)
                doc_idxs.append(i)
                task_idxs.append(t)

    umap_model = UMAP(n_neighbors=3, n_components=100, min_dist=0.0, metric='cosine', random_state=42)
    # Configure HDBSCAN to form more clusters (lower min_cluster_size) 5, 1
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, metric='euclidean', prediction_data=True)

    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)
    topics, probs = topic_model.fit_transform(docs)
    unique_topics = len(np.unique(topic_model.get_document_info(docs).Topic))
    unique_docs = np.unique(doc_idxs)
    print("\n\nNumber of topics found: ", unique_topics)
    representation = {} # key: doc_id, value: probs across topics
    task_representation = {} # key: doc_id, value: probs across topics
    for _id in unique_docs:
        representation[int(_id)] = np.zeros((unique_topics))
    for _id in unique_docs:
        for _task in range(4):
            task_representation[f"{int(_id)}_{_task}"] = np.zeros((unique_topics))

    topic_modeling_res = topic_model.get_document_info(docs)[["Topic", "Probability"]]
    for i, row in topic_modeling_res.iterrows():
        representation[doc_idxs[i]][int(row.Topic)] += row.Probability
        task_representation[f"{doc_idxs[i]}_{task_idxs[i]}"][int(row.Topic)] += row.Probability

    # normalize distributions
    normalized = {}
    task_normalized = {}
    for key in representation.keys():
        normalized[key] = representation[key] / np.sum(representation[key])
    for key in task_representation.keys():
        task_normalized[key] = task_representation[key] / np.sum(task_representation[key])

    # use hellinger distance because hellinger is symmetric for (i, j) input
    pairwise_heatmap = np.ones((15, 15))
    task_pairwise_heatmap = np.ones((4, 15, 15))
    keys = list(normalized.keys())
    keys = sorted(keys)
    for (i, key_i) in enumerate(keys):
        for (j, key_j) in enumerate(keys):
            # smaller distance means closer
            if i != j:
                dist = hellinger(normalized[key_i], normalized[key_j])
                pairwise_heatmap[i, j] = 1-dist
                pairwise_heatmap[j, i] = 1-dist
    for t in range(4):
        for (i, key_i) in enumerate(keys):
            for (j, key_j) in enumerate(keys):
                if i != j:
                    dist = hellinger(task_normalized[f"{key_i}_{t}"], task_normalized[f"{key_j}_{t}"])
                    task_pairwise_heatmap[t, i, j] = 1-dist
                    task_pairwise_heatmap[t, j, i] = 1-dist

    # check 2 kinds of correlations (per rater, per rater x task) with human-eval scores
    human_eval = pickle.load(open("reasoning_human-eval_scores.pkl","rb"))
    avg_human_eval = human_eval.mean(axis=0)

    i_lower = np.tril_indices(15, k=-1)
    coeff = np.corrcoef(avg_human_eval[i_lower].reshape(-1), pairwise_heatmap[i_lower].reshape(-1))
    print("Avg correlation coeff: ", coeff[0,1],"\n")
    human_scores = avg_human_eval[i_lower].reshape(-1)
    model_scores = pairwise_heatmap[i_lower].reshape(-1)
    # turn into ranks
    human_ranks = rankdata(human_scores, method='average')
    model_ranks = rankdata(model_scores, method='average')
    rho, p_value = spearmanr(human_ranks, model_ranks)
    print("spearman ", rho, p_value)

