"""
Prompt LLM for pick-the-odd-one-out task with student reasoning data.
"""

import os
import anthropic
from openai import OpenAI
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict 
import pickle 
import numpy as np
from itertools import combinations
import re

items = list(range(15))
triplets = list(combinations(items, 3))

_use_openai = False
_use_claude = True
""" Set up the client model """
if _use_openai:
    client = OpenAI()
elif _use_claude:
    client = anthropic.Anthropic()

triplet_scores = np.zeros((4, 15, 15))
triplet_counts = np.zeros((4, 15, 15))

# Load the meta-data
tasks = [task1, task2, task3, task4]
query_n = 0
N_REPEAT = 10
# Load the annotation doc from the doclist
for (task, docs) in enumerate(doclist):
    input_docs = []
    for doc in docs:
        # large context length allows document-embeddings
        trimmed_doc = [x.strip() for x in doc]
        input_doc = " ".join(trimmed_doc)
        input_docs.append(input_doc)
    for triplet in tqdm(triplets):
        # retrieve the 3 docs
        x1, x2, x3 = triplet
        # randomize the ordering
        for _ in tqdm(range(N_REPEAT)):
            ordering = np.random.permutation([x1, x2, x3])
            option_a = input_docs[ordering[0]]
            option_b = input_docs[ordering[1]]
            option_c = input_docs[ordering[2]]
            task_prompt = tasks[task]
            system_prompt = f"You are a helpful math assistant. Teachers were given a transcript of students working on a math lesson and this information about that lesson's purpose: {task_prompt}\n\nTeachers were then asked to make notes about what students said in the transcript that would help them assess and/or advance student's understanding toward the lesson purpose. Choose the set of notes that is least like the other two sets of notes. Decide based on the content rather than the style, tone, or length of the notes."
            user_prompt = f"##A##\n{option_a}\n\n##B##{option_b}\n\n##C##{option_c}\n\nChoose the set of notes that is least like the other two sets of notes. Respond with ##A## if A is most different, ##B## if B is most different, and ##C## if C is most different. Decide based on the content rather than the style, tone, or length of the notes."
            if _use_openai:
                completion = client.chat.completions.create(
                                    model="gpt-4.1",
                                    messages=[
                                            {"role": "system", "content": system_prompt},
                                            {"role": "user", "content": user_prompt}
                                            ]
                                    )
                text = completion.choices[0].message.content
            elif _use_claude:
                message = client.messages.create(
                    model=model,
                    max_tokens=1000,
                    temperature=0.7,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return message.content[0].text
            try:
                odd = re.findall(r"##([a-zA-Z])##", text)[0]
            except Exception:
                print(text)
                odd = "NA"
            sim_i, sim_j, sim_k = ordering[0], ordering[1], ordering[2]
            if odd == "A":
                # add to anything but i
                triplet_scores[task, sim_k, sim_j] += 1
                triplet_scores[task, sim_j, sim_k] += 1
            elif odd == "B":
                # add to anything but j
                triplet_scores[task, sim_k, sim_i] += 1
                triplet_scores[task, sim_i, sim_k] += 1
            elif odd == "C":
                triplet_scores[task, sim_j, sim_i] += 1
                triplet_scores[task, sim_i, sim_j] += 1
            elif odd == "NA":
                triplet_scores[task, sim_k, sim_j] += 1
                triplet_scores[task, sim_j, sim_k] += 1
                triplet_scores[task, sim_k, sim_i] += 1
                triplet_scores[task, sim_i, sim_k] += 1
                triplet_scores[task, sim_j, sim_i] += 1
                triplet_scores[task, sim_i, sim_j] += 1

            triplet_counts[task, sim_i, sim_j] += 1
            triplet_counts[task, sim_j, sim_i] += 1
            triplet_counts[task, sim_i, sim_k] += 1
            triplet_counts[task, sim_j, sim_k] += 1
            triplet_counts[task, sim_k, sim_j] += 1
            triplet_counts[task, sim_k, sim_i] += 1
            
            query_n += 1

