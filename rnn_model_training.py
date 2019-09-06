from keras.layers import SimpleRNN, Embedding, Dense, LSTM
from keras.models import Sequential

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

data = pd.read_csv("./csv/preprocessed_questions_random_two_million_rows.csv", header=None, names=["question", "tags"])

question = []
tags = []
for i in enumerate(data['tags']):
    question.append(data['question'])
    tags.append(data['tags'])

questions = np.asarray(question)
tags = np.asarray(tags)

print("number of texts :" , len(questions))
print("number of labels: ", len(tags))
print(question[:10], "\n", tags[:10])
