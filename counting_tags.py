import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import os
import csv
import pandas as pd
from wordcloud import WordCloud

data = pd.read_csv('./csv/no_duplicates.csv')

# Initializing CountVectorizer object which is scikit-learn's bag of words
vectorizer = CountVectorizer(tokenizer=lambda x: x.split())

# fit_Transform() will learn the vocabulary dictionary and return term-document matrix
tag_vocabulary = vectorizer.fit_transform(data['Tags'].values.astype(str))

print("Number of total data: ", tag_vocabulary.shape[0])
print("Number of unique tags: ", tag_vocabulary.shape[1])

tags = vectorizer.get_feature_names()
print("Tags -> ", tags[:20])

tag_frequency = tag_vocabulary.sum(axis=0).A1
result = dict(zip(tags, tag_frequency))

if not os.path.isfile('./csv/tags_count.csv'):
    with open('./csv/tags_count.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in result.items():
            writer.writerow([key, value])

tag_dataframe = pd.read_csv('./csv/tags_count.csv', names=['Tags', 'Counts'])
print(tag_dataframe.head())
sorted_tag_dataframe = tag_dataframe.sort_values(['Counts'], ascending=False)
tags_count = sorted_tag_dataframe['Counts'].values

# Plotting the tag appearance in the question
plt.plot(tags_count)
plt.title("Number of times tag appeared in question distribution")
plt.xlabel("Tag Number")
plt.ylabel("Number of times tag appeared ")
plt.show()

# Plotting the first 1k tag appearance in the question
plt.plot(tags_count[:1000])
plt.title("First 1k tags: Number of times tag appeared in question distribution")
plt.grid()
plt.xlabel("Tag Number")
plt.ylabel("Number of times tag appeared ")
plt.show()
print(len(tags_count[:1000:5]))
print(tags_count[:1000:5])

# Plotting first 100 tags distribution
plt.plot(tags_count[:100], c='b')
plt.scatter(x=list(range(0, 100, 10)), y=tags_count[0:100:10], c='red', label="Quantiles with 0.10 intervals")
plt.scatter(x=list(range(0, 100, 25)), y=tags_count[0:100:25], c='green', label="Quantiles with 0.25 intervals")
plt.title("First 100 tags: Number of times tag appeared in question distribution")
plt.xlabel("Tag Number")
plt.ylabel("Number of times tag appeared ")
plt.legend()
plt.show()
print(len(tags_count[:100:5]))
print(tags_count[:100:5])

# Extracting tags that are used more than 10k and 100k
tags_10k = tag_dataframe[tag_dataframe.Counts > 10000].Tags
tags_100k = tag_dataframe[tag_dataframe.Counts > 100000].Tags
print(len(tags_10k), ' tags are used more than 10k times')
print(len(tags_100k), ' tags are used more than 100k times')

# Extracting number of tags per question
question_tag_count = tag_vocabulary.sum(axis=1).tolist()
question_tag_count = [int(j) for i in question_tag_count for j in i]
print("Total questions: ", len(question_tag_count))
print(question_tag_count[:10])
print("Maximum number of tags per question: ", max(question_tag_count))
print("Minimum number of tags per question: ", min(question_tag_count))
print("Average number of tags per question: ", ((sum(question_tag_count)*1.0)/len(question_tag_count)))

# Plotting the question_tag_count
sns.countplot(question_tag_count, palette='rocket')
plt.title("Number of tags in the questions")
plt.xlabel("Number of tags")
plt.ylabel("Number of questions")
plt.show()

# Plotting WordCloud for the tags
tag_tuple = dict(result.items())
wordcloud = WordCloud(background_color='white', width=1600, height=900).generate_from_frequencies(tag_tuple)
wordcloud_fig = plt.figure(figsize=(50, 30))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
wordcloud_fig.savefig('./output/tag_count_wordcloud.png')
plt.show()

i = np.arange(20)
sorted_tag_dataframe.head(20).plot(kind='bar')
plt.title("Frequency of top 20 tags")
plt.xticks(i, sorted_tag_dataframe['Tags'])
plt.xlabel("Tags")
plt.ylabel("Counts")
plt.show()
