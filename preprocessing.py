import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import csv


def html_stripper(body_data):
    html_strip_re = re.compile('<.*?>')
    clean_text = re.sub(html_strip_re, ' ', str(body_data))
    return clean_text


stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

questions_with_code = question_processed = pre_question_length = post_question_length = 0
data = pd.read_csv("./csv/no_duplicates.csv", names=['Title', 'Body', 'Tags'], header=1)
data = data.sample(n=2000000)   # Random 2 millions rows
print(data.head())
print(data.shape)

for dataframe in data.values:
    title, body, tags = dataframe[0], dataframe[1], dataframe[2]
    is_code = 0
    if '<code>' in body:
        questions_with_code += 1
        is_code = 1

    x = len(title) + len(body)
    pre_question_length += x

    code = str(re.findall(r'<code>(.*?)</code>', body, flags=re.DOTALL))

    body = re.sub(r'<code>(.*?)</code>', '', body, flags=re.DOTALL | re.MULTILINE)

    body = html_stripper(body.encode('utf-8'))
    title = title.encode('utf-8')

    question = str(title) + " " + str(body)
    question = re.sub(r'[^A-Za-z]+', ' ', question)
    words = word_tokenize(str(question.lower()))

    question = ' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j) != 1 or j == 'c'))
    post_question_length += len(question)

    question_preprocessed = (question, tags)
    question_processed += 1

    if question_processed % 10000 == 0:
        print("Number of questions processed = ", question_processed)

    with open('./csv/preprocessed_questions_random_two_million_rows.csv', mode='a') as file:
        writer = csv.writer(file)
        writer.writerow(question_preprocessed)
    file.close()

average_length_preprocess = (pre_question_length * 1.0) / question_processed
average_length_postprocess = (post_question_length * 1.0) / question_processed

print("Average length of question before pre_processing: ", average_length_preprocess)
print("Average length of question before post_processing: ", average_length_postprocess)
print("Question containing code percent: ", (questions_with_code * 100.0) / question_processed)
