import praw
import pandas as pd
import string
import numpy as np
from math import sqrt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt



def return_subreddit_data(client_id, client_secret, user_agent, subreddit, time, num_posts):

    reddit = praw.Reddit(client_id=client_id,
                         client_secret=client_secret,
                         user_agent=user_agent)

    dict = { "title":[],
             "subreddit":[],
             "score":[],
             "id":[],
             "url":[],
             "comms_num": []}

    subreddit = reddit.subreddit(subreddit)
    top_reddit = subreddit.top(time_filter=time,limit=num_posts)

    for submission in top_reddit:
        dict["title"].append(submission.title)
        dict['subreddit'].append(submission.subreddit)
        dict["score"].append(submission.score)
        dict["id"].append(submission.id)
        dict["url"].append(submission.url)
        dict["comms_num"].append(submission.num_comments)

    df = pd.DataFrame(dict)

    return df

#remove punctuation
def remove_punctuation(df):

    df['title'] = df['title'].apply(lambda x : ''.join([i for i in x if i not in string.punctuation]))
    df['title'] = df['title'].str.lower()

    return df

def clean_df(df):
    clean_df = df.copy(deep=True)
    stop = stopwords.words('english')
    clean_df['title'] = clean_df['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    clean_df['title'] = clean_df['title'].str.split().map(lambda sl: " ".join(s for s in sl if len(s) > 3))
    return clean_df

def vectorize(clean_df):
    clean_df['tokens'] = clean_df['title'].apply(word_tokenize)
    clean_df['tokens'] = clean_df['tokens'].apply(lambda x: ' '.join([word for word in x]))
    vectorizer = CountVectorizer(lowercase=False)
    x = vectorizer.fit_transform(clean_df['tokens'].values)
    return x

def analyze_sentiment(text):

    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1


def main():

    #insert the following values
    client_id = ''
    client_secret = ''
    user_agent = ''

    df = return_subreddit_data(client_id,client_secret,user_agent,'askreddit', 'year', 1000)
    df = remove_punctuation(df)
    clean = clean_df(df)
    counts = vectorize(clean)

    #prediction of subreddit scores
    X_train, X_test, y_train, y_test = train_test_split(counts, df["score"], test_size=0.2, random_state=1)
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    mse = mean_squared_error(y_test,predictions)
    rmse = sqrt(mse)
    print("Root mean squared error: " + str(rmse))

    #generate word cloud
    text = clean.title.values

    wordcloud = WordCloud(
        width=3000,
        height=2000,
        background_color='black',
        collocations=False).generate(str(text))
    fig = plt.figure(
        figsize=(40, 30),
        facecolor='k',
        edgecolor='k')
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

    #perform sentiment analysis
    clean['SA'] = np.array([analyze_sentiment(text) for text in clean['title']])
    pos_posts = [text for index, text in enumerate(clean['title']) if clean['SA'][index] > 0]
    neu_posts = [text for index, text in enumerate(clean['title']) if clean['SA'][index] == 0]
    neg_posts = [text for index, text in enumerate(clean['title']) if clean['SA'][index] < 0]

    print("Percentage of positive posts: {}%".format(len(pos_posts)*100/len(clean['title'])))
    print("Percentage of neutral posts: {}%".format(len(neu_posts)*100/len(clean['title'])))
    print("Percentage of negative posts: {}%".format(len(neg_posts)*100/len(clean['title'])))

if __name__ == "__main__":
    main()