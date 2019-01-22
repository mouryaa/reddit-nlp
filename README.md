# reddit-nlp
In this project, I used the Reddit API to extract the top 1000 posts for a whole year from the askreddit subreddit and applied some NLP techniques to analyze the data. Reddit is an online community where users can post links and text posts which are then up and downvoted by fellow users. The askreddit subreddit is a sub-community where users can ask questions to other users who then comment on the post. Taking the top 1000 posts from the last year, I tried to predict the score of posts based on the titles, created a word cloud off the titles and performed sentiment analysis on the titles. 

Prediction of titles

I tried to predict the score a post would get using linear regression. First, in order to apply regression, I needed to convert the textual data into a numerical form. To get the data into numerical format, I need to clean, tokenize and vectorize it. Then I split up the data, into train and test sets and fit the model and predicted it. The RMSE that I got was 21187. Moving forward, I can better engineer the features and apply other machine learning techniques in order to get a smaller RMSE.

Word cloud

As an avid reddit user, the word cloud that was generated from the askreddit titles was not surprising. The words that appeared more frequently such as people, reddit, serious, and what are the words that I expected. The word cloud image is uploaded as a png and you can see it for yourself.

Sentiment analysis

After performing sentiment analysis on the titles, the titles have a 35% positive sentiment, 36% neutal sentiment and a 30% negative sentiment. 
