# Predicting Comment Popularity on Imgur

This machine learning project aims to predict the popularity of comments on the image-sharing community [Imgur](http://imgur.com/).
It contains code to train and predict comment populary, code to collect a data set of comments from Imgur, and a small data set of comments.

## Results

For information about the results we achieved with this project, take a look at our [report](https://drive.google.com/file/d/0B9wEkmxfc7Y2VkZCUkJIQVpiWmc/view?usp=sharing) and [presentation](https://drive.google.com/file/d/0B9wEkmxfc7Y2enNBZ1F3ZlBSVzA/view?usp=sharing).

## Data sets

We collected our own data sets by extracting comments from Imgur’s API with the imgur-python client.

In addition to the small data set included in this repository, we collected a much larger data set available for download [here](http://vinstuff.com/files/commentFeaturesList_1449189704.csv) (about 700MB).
Specifically, we extracted the 2.7 million comments, labeled with net upvote score, that comprise all comments on the “front page” of Imgur in the thirty days of November 2015.
Each example contains the comment’s text, post time, reply status, upvotes, and downvotes; whether the comment’s author is the original poster of the image; and the original image post’s time and net upvote score.
