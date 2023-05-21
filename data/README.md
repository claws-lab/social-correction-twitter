
# Dataset

1. To download the dataset, please use this [link](https://www.dropbox.com/sh/97zqw84ae5tej7t/AAB49_iNqEJGC0sjXEb8WiqLa?dl=0).
2. We notice the change of Twitter API. If you do not have access to the Twitter data, we can share the whole data with you properly. Please contact Bing He (bhe46@gatech.edu)


# Annotated Tweet-reply Pairs

* for a reply, per Twitter's rule, we provide the tweet ID information when retrieving using Twitter API. Here, you can (1) get the reply text by the provided id; (2) get the corresponding tweet id and text using the counversation_id.
* to determine if a reply is countering reply or non-countering reply, please refer to the column if_counterreply where 1 means countering and 0 indicates non-countering
* to determine if a reply is explicitly countering or implicitly countering, please refer to the column if_explicitly_counter where 1 means explicitly countering, 0 means implicitly countering and -1 means non-countering
* when creating the counter-reply classifier in Section 3.3.2, we use the column if_counterreply to build a binary classifier.
* if you only want to collect the pairs of (misinformation tweets, explicit counter-misinformation replies), please also check our another [paper](https://github.com/claws-lab/MisinfoCorrect).

# The Whole Twitter Dataset

Note that: Due to the change of Twitter API. If you do not have access to the Twitter data, we can share the following CSV with you properly. Please contact Bing He (bhe46@gatech.edu). 

## Misinformation tweets

`misinfo_tweets.csv`: contains misinformation tweet information (tweet ID, text, author ID, engagement statistics, etc.).
* `misinfo_tweets_ids_only.csv`: contains tweet ID only.

## Replies to misinformation tweets

`misinfo_replies.csv`: contains all replies to misinformation tweets (tweet ID, text, author ID, engagement statistics, conversation ID, etc.)
* `misinfo_replies_ids_only.csv`: contains (reply ID, misinfo tweet ID) pairs only.
* NOTE: `conversation_id` attribute refers to the ID of the tweet the reply is replying to.

`misinfo_counterreplies.csv`: contains all counter-replies to misinformation tweets (tweet ID, text, author ID, engagement statistics, conversation ID, etc.)
* `misinfo_counterreplies_ids_only.csv`: contains (reply ID, misinfo tweet ID) pairs only.
* NOTE: `conversation_id` attribute refers to the ID of the tweet the reply is replying to.

## Misinformation posters

`misinfo_users.csv`: contains misinformation tweet poster information (user ID, username, engagement statistics, etc.).
* `misinfo_users_ids_only.csv`: contains user ID only.

## Misinformation poster historical tweets

`premisinfo_tweets.csv`: contains misinformation tweet posters' original tweets (non-RT, non-quote, non-reply) in the seven days leading up to the misinformation tweet
* `premisinfo_tweets_ids_only.csv`: contains (tweet ID, author ID, ref misinfo tweet ID) only.
* NOTE: `ref_misinfo_tweet_id` attribute refers to the ID of the misinformation tweet the user with `author_id` posted.

`relevant_premisinfo_tweets.csv`: contains tweets in `premisinfo_tweets.csv` that contain a COVID-19 related keyword
* `relevant_premisinfo_tweets_ids_only.csv`: contains (tweet ID, author ID, ref misinfo tweet ID) only.
* NOTE: `ref_misinfo_tweet_id` attribute refers to the ID of the misinformation tweet the user with `author_id` posted.