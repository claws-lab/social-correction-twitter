# annotated tweet-reply pairs in Section x

* for a reply, per Twitter's rule, we provide the id information for retrieval via Twitter API.
* to determine if a reply is countering reply or non-countering reply, please refer to the column if_counterreply where 1 means countering and 0 indicates non-countering
* to determine if a reply is explicitly countering or implicitly countering, please refer to the column if_explicitly_counter where 1 means explicitly countering, 0 means implicitly countering and -1 means non-countering
* when creating the classifier in Section X, we use the column if_counterreply to build a binary classifier
