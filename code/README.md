# the Counter-misinformation Reply Classifier

* We use the annotated tweet-reply pairs to build the classifier
    1. Input: a (tweet, reply) pair
    2. Output: the probability that the reply explicitly or implicitly counters the misinformation tweet, and the predicted label (0: non-countering, 1: countering)
* The environment file is requirements.txt

### 1. Setup and Installation

Our framework can be compiled on Python 3 environments. The modules used in our code can be installed using:
```
pip install -r requirements.txt
```
The requirements.txt is located in code folder

### 2. Train the Counter-misinformation Classifier

```
Python tweet_counterreply_clf.py
```

### 3. If you have questions regarding the code and dataset, please contact Bing He (bhe46@gatech.edu) and Yingchen Ma (yma473@gatech.edu).