# Characterizing and Predicting Social Correction on Twitter
This repository contains data and code for our ACM Web Science 2023 publication regarding social correction on Twitter. 

The PDF can be accessed here: [PDF](https://faculty.cc.gatech.edu/~srijan/pubs/ma-websci23-social-correction.pdf)

If our code or data helps you in your research, please cite:

```
@inproceedings{ma2023characterizing,
  title={Characterizing and Predicting Social Correction on Twitter},
  author={Ma, Yingchen and He, Bing and Subrahmanian, Nathan and Kumar, Srijan},
  booktitle={15th ACM Web Science Conference 2023},
  year={2023}
}
```

## Introduction

Online misinformation has been a serious threat to public health and society. Social media users are known to reply to misinfor- mation posts with counter-misinformation messages, which have been shown to be effective in curbing the spread of misinforma- tion. This is called social correction. However, the characteristics of tweets that attract social correction versus those that do not remain unknown. To close the gap, we focus on answering the following two research questions: (1) “Given a tweet, will it be countered by other users?”, and (2) “If yes, what will be the magni- tude of countering it?”. This exploration will help develop mech- anisms to guide users’ misinformation correction efforts and to measure disparity across users who get corrected. In this work, we first create a novel dataset with 690,047 pairs of misinformation tweets and counter-misinformation replies. Then, stratified anal- ysis of tweet linguistic and engagement features as well as tweet posters’ user attributes are conducted to illustrate the factors that are significant in determining whether a tweet will get countered. Finally, predictive classifiers are created to predict the likelihood of a misinformation tweet to get countered and the degree to which that tweet will be countered.


## Code

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

## Data

### 1. Please refer to README.md in the data folder


We notice the change of Twitter API. If you have problems regarding the access to the whole dataset or the code, please contact Bing He (bhe46@gatech.edu) and Yingchen Ma (yma473@gatech.edu).



