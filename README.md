# Introduction

This repository contains the datasets and associated code used in the paper *Information Diffusion with Collaborative Signals*

In this work, we introduce a novel paradigm that incorporates collaborative signals into information diffusion tasks. By conceptualizing diffusion cascades as temporal user-information interaction sequences, we jointly model user representations and information representations to capture their latent affinities. To facilitate the exploration of collaborative signals, we construct two large-scale datasets from Twitter and Weibo, addressing critical limitations in existing benchmarks.  Through rigorous experimentation, we validate the effectiveness of collaborative signals using four model categories: basic collaborative models, graph-based models, sequential models, and context-aware sequential architectures.

# Dataset

The Dataset contains the Twitter and the Weibo that we constructed, and the two types of dataset formats that were used in the experiments to test the baseline and rec models, respectively. The basic information of the dataset is as follows.

|             | Twitter | Weibo |
| ----------- | ------- | ----- |
| #user       | 8,292   | 6,512 |
| #cascades   | 2,798   | 2,267 |
| #Links      | 86,768  | -     |
| Avg. Length | 3.39    | 43.73 |

## Twitter

The Twitter dataset raw information contains two files: interaction.csv and post.csv. Among them, interaction.csv contains the record of each post being retweeted by users, and post.csv records the information related to each post when it is published. Due to the requirements of the Twitter platform, the post and the user's repost text information are converted to embedding format using the sentiment-transformers/all-MiniLM-L6-v2 model.

Due to Github repository capacity limitations, the full dataset can be downloaded from the link: https://huggingface.co/datasets/collaborativeS/Information-Diffusion-with-Collaborative-Signals

The meaning of each field in the interaction.csv file is as follows:

* post_id: Unique identifier for the retweeted post. The post_id remains consistent across other files in the dataset.
* user_id: Unique identifier for the user who retweet the post. The user_id remains consistent across other files in the dataset.
* timestamp: Relative time for users to retweet the post
* repost_content: Text embedding when retweeted by the user

The meaning of each field in the post.csv file is as follows:

* post_id: Unique identifier for the post.
* user_id: Unique identifier for the user who publish the post.
* timestamp: Relative time for users to publish the post
* post_content: Text embedding when published by the user
* length: The length of the post

## Weibo

The Twitter dataset raw information contains three files: interaction.csv, post.csv and user.csv. Among them, interaction.csv contains the record of each post being reposted by users, post.csv records the information related to each post when it is published, and the user.csv is the profile information for the users in interaction.csv.

The meaning of each field in the interaction.csv file is as follows:

* post_id: Unique identifier for the commented post. The post_id remains consistent across other files in the dataset.
* user_id: Unique identifier for the user who comment the post. The user_id remains consistent across other files in the dataset.
* timestamp: Relative time for users to comment the post
* comment_content: Text messages when commented by the user
* like_count: The number of likes received for this comment
* response_count: The number of comments received for this comment

The meaning of each field in the post.csv file is as follows:

* post_id: Unique identifier for the post.
* user_id: Unique identifier for the user who publish the post.
* timestamp: Relative time for users to publish the post.
* post_content: Text messages when published by the user.
* repost_count: The number of reposts received for this post
* like_count: The number of likes received for this post
* comment_count: The number of comments received for this post

The meaning of each field in the user.csv is as follows:

* user_id: Unique identifier for the user.
* gender: The gender category of the user.
* follower_count: The number of followers (accounts following the user) of this user.
* followee_count : The number of accounts the user is following.
* post_count: The total number of posts published by the user.
* comment_count: The total number of comments published by the user
* like_count: The total number of likes given by the user.

# Getting started

We use the Rechorus framework to complete model testing with co-signals. Link to information on deployment and use of the Rechorus framework: https://github.com/THUwangcy/ReChorus

data_process.py is the code that processes the Twitter, Weibo, and Urban datasets into the datasets needed for the baseline and Rechorus framework tests. For each dataset, the raw file information should be placed in the origin folder, and the processing results will be saved in the IF folder and Rec folder respectively.
