# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:53:57 2020
Keywords: belajar daring ,  daring rumah , online siswa , belajar online
 corona sekolah ,  corona kuliah , kuliah online ,  sekolah online
@author: ASUS

"""

import tweepy
import csv
import io

api_key = "(insert your api key here)"
api_secret = "(insert your api secret here)"
consumer_token = "(insert your consumer token here)"
consumer_token_secret= "(insert your consumer token secret here)"

auth = tweepy.OAuthHandler(api_key, api_secret)

auth.set_access_token(consumer_token,consumer_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

search_words = "belajar daring -filter:retweets"
date_since = "2020-05-27"

tweets = tweepy.Cursor(api.search,
              q=search_words,
              lang="id",
              since=date_since, tweet_mode='extended').items(500)

with io.open('BelajarDaring.csv', 'w', newline='', encoding="utf-16") as file:
         writer = csv.writer(file, quoting=csv.QUOTE_ALL)
         writer.writerow(["Comment"])
         for tweet in tweets:
             writer.writerow([tweet.full_text])
             
for tweet in tweets:
    print(tweet.full_text)
