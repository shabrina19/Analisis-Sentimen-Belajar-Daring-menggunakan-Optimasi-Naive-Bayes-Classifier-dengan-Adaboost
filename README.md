
# Sentiment Analysis of Online Learning Using Naive Bayes Classifier Optimization with Adaboost

This project aims to:
1. Knowing the trend of public sentiment regarding online learning based on tweets taken from July 2020 to December 2020
2. Comparing the performance of Naive Bayes Classifier (NBC) and NBC with Adaboost in classifying sentiment
## Here are the steps taken in this project :
## 1. Collecting data from Twitter
The keywords used to collect data are Indonesian words related to the word "belajar daring" or online learning in English. Keywords are determined based on Google Trends. The keywords used are: “belajar daring”, “daring rumah”, “belajar online”, “online siswa”, “corona sekolah”, “corona kuliah”, “kuliah online”, “sekolah online”, and “new normal”.

Collecting tweets requires the api_key, api_secret, consumer_token, and consumer_token_secret from Twitter. After all these things are obtained, enter it to this [code](https://github.com/shabrina19/Analisis-Sentimen-Belajar-Daring-menggunakan-Optimasi-Naive-Bayes-Classifier-dengan-Adaboost/blob/6bb8e8e762b537e6b27bdf9723e2c4afdbc3c0fb/collect-tweets) and run it on the Spyder IDE (Python3.8).
The data will be obtained in csv format and will look like this following image.
![App Screenshot](https://github.com/shabrina19/Gambar/blob/main/pict%20sentiment%20analysis%20of%20online%20learning/resultTweets.png?raw=true)
After that the csv data format is changed to xlsx to make it easier to translate data using Google Translate.




## 2. Remove Duplicate Data and Translate Data
There is a lot of spam that has a similar tweet content. Duplicate data is removed in order to speed up the next process and reduce repetitive data, so that the data processed is unique data, to remove duplicate data you can use this following command.
```bash
# Load tweets
tweets_raw = pd.read_excel("insert your path file here") 

# Drop duplicated rows
tweets_raw.drop_duplicates(keep='first', inplace=True)

# Replace NaN Values
tweets_raw['Processed'].replace('', np.nan, inplace=True)

# Drop Columns with Null values
tweets_raw.dropna(subset=['Processed'], inplace=True)

# Save in xlsx format
# tweets_raw[["Processed"]].to_excel("enter the path to save the file /DocumentName.xlsx")
```
Data is translated to English because labeling using subjectivity and polarity from the TextBlob library is only available in English. NBC is supervised machine learning that requires labeled data. 
![App Screenshot](https://github.com/shabrina19/Gambar/blob/2ccc78b41bb72865dfae3e425fdb5ad78aab8bd4/pict%20sentiment%20analysis%20of%20online%20learning/GoogleTranslate.png?raw=true)
The translation results are shown in the following image.
![App Screenshot](https://github.com/shabrina19/Gambar/blob/df46a237057a556ef412bb9151b1d41a6d2331b4/pict%20sentiment%20analysis%20of%20online%20learning/ResultTranslate.png?raw=true)


## 3. Data Sampling
50% of the data in the early months is used to create the model. The initial data was chosen so that the model represents real conditions when the topic of online learning is being discussed by many people. All data is used for sentiment analysis.
## 4. Data Preprocessing
This stage is done to process the raw data so that the data is free from errors. This stage consists of several processes, namely: data cleaning, case folding, tokenizing, replace words, remove punctuation, lemmatization, stopwords, filtering, and remove duplicates. Code for data preprocessing can be accessed [here ](https://github.com/shabrina19/Analisis-Sentimen-Belajar-Daring-menggunakan-Optimasi-Naive-Bayes-Classifier-dengan-Adaboost/blob/6bb8e8e762b537e6b27bdf9723e2c4afdbc3c0fb/Sentiment_Analysis_Belajar_Daring_Online_Learning.ipynb) in the "Preprocessing Tweets" section. The results of the preprocessing are shown in the following image. 
![App Screenshot](https://github.com/shabrina19/Gambar/blob/2ccc78b41bb72865dfae3e425fdb5ad78aab8bd4/pict%20sentiment%20analysis%20of%20online%20learning/RPreprocessing.png?raw=true)
[The complete code from start to finish can be accessed here ](https://github.com/shabrina19/Analisis-Sentimen-Belajar-Daring-menggunakan-Optimasi-Naive-Bayes-Classifier-dengan-Adaboost/blob/6bb8e8e762b537e6b27bdf9723e2c4afdbc3c0fb/Sentiment_Analysis_Belajar_Daring_Online_Learning.ipynb)
## 5. Data Labeling
TextBlob calculates an average to give values for polarity and subjectivity. Polarity values with certain numbers will be labeled as positive (Positif) , neutral (Netral), and negative (Negatif) as written in Table. TextBlob ignores unknown words. 
| Polarity             | Label                                                                |
| ----------------- | ------------------------------------------------------------------ |
| polarity > 0 | Positif |
| polarity = 0 | Netral |
| polarity < 0 | Negatif |

Code for data labeling can be accessed [here ](https://github.com/shabrina19/Analisis-Sentimen-Belajar-Daring-menggunakan-Optimasi-Naive-Bayes-Classifier-dengan-Adaboost/blob/6bb8e8e762b537e6b27bdf9723e2c4afdbc3c0fb/Sentiment_Analysis_Belajar_Daring_Online_Learning.ipynb) in the "Labeling Tweets" section. These are some results of labeling data using TextBlob.
![App Screenshot](https://github.com/shabrina19/Gambar/blob/2ccc78b41bb72865dfae3e425fdb5ad78aab8bd4/pict%20sentiment%20analysis%20of%20online%20learning/LabelP.png?raw=true)
In this project only handles positive and negative labels, so data with neutral labels is removed.
## 6. Label Validation
Label validation is done manually by an expert. Validated data is a sample of all data that has been labeled by TextBlob. Sampling was carried out using a non-probability technique with judgment sampling. The data that has been labeled by TextBlob is 24,457 with 15,593 positive labels and 8,864 negative labels. The sample used is 200 data with 91 positive labels, and 109 negative labels. The picture shows some results from label validations, with "Sentimen Pakar" means sentiment from the expert, 0 being a negative label, and 1 being a positive label. B is “True”, where the label result from TextBlob is the same as expert, and S is “False” which means the label result from TextBlob is different from expert.
![App Screenshot](https://github.com/shabrina19/Gambar/blob/2ccc78b41bb72865dfae3e425fdb5ad78aab8bd4/pict%20sentiment%20analysis%20of%20online%20learning/label2.png?raw=true)