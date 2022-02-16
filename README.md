
# Sentiment Analysis of Online Learning Using Naive Bayes Classifier Optimization with Adaboost

This project aims to:
1. Knowing the trend of public sentiment regarding online learning based on tweets taken from July 2020 to December 2020
2. Comparing the performance of Naive Bayes Classifier (NBC) and NBC with Adaboost in classifying sentiment
## ⬇ Here are the steps taken in this project :
## 1. Collecting data from Twitter
The keywords used to collect data are Indonesian words related to the word "belajar daring" or online learning in English. Keywords are determined based on Google Trends. The keywords used are: “belajar daring”, “daring rumah”, “belajar online”, “online siswa”, “corona sekolah”, “corona kuliah”, “kuliah online”, “sekolah online”, and “new normal”.

Collecting tweets requires the api_key, api_secret, consumer_token, and consumer_token_secret from Twitter. After all these things are obtained, enter it to this [code](https://github.com/shabrina19/Analisis-Sentimen-Belajar-Daring-menggunakan-Optimasi-Naive-Bayes-Classifier-dengan-Adaboost/blob/6bb8e8e762b537e6b27bdf9723e2c4afdbc3c0fb/collect-tweets) and run it on the Spyder IDE (Python3.8).
The data will be obtained in csv format and will look like this following image.

![App Screenshot](https://github.com/shabrina19/Gambar/blob/main/pict%20sentiment%20analysis%20of%20online%20learning/resultTweets.png?raw=true)

After that, change the csv data format to xlsx to make it easier for the next step (translate data using Google Translate ).




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
50% of the data in the early months is used to create the model, the file name is “stgh awl hsllabelno_net_all.xlsx” can be accessed [here](https://github.com/shabrina19/Analisis-Sentimen-Belajar-Daring-menggunakan-Optimasi-Naive-Bayes-Classifier-dengan-Adaboost/blob/ff695b115fdc4a6af6331138c8cfd9cd1abc0395/stgh%20awl%20hsllabelno_net_all.xlsx). The initial data was chosen so that the model represents real conditions when the topic of online learning is being discussed by many people. All data is used for sentiment analysis.

## 4. Data Preprocessing
This stage is done to process the raw data so that the data is free from errors. This stage consists of several processes, namely: data cleaning, case folding, tokenizing, replace words, remove punctuation, lemmatization, stopwords, filtering, and remove duplicates. Code for data preprocessing can be accessed [here ](https://github.com/shabrina19/Analisis-Sentimen-Belajar-Daring-menggunakan-Optimasi-Naive-Bayes-Classifier-dengan-Adaboost/blob/6bb8e8e762b537e6b27bdf9723e2c4afdbc3c0fb/Sentiment_Analysis_Belajar_Daring_Online_Learning.ipynb) in the "Preprocessing Tweets" section. The results of the preprocessing are shown in the following image.
[The complete code from start to finish can be accessed here ](https://github.com/shabrina19/Analisis-Sentimen-Belajar-Daring-menggunakan-Optimasi-Naive-Bayes-Classifier-dengan-Adaboost/blob/6bb8e8e762b537e6b27bdf9723e2c4afdbc3c0fb/Sentiment_Analysis_Belajar_Daring_Online_Learning.ipynb)
 
![App Screenshot](https://github.com/shabrina19/Gambar/blob/2ccc78b41bb72865dfae3e425fdb5ad78aab8bd4/pict%20sentiment%20analysis%20of%20online%20learning/RPreprocessing.png?raw=true)
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

The results of labeling using TextBlob with labels from experts obtained an accuracy of 80% as shown in the picture below.

![App Screenshot](https://github.com/shabrina19/Gambar/blob/453845d0a1200f7a620ef07807017d1e65d7b79e/pict%20sentiment%20analysis%20of%20online%20learning/textblobakurasi.png?=true)

## 7. Labeled Data Visualization
The visualization of labeled data is displayed using a bar chart. The bar chart is used to display the number of positive and negative labels. 
The results of labeling with TextBlob show that the number of positive labels is more dominant, which is 15,593 tweets, while negative sentiments are found in 8,864 tweets.

![App Screenshot](https://github.com/shabrina19/Gambar/blob/c1d15058599e73757b95d647a190af9cf8e7efcb/pict%20sentiment%20analysis%20of%20online%20learning/barchartlabel.png?raw=true)

The code can be accessed [here ](https://github.com/shabrina19/Analisis-Sentimen-Belajar-Daring-menggunakan-Optimasi-Naive-Bayes-Classifier-dengan-Adaboost/blob/6bb8e8e762b537e6b27bdf9723e2c4afdbc3c0fb/Sentiment_Analysis_Belajar_Daring_Online_Learning.ipynb) in the "# Visualizations of Labeled Tweets" section.
## 8. Feature Extraction
In the feature extraction stage, TF-IDF is used to obtain combined weights for each word in each document (tweets). TF-IDF has a min_df parameter that functions to filter words, for example if you write min_df=3 then the words that will be processed are words that appear in at least three documents. TF-IDF is applied to the data with various min_df parameters in order to obtain relevant data, can be processed faster, and with optimal accuracy. The difference in accuracy from applying various min_df values ​​will be seen in the evaluation process. The image shows the results of TF-IDF with min_df=3.

![App Screenshot](https://github.com/shabrina19/Gambar/blob/c1d15058599e73757b95d647a190af9cf8e7efcb/pict%20sentiment%20analysis%20of%20online%20learning/tfidf.png?raw=true)

For a clearer explanation, I will explain with an example using 4 tweets which have also been processed in the previous stage. Tweets used are shown in the image below.

![App Screenshot](https://github.com/shabrina19/Gambar/blob/c1d15058599e73757b95d647a190af9cf8e7efcb/pict%20sentiment%20analysis%20of%20online%20learning/contoh4tweet.png?raw=true)
 
The picture below shows an example of TF-IDF results from four documents or four tweets without setting the min_df value, because the example only uses four tweets. The min_df parameter is especially useful when applied to large data. The order of counting starts from 0, so that the resulting sequence of document 0, document 1, document 2, and document 3 is generated. The results of feature extraction are different for each tweet because the number of occurrences of each word is different.

![App Screenshot](https://github.com/shabrina19/Gambar/blob/c1d15058599e73757b95d647a190af9cf8e7efcb/pict%20sentiment%20analysis%20of%20online%20learning/hasiltfidf4tweets.png?raw=true)
## 9. Cross Validation
Cross validation with a value of k=10 is applied in this step. The value of k=10 means dividing the proportion between 90% training data and 10% test data. Cross validation serves to prevent leakage of test data so as to prevent the possibility of overfitting. An example of the results of the cross validation score for each fold with min_df=3 is shown in the image.

![App Screenshot](https://github.com/shabrina19/Gambar/blob/main/pict%20sentiment%20analysis%20of%20online%20learning/crossval.png?raw=true)

Based on the picture, the average cross validation score using NBC is 0.7843969955538611 while NBC with Adaboost gets 0.6375271007128201.
## 10. Modeling
The code can be accessed [here ](https://github.com/shabrina19/Analisis-Sentimen-Belajar-Daring-menggunakan-Optimasi-Naive-Bayes-Classifier-dengan-Adaboost/blob/6bb8e8e762b537e6b27bdf9723e2c4afdbc3c0fb/Sentiment_Analysis_Belajar_Daring_Online_Learning.ipynb) in the "Build Model" to "Model Evaluation" section. At modeling stage, the process of model training and model evaluation is carried out:

a. Training Model

The Scikit-learn library is used in the model training process. First, training was conducted on the NBC model. NBC is trained using data from cross validation. After completion, continued with the training process on Adaboost, NBC became the base_estimator of Adaboost. After all the models have been trained, an evaluation process will be carried out.

b. Model Evaluation

The model will be tested for its performance using two points of view (positive and negative) for recall, precision, f-measure, and to get the overall results used accuracy. The picture shows the accuracy results of NBC and NBC with Adaboost which applies a different min_df value. The number of rows is the number of tweets, while the number of columns is the number of words or terms that were processed.

![App Screenshot](https://github.com/shabrina19/Gambar/blob/main/pict%20sentiment%20analysis%20of%20online%20learning/mindfakurasi.png?raw=true)

The larger the min_df value, the smaller the processed data, so as to increase the accuracy of NBC. Accuracy results from NBC with Adaboost did not change significantly with every change in the amount of data. The model with min_df=3 gets the best result compared to the others, which is 0.78 so the model with min_df=3 will be used for the sentiment classification process. The following is the evaluation result of the model with min_df=3.

b.1. NBC

A total of 24,457 data that have been applied cross validation are used to create the NBC model. Here is a picture of the confusion matrix from NBC.

![App Screenshot](https://github.com/shabrina19/Gambar/blob/main/pict%20sentiment%20analysis%20of%20online%20learning/confmatrixnbc.png?raw=true)

The figure below shows the results of precision, recall, and f1-score on each label from NBC.

![App Screenshot](https://github.com/shabrina19/Gambar/blob/main/pict%20sentiment%20analysis%20of%20online%20learning/evalnbc.png?raw=true)

The results of the evaluation of the NBC model obtained an accuracy of 0.78 or 78% while the cross validation score resulted in an average value of 0.7843969955538611, this indicates that NBC is able to classify well even though various data are given to NBC. In addition, the results of precision, recall, and F1-Score on positive labels get quite good results. Meanwhile, on the negative label, the results were not good with a recall of 0.48 and an F1-Score of 0.62.

b.2. NBC with Adaboost

Adaboost with NBC base estimator. The data of 24,457 that have gone through the cross validation process is used to build the NBC model with Adaboost. The image below is the confusion matrix of NBC with Adaboost.

![App Screenshot](https://github.com/shabrina19/Gambar/blob/main/pict%20sentiment%20analysis%20of%20online%20learning/confmatrixabnb.png?raw=true)

The image below shows the results of precision, recall, and f1-score on each label from NBC with Adaboost.

![App Screenshot](https://github.com/shabrina19/Gambar/blob/main/pict%20sentiment%20analysis%20of%20online%20learning/evalabnb.png?raw=true)

The results of the evaluation of the NBC model with Adaboost obtained an accuracy of 64% while the cross validation score resulted in an average value of 0.6375271007128201. This accuracy can be achieved because the number of positive data is more than negative data, although NBC with Adaboost cannot classify negative labels.

An experiment was carried out by applying Adaboost with the Decision Tree base estimator which is the default base estimator from Adaboost in the Scikit-learn library. This experiment was conducted in order to find out whether there is a mismatch between the data used and Adaboost. The results of the Adaboost evaluation with the Decision Tree are shown in the image below.

![App Screenshot](https://github.com/shabrina19/Gambar/blob/main/pict%20sentiment%20analysis%20of%20online%20learning/evaldecidiontree.png?raw=true)

The results show that the data used can be classified properly when using Decision Tree with Adaboost. The conclusion that can be drawn from this experiment is that the decrease in accuracy in NBC with Adaboost is not due to a mismatch between the data with Adaboost or the data with NBC. However, the cause of the decline in the accuracy of NBC with Adaboost is the combination of data, NBC, and Adaboost. The data used in the modeling is the dominant data on the positive label (unbalanced) so that it affects the performance of NBC with Adaboost in classifying data.  
## 11. Sentiment Classification 
Sentiment classification is done using the model that has been built. From the 77,919 tweets collected, the sentiment classification results from NBC obtained 69,738 tweets labeled positive, and 8,181 tweets labeled negative. Meanwhile, Adaboost failed to classify because it labeled all tweets (77,919 tweets) with positive labels. The figure below shows some of the results of the classification carried out by each model, where 0 indicates a negative label, and 1 indicates a positive label.

![App Screenshot](https://github.com/shabrina19/Gambar/blob/main/pict%20sentiment%20analysis%20of%20online%20learning/klasifikasi.png?raw=true)

## 12. Sentiment Analysis Results
Sentiment classification results obtained from NBC, from 77,919 tweets generated 89.5% data or 69,738 tweets labeled positive, and 10.5% data or 8,181 tweets labeled negative. Meanwhile, NBC with Adaboost labeled 77,919 tweets with positive labels. Based on the two sentiment classification results, it can be concluded that the sentiment classification towards online learning shows a tendency towards positive sentiment.
The image below shows the results of the visualization using a bar chart. The bar chart shows the top 50 words that often appear on positive sentiment. The number of occurrences of words is on the y-axis while the words that occur frequently are on the x-axis. The most words used in tweets with positive sentiments were “offline”, “quota”, “good”, “students”, “pandemic”, “covid”, “face”, “no”, “children”, and “house” .

![App Screenshot](https://github.com/shabrina19/Gambar/blob/main/pict%20sentiment%20analysis%20of%20online%20learning/senpos.png?raw=true)

The image below shows the 50 most frequently used words in negative tweets. The most used words in negative tweets were “offline”, “stupid”, “lazy”, “sorry”, “feel”, “bad”, “students”, “afraid”,”no”, “hard”, “covid ”, “long”, “children”, and “understand”.

![App Screenshot](https://github.com/shabrina19/Gambar/blob/main/pict%20sentiment%20analysis%20of%20online%20learning/senneg.png?raw=true)

Based on the results obtained from the positive sentiment bar chart, the word "quota" is dominantly used. This is because in 2020 the government provides quota assistance for students who get a positive response from the community. The word "good" has a broad meaning, namely "good" because learning is done from home (linked to the word "house") so that the learning atmosphere is more relaxed, it can also be due to the distribution of quota assistance provided by the government. The word "face" can be related to the word "face-to-face learning" which can be something that people tend to like.

The results of the negative sentiment bar chart of the words "stupid", "lazy", "feel" are dominantly used, this is related to the number of students who feel stupid because online learning causes an increase in laziness, the subject matter is difficult to understand (related to the word "hard" , “no”, and, “understand”). The word "sorry" can be related to negative things, such as sadness or regret, this can be associated with many people feeling sad about the application of online learning because online learning also requires a gadget that supports it, even though not everyone has a gadget. The word "bad" can be associated with people who think online learning is something bad, this can be caused by many people who still have difficulties in implementing online learning, because there are not enough internet network facilities. The Covid-19 pandemic has also lasted a long time, and this creates fear so that learning activities become less comfortable. This is related to the words "covid", "long", and "afraid".

Words that often appear in positive and negative sentiments are "offline", "students", "covid", and "children". This shows that there are various views on offline learning. The words "students", "covid", "children" are also words that often appear in both sentiments because they are neutral words that have various contexts.

Bar chart can be created using the code below

```bash
# Install Library
!pip install yellowbrick
!pip install sklearn
from sklearn.feature_extraction.text import CountVectorizer
from yellowbrick.text import FreqDistVisualizer


# Load the text data
text = open('insert file in txt format', mode='r', encoding='utf-8').read()
corpus = [text]

# Count vectorized documents and the features (Words from the corpus)
vectorizer = CountVectorizer(stop_words=["insert stopwords if you need it"])
docs = vectorizer.fit_transform(corpus)
features = vectorizer.get_feature_names()

visualizer = FreqDistVisualizer(features=features, orient='v')
visualizer.fit(docs)
visualizer.show()
```
## Data

- 50% of the data in the early months that used to create the model can be accessed [here](https://github.com/shabrina19/Analisis-Sentimen-Belajar-Daring-menggunakan-Optimasi-Naive-Bayes-Classifier-dengan-Adaboost/blob/ff695b115fdc4a6af6331138c8cfd9cd1abc0395/stgh%20awl%20hsllabelno_net_all.xlsx).
- Some of the results of collecting data tweets in Indonesian can be accessed [here](https://github.com/shabrina19/Analisis-Sentimen-Belajar-Daring-gunakan-Optimasi-Naive-Bayes-Classifier-dengan-Adaboost/blob/ff695b115fdc4a6af6331138c8cfd995cd1abc03995/ind%20cb.txt). 
- Some tweets containing negative sentiments can be accessed [here](https://github.com/shabrina19/Analisis-Sentimen-Learning-Online-using-Optimization-Naive-Bayes-Classifier-with-Adaboost/blob/ff695b115fdc4a6af6331138c8cfd9cd1abc0395/label%20neg%20.xlsx). 

##  💡 References

 - [Document contains complete description of this project](http://digilib.uinsby.ac.id/49606/)
 - [The complete code can be accessed here ](https://github.com/shabrina19/Analisis-Sentimen-Belajar-Daring-menggunakan-Optimasi-Naive-Bayes-Classifier-dengan-Adaboost/blob/6bb8e8e762b537e6b27bdf9723e2c4afdbc3c0fb/Sentiment_Analysis_Belajar_Daring_Online_Learning.ipynb) 
 - [Sentiment analysis using TextBlob](https://towardsdatascience.com/sentiment-analysis-on-the-tweets-about-distance-learning-with-textblob-cc73702b48bc)
 - [Visualization](https://www.scikit-yb.org/en/latest/api/text/freqdist.html)
 - [Recommended channel to learn cross validation etc in Indonesian](https://www.youtube.com/c/JCOpUntukIndonesia/videos)
 

## 👋 Authors 

- [@shabrina](https://github.com/shabrina19)

## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tazkia-shabrina-az-zahra/)
