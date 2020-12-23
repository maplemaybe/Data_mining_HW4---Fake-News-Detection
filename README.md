# Data_mining_HW4-Fake-News-Detection

針對假新聞作分析，預測一則新聞是否reliable 

1: fake
0: true
分別利用GBDT、LightGBM、xgboost對"train.csv"的資料建模，並用"test.csv"進行測試

註："test.csv"的label在"sample_submission.csv"裡面

(資料來源：https://www.kaggle.com/c/fakenewskdd2020/data)

 

流程：

1. 資料前處理

 a. 讀取"train.csv"與"test.csv"並利用分割符號切割、建立train&test之DataFrame

註：分割符號為tab(\t)

 b. 去除停頓詞stop words 

參考：

sklearn.feature_extraction.text.CountVectorizer
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

自訂stop words
https://stackoverflow.com/questions/52712254/how-to-eliminate-stop-words-only-using-scikit-learn


 c. 文字探勘前處理，將文字轉換成向量，像是常見的方法 tf-idf、word2vec...等

參考：

sklearn.feature_extraction.text.TfidfVectorizer
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text

Word2vec
https://radimrehurek.com/gensim/models/word2vec.html


2. 建模：分別使用以下三種模型

xgboost
https://xgboost.readthedocs.io/en/latest/python/python_intro.html#install-xgboost

GBDT

LightGBM
https://machinelearningmastery.com/gradient-boosting-with-scikit-learn-xgboost-lightgbm-and-catboost/


3. 評估模型

利用"test.csv"的資料對2.所建立的模型進行測試，並計算Accuracy、Precision、Recall、F-measure
