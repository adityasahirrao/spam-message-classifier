
# 📩 SMS Spam Message Filter using NLP & Machine Learning  

A **machine learning-based SMS spam detection system** using **Natural Language Processing (NLP)** techniques and multiple classifiers to classify messages as **Spam** or **Ham (Not Spam)**.  

## 🚀 Features  
- 📝 **Preprocesses SMS messages** with **tokenization, stopword removal, stemming, and TF-IDF vectorization**.  
- 🏆 **Trains multiple classifiers**:  
  - ✅ **Logistic Regression**  
  - ✅ **Naïve Bayes**
  - ✅ **Random Forest**  
  - ✅ **Support Vector Machine (SVM)**  
- 📊 **Evaluates model performance** using accuracy, precision, recall, and F1-score.  
- 🔍 **Detects and classifies messages** with high precision, minimizing false positives.  

## 🛠️ Tech Stack  
- **Python**  
- **NLTK** (Natural Language Toolkit)  
- **Scikit-Learn**  
- **Pandas & NumPy**  

## 📌 Dataset  
- The model is trained on a labeled **SMS spam dataset** containing real-world spam and ham messages.  

## 🚀 How It Works  
1. **Data Preprocessing**  
   - Tokenization  
   - Stopword Removal  
   - Stemming (Porter Stemmer)  
   - TF-IDF Vectorization  

2. **Model Training**  
   - Train multiple ML classifiers  
   - Hyperparameter tuning for optimization  

3. **Evaluation & Selection**  
   - Compare model performance  
   - **Naïve Bayes achieves the highest accuracy (>95%)**  

4. **Spam Prediction**  
   - Classify new SMS messages as **Spam or Ham**  

## 📌 Notes  
- Uses **TF-IDF vectorization** for feature extraction.  
- **Naïve Bayes is the most efficient model** for spam filtering.  
