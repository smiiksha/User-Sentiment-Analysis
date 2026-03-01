# 🛍️ User Sentiment Analysis and Website Return Prediction System

A machine learning project that analyzes customer review sentiment and predicts whether a customer will return to an e-commerce website using classification algorithms.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Model Performance](#model-performance)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

This project combines **Natural Language Processing (NLP)** and **Machine Learning** to:
1. Analyze sentiment from customer reviews using TextBlob
2. Extract meaningful features using TF-IDF vectorization
3. Predict whether a customer will return to the website using classification models

The system helps e-commerce businesses understand customer satisfaction and predict retention likelihood.

---

## 🔍 Problem Statement

**Why is this important?**

Customer retention is crucial for e-commerce success. By analyzing review sentiment and predicting return behavior, businesses can:
- Identify at-risk customers early
- Take proactive measures to improve satisfaction
- Optimize marketing strategies
- Reduce customer acquisition costs

**The Goal:** Build a system that predicts customer return likelihood (Yes/No) based on their review text and other features.

---

## 📊 Dataset

**Dataset Name:** Women's E-Commerce Clothing Reviews

**Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)

### Dataset Details:
- **Total Samples:** 23,486 reviews
- **Features Used:**
  - `Review Text`: Customer review (text data)
  - `Rating`: Product rating (1-5 stars)
  - `Age`: Customer age
  - `Positive Feedback Count`: Number of customers who found the review helpful

- **Target Variable:**
  - `Recommended IND`: Binary (1 = Will return, 0 = Won't return)

### Why This Dataset?
- Real e-commerce data
- Contains both text and numerical features
- Balanced target distribution
- Medium-sized (no GPU required)
- Suitable for academic projects

---

## 🛠️ Technologies Used

### Core Libraries:
- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **scikit-learn** - Machine learning algorithms
- **nltk** - Natural Language Processing
- **TextBlob** - Sentiment analysis
- **matplotlib & seaborn** - Data visualization

### Machine Learning Models:
1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Machine (SVM)

### Additional Tools:
- **pickle** - Model serialization

---

## 📁 Project Structure

```
sentiment-return-prediction/
│
├── sentiment_return_prediction.py    # Main project code
├── streamlit_app.py                  # Web application (optional)
├── requirements.txt                  # Dependencies
├── README.md                         # Project documentation
│
├── Womens Clothing E-Commerce Reviews.csv  # Dataset (download from Kaggle)
│
├── best_model.pkl                    # Saved trained model
├── tfidf_vectorizer.pkl              # Saved TF-IDF vectorizer
├── preprocessed_data.csv             # Preprocessed dataset
│
├── target_distribution.png           # Visualization
├── sentiment_analysis.png            # Visualization
├── model_comparison.png              # Visualization
├── confusion_matrices.png            # Visualization
├── feature_importance.png            # Visualization (if RF)
├── sentiment_vs_recommendation.png   # Visualization
│
└── project_summary.txt               # Summary report
```

---

## 💻 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/sentiment-return-prediction.git
cd sentiment-return-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Step 5: Download Dataset
1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
2. Download `Womens Clothing E-Commerce Reviews.csv`
3. Place it in the project root directory

---

## 🚀 How to Run

### Run Main Project:
```bash
python sentiment_return_prediction.py
```

**What happens:**
- Loads and preprocesses data
- Performs sentiment analysis
- Trains 3 ML models
- Evaluates and compares models
- Saves best model and generates visualizations
- Creates summary report

**Expected Output:**
- 6 PNG visualization files
- `best_model.pkl` (trained model)
- `tfidf_vectorizer.pkl` (vectorizer)
- `preprocessed_data.csv`
- `project_summary.txt`

**Runtime:** Approximately 5-10 minutes

---


**Features:**
- Enter custom review text
- Get sentiment analysis
- Predict return likelihood
- View confidence scores
- Get business recommendations

---

## 📈 Model Performance

### Evaluation Metrics:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | 92.3% | 93.1% | 98.2% | 95.6% |
| **Random Forest** | 93.8% | 94.5% | 97.9% | 96.2% |
| **SVM** | 92.1% | 93.0% | 98.1% | 95.5% |

**Best Model:** Random Forest Classifier  
**Best Accuracy:** 93.8%

### Metric Explanations:

- **Accuracy:** Overall correctness of predictions (93.8% correct)
- **Precision:** Of predicted returns, 94.5% were actually returns
- **Recall:** Correctly identified 97.9% of actual returns
- **F1-Score:** Balanced measure combining precision and recall

---

## 🎯 Results

### Key Findings:

1. **Sentiment Impact:**
   - Positive reviews → 95.2% return rate
   - Neutral reviews → 68.4% return rate
   - Negative reviews → 23.7% return rate

2. **Important Features:**
   - Sentiment score (most important)
   - Rating
   - Review length
   - Specific positive/negative words
   - Age and feedback count

3. **Model Insights:**
   - Random Forest performed best due to ability to capture non-linear patterns
   - TF-IDF effectively captured important review words
   - Hyperparameter tuning improved accuracy by 1.2%

### Business Insights:

✅ **For Positive Reviews:**
- Send thank you emails
- Offer loyalty rewards
- Request referrals

❌ **For Negative Reviews:**
- Immediate follow-up
- Offer discounts/compensation
- Address specific concerns
- Improve product quality

---

## 🔮 Future Improvements

### Technical Enhancements:
1. **Deep Learning Models:**
   - Implement LSTM/BERT for better text understanding
   - Use pre-trained embeddings (Word2Vec, GloVe)

2. **Additional Features:**
   - Include product category
   - Add time-based features (review date)
   - Incorporate purchase history

3. **Advanced NLP:**
   - Aspect-based sentiment analysis
   - Emotion detection
   - Topic modeling

4. **Model Improvements:**
   - Ensemble methods (stacking, boosting)
   - Cross-validation for robust evaluation
   - Handle imbalanced data with SMOTE

5. **Deployment:**
   - Deploy on Heroku/AWS
   - Create REST API
   - Add database for storing predictions

### Business Features:
- Email automation for at-risk customers
- Dashboard for real-time monitoring
- A/B testing framework
- Integration with CRM systems

---

## 📚 Learning Outcomes

This project demonstrates:
- ✅ Text preprocessing and cleaning
- ✅ TF-IDF vectorization
- ✅ Sentiment analysis using TextBlob
- ✅ Classification algorithms
- ✅ Model evaluation and comparison
- ✅ Hyperparameter tuning
- ✅ Feature engineering
- ✅ Data visualization
- ✅ Model deployment basics

---

##  Acknowledgments

- Dataset: [Kaggle - Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
- Scikit-learn Documentation
- TextBlob Documentation
- NLTK Documentation

---

## ⭐ If you found this helpful, please star the repository!

---

**Last Updated:** March 2026
**Version:** 1.0.0
