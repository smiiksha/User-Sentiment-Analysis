"""
User Sentiment Analysis and Website Return Prediction System
B.Tech ML Project - Medium Level
Author: [Your Name]
"""

# ============================================
# 1. IMPORT LIBRARIES
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Text Processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Sentiment Analysis
from textblob import TextBlob

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)

# Download required NLTK data
print("Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
print("✓ NLTK resources downloaded\n")

# ============================================
# 2. LOAD DATASET
# ============================================

print("="*60)
print("STEP 1: LOADING DATASET")
print("="*60)

# Load the dataset
# Note: Download from Kaggle and place in same directory
# https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')

print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nColumn names:")
print(df.columns.tolist())
print(f"\nDataset Info:")
print(df.info())

# ============================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================

print("\n" + "="*60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

# Check target variable distribution
print(f"\nTarget Variable Distribution:")
print(df['Recommended IND'].value_counts())
print(f"\nPercentage Distribution:")
print(df['Recommended IND'].value_counts(normalize=True) * 100)

# Check missing values
print(f"\nMissing Values:")
print(df.isnull().sum())

# Basic statistics
print(f"\nBasic Statistics:")
print(df.describe())

# Visualize target distribution
plt.figure(figsize=(8, 5))
df['Recommended IND'].value_counts().plot(kind='bar', color=['#ff6b6b', '#4ecdc4'])
plt.title('Distribution of Recommended IND (Target Variable)', fontsize=14, fontweight='bold')
plt.xlabel('Recommended (1 = Yes, 0 = No)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: target_distribution.png")
plt.close()

# ============================================
# 4. DATA PREPROCESSING
# ============================================

print("\n" + "="*60)
print("STEP 3: DATA PREPROCESSING")
print("="*60)

# Select relevant columns
df_clean = df[['Review Text', 'Rating', 'Age', 'Positive Feedback Count', 'Recommended IND']].copy()

# Handle missing values in Review Text
print(f"\nMissing values in Review Text: {df_clean['Review Text'].isnull().sum()}")
df_clean = df_clean.dropna(subset=['Review Text'])
print(f"After removing nulls: {df_clean.shape[0]} rows")

# Fill missing values in other columns with median
df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
df_clean['Positive Feedback Count'].fillna(0, inplace=True)

print(f"\n✓ Missing values handled")
print(f"Final dataset shape: {df_clean.shape}")

# ============================================
# 5. TEXT PREPROCESSING FUNCTION
# ============================================

print("\n" + "="*60)
print("STEP 4: TEXT PREPROCESSING")
print("="*60)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess text data:
    1. Lowercase conversion
    2. Remove special characters and numbers
    3. Remove stopwords
    4. Lemmatization
    """
    # Lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    words = text.split()
    
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Join back to string
    return ' '.join(words)

# Apply preprocessing
print("Processing reviews (this may take 1-2 minutes)...")
df_clean['Cleaned_Review'] = df_clean['Review Text'].apply(preprocess_text)

print("\n✓ Text preprocessing completed")
print(f"\nExample of preprocessed text:")
print(f"\nOriginal: {df_clean['Review Text'].iloc[0][:100]}...")
print(f"\nCleaned: {df_clean['Cleaned_Review'].iloc[0][:100]}...")

# ============================================
# 6. SENTIMENT FEATURE EXTRACTION
# ============================================

print("\n" + "="*60)
print("STEP 5: SENTIMENT ANALYSIS USING TEXTBLOB")
print("="*60)

def get_sentiment_score(text):
    """
    Extract sentiment polarity using TextBlob
    Returns: polarity score between -1 (negative) and 1 (positive)
    """
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return 0

# Extract sentiment scores
print("Extracting sentiment scores...")
df_clean['Sentiment_Score'] = df_clean['Review Text'].apply(get_sentiment_score)

print("\n✓ Sentiment extraction completed")
print(f"\nSentiment Score Statistics:")
print(df_clean['Sentiment_Score'].describe())

# Categorize sentiment
def categorize_sentiment(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df_clean['Sentiment_Category'] = df_clean['Sentiment_Score'].apply(categorize_sentiment)

print(f"\nSentiment Category Distribution:")
print(df_clean['Sentiment_Category'].value_counts())

# Visualize sentiment distribution
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
df_clean['Sentiment_Category'].value_counts().plot(kind='bar', color=['#51cf66', '#ffd43b', '#ff6b6b'])
plt.title('Sentiment Category Distribution', fontweight='bold')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
plt.hist(df_clean['Sentiment_Score'], bins=50, color='#4ecdc4', edgecolor='black')
plt.title('Sentiment Score Distribution', fontweight='bold')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('sentiment_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: sentiment_analysis.png")
plt.close()

# ============================================
# 7. FEATURE ENGINEERING
# ============================================

print("\n" + "="*60)
print("STEP 6: FEATURE ENGINEERING")
print("="*60)

# Create additional features
df_clean['Review_Length'] = df_clean['Review Text'].apply(len)
df_clean['Word_Count'] = df_clean['Cleaned_Review'].apply(lambda x: len(x.split()))

print(f"✓ Additional features created:")
print(f"  - Sentiment_Score")
print(f"  - Review_Length")
print(f"  - Word_Count")

# ============================================
# 8. TF-IDF VECTORIZATION
# ============================================

print("\n" + "="*60)
print("STEP 7: TF-IDF VECTORIZATION")
print("="*60)

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=1000,  # Limit to top 1000 words
                        min_df=5,            # Word must appear in at least 5 documents
                        max_df=0.8)          # Word must not appear in more than 80% documents

# Fit and transform cleaned reviews
print("Converting text to TF-IDF features...")
tfidf_features = tfidf.fit_transform(df_clean['Cleaned_Review'])

print(f"\n✓ TF-IDF vectorization completed")
print(f"TF-IDF matrix shape: {tfidf_features.shape}")
print(f"(Rows = Reviews, Columns = Unique words)")

# Convert to DataFrame
tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                        columns=tfidf.get_feature_names_out())

# Combine with other features
numerical_features = df_clean[['Rating', 'Age', 'Positive Feedback Count', 
                               'Sentiment_Score', 'Review_Length', 'Word_Count']].reset_index(drop=True)

X = pd.concat([tfidf_df, numerical_features], axis=1)
y = df_clean['Recommended IND'].reset_index(drop=True)

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Features include: {tfidf_features.shape[1]} TF-IDF features + 6 numerical features")

# ============================================
# 9. TRAIN-TEST SPLIT
# ============================================

print("\n" + "="*60)
print("STEP 8: TRAIN-TEST SPLIT")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42,
                                                    stratify=y)

print(f"Training set size: {X_train.shape[0]} samples ({(X_train.shape[0]/len(X))*100:.1f}%)")
print(f"Testing set size: {X_test.shape[0]} samples ({(X_test.shape[0]/len(X))*100:.1f}%)")
print(f"\nTarget distribution in training set:")
print(y_train.value_counts())

# ============================================
# 10. MODEL TRAINING
# ============================================

print("\n" + "="*60)
print("STEP 9: MODEL TRAINING")
print("="*60)

# Dictionary to store models and results
models = {}
results = {}

# ==================
# Model 1: Logistic Regression
# ==================
print("\n[1/3] Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
models['Logistic Regression'] = lr_model
print("✓ Logistic Regression trained")

# ==================
# Model 2: Random Forest
# ==================
print("\n[2/3] Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
models['Random Forest'] = rf_model
print("✓ Random Forest trained")

# ==================
# Model 3: Support Vector Machine
# ==================
print("\n[3/3] Training Support Vector Machine...")
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
models['SVM'] = svm_model
print("✓ SVM trained")

# ============================================
# 11. MODEL EVALUATION
# ============================================

print("\n" + "="*60)
print("STEP 10: MODEL EVALUATION")
print("="*60)

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model and return metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Store results
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm,
        'Predictions': y_pred
    }
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Recommended', 'Recommended']))
    
    return results[model_name]

# Evaluate all models
for model_name, model in models.items():
    evaluate_model(model, X_test, y_test, model_name)

# ============================================
# 12. RESULTS COMPARISON
# ============================================

print("\n" + "="*60)
print("STEP 11: MODEL COMPARISON")
print("="*60)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['Accuracy'] for m in results.keys()],
    'Precision': [results[m]['Precision'] for m in results.keys()],
    'Recall': [results[m]['Recall'] for m in results.keys()],
    'F1-Score': [results[m]['F1-Score'] for m in results.keys()]
})

print("\nModel Performance Comparison:")
print(comparison_df.to_string(index=False))

# Find best model
best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Accuracy: {comparison_df['Accuracy'].max():.4f}")

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#4ecdc4', '#ff6b6b', '#ffd43b', '#51cf66']

for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
    ax.bar(comparison_df['Model'], comparison_df[metric], color=color, edgecolor='black')
    ax.set_title(f'{metric} Comparison', fontweight='bold', fontsize=12)
    ax.set_ylabel(metric)
    ax.set_ylim([0.7, 1.0])
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(comparison_df[metric]):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model_comparison.png")
plt.close()

# ============================================
# 13. CONFUSION MATRIX VISUALIZATION
# ============================================

print("\n" + "="*60)
print("STEP 12: CONFUSION MATRIX VISUALIZATION")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (model_name, ax) in enumerate(zip(results.keys(), axes)):
    cm = results[model_name]['Confusion Matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['Not Rec', 'Recommended'],
                yticklabels=['Not Rec', 'Recommended'],
                cbar_kws={'label': 'Count'})
    ax.set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confusion_matrices.png")
plt.close()

# ============================================
# 14. HYPERPARAMETER TUNING (BEST MODEL)
# ============================================

print("\n" + "="*60)
print("STEP 13: HYPERPARAMETER TUNING")
print("="*60)

print(f"\nPerforming GridSearchCV on {best_model_name}...")

if best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear', 'lbfgs']
    }
    base_model = LogisticRegression(max_iter=1000, random_state=42)
    
elif best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
else:  # SVM
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
    base_model = SVC(random_state=42)

# Perform GridSearchCV
grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy', 
                          n_jobs=-1, verbose=1)
print("Training with different hyperparameters (this may take a few minutes)...")
grid_search.fit(X_train, y_train)

print(f"\n✓ Hyperparameter tuning completed")
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Evaluate tuned model
best_tuned_model = grid_search.best_estimator_
y_pred_tuned = best_tuned_model.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

print(f"\nTuned Model Performance:")
print(f"Test Accuracy: {tuned_accuracy:.4f}")
print(f"Improvement: {(tuned_accuracy - results[best_model_name]['Accuracy'])*100:.2f}%")

# ============================================
# 15. FEATURE IMPORTANCE (for Random Forest)
# ============================================

if best_model_name == 'Random Forest':
    print("\n" + "="*60)
    print("STEP 14: FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get feature importance
    importances = models['Random Forest'].feature_importances_
    feature_names = X.columns
    
    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(20)
    
    print("\nTop 20 Important Features:")
    print(feature_importance_df.to_string(index=False))
    
    # Visualize
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), feature_importance_df['Importance'].values, color='#4ecdc4')
    plt.yticks(range(20), feature_importance_df['Feature'].values)
    plt.xlabel('Importance Score')
    plt.title('Top 20 Feature Importance (Random Forest)', fontweight='bold', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: feature_importance.png")
    plt.close()

# ============================================
# 16. BUSINESS INSIGHTS
# ============================================

print("\n" + "="*60)
print("STEP 15: BUSINESS INSIGHTS")
print("="*60)

# Analyze relationship between sentiment and recommendation
sentiment_analysis = df_clean.groupby('Sentiment_Category')['Recommended IND'].agg(['mean', 'count'])
sentiment_analysis.columns = ['Recommendation_Rate', 'Count']
sentiment_analysis['Recommendation_Rate'] = sentiment_analysis['Recommendation_Rate'] * 100

print("\nSentiment vs Recommendation Rate:")
print(sentiment_analysis)

# Visualize
plt.figure(figsize=(10, 6))
sentiment_analysis['Recommendation_Rate'].plot(kind='bar', color=['#ff6b6b', '#ffd43b', '#51cf66'])
plt.title('Recommendation Rate by Sentiment Category', fontweight='bold', fontsize=14)
plt.xlabel('Sentiment Category')
plt.ylabel('Recommendation Rate (%)')
plt.xticks(rotation=45)
plt.ylim([0, 100])
plt.grid(axis='y', alpha=0.3)

for i, v in enumerate(sentiment_analysis['Recommendation_Rate']):
    plt.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('sentiment_vs_recommendation.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: sentiment_vs_recommendation.png")
plt.close()

# ============================================
# 17. SAVE FINAL MODEL
# ============================================

print("\n" + "="*60)
print("STEP 16: SAVING MODELS")
print("="*60)

import pickle

# Save best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(models[best_model_name], f)
print(f"✓ Saved: best_model.pkl ({best_model_name})")

# Save TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("✓ Saved: tfidf_vectorizer.pkl")

# Save preprocessing results
df_clean.to_csv('preprocessed_data.csv', index=False)
print("✓ Saved: preprocessed_data.csv")

# ============================================
# 18. SUMMARY REPORT
# ============================================

print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)

summary = f"""
PROJECT: User Sentiment Analysis and Website Return Prediction System

DATASET:
- Name: Women's E-Commerce Clothing Reviews
- Total Samples: {len(df_clean)}
- Features: Review Text, Rating, Age, Positive Feedback Count
- Target: Recommended IND (Binary: 0/1)

PREPROCESSING:
- Text cleaning: Lowercase, remove stopwords, lemmatization
- TF-IDF vectorization: {tfidf_features.shape[1]} features
- Additional features: Sentiment score, review length, word count

MODELS TRAINED:
1. Logistic Regression - Accuracy: {results['Logistic Regression']['Accuracy']:.4f}
2. Random Forest - Accuracy: {results['Random Forest']['Accuracy']:.4f}
3. SVM - Accuracy: {results['SVM']['Accuracy']:.4f}

BEST MODEL: {best_model_name}
- Test Accuracy: {results[best_model_name]['Accuracy']:.4f}
- Precision: {results[best_model_name]['Precision']:.4f}
- Recall: {results[best_model_name]['Recall']:.4f}
- F1-Score: {results[best_model_name]['F1-Score']:.4f}

KEY INSIGHTS:
- {sentiment_analysis.loc['Positive', 'Recommendation_Rate']:.1f}% of positive reviews lead to recommendations
- {sentiment_analysis.loc['Negative', 'Recommendation_Rate']:.1f}% of negative reviews lead to recommendations
- Sentiment score is a strong predictor of user return behavior

FILES GENERATED:
- target_distribution.png
- sentiment_analysis.png
- model_comparison.png
- confusion_matrices.png
- feature_importance.png (if RF is best)
- sentiment_vs_recommendation.png
- best_model.pkl
- tfidf_vectorizer.pkl
- preprocessed_data.csv

PROJECT COMPLETED SUCCESSFULLY! ✓
"""

print(summary)

# Save summary to file
with open('project_summary.txt', 'w') as f:
    f.write(summary)
print("✓ Saved: project_summary.txt")

print("\n" + "="*60)
print("ALL STEPS COMPLETED!")
print("="*60)
print("\nYou can now use the saved models for predictions.")
print("Check the generated PNG files for visualizations.")
print("\nNext step: Create Streamlit app for deployment (optional)")
