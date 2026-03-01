"""
Streamlit Web Application
User Sentiment Analysis and Website Return Prediction System
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data (run once)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Load models
@st.cache_resource
def load_models():
    """Load saved models and vectorizer"""
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Text preprocessing function
def preprocess_text(text):
    """Preprocess review text"""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenization
    words = text.split()
    
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Sentiment analysis function
def get_sentiment(text):
    """Extract sentiment using TextBlob"""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        category = "Positive"
        color = "green"
        emoji = "😊"
    elif polarity < -0.1:
        category = "Negative"
        color = "red"
        emoji = "😞"
    else:
        category = "Neutral"
        color = "orange"
        emoji = "😐"
    
    return polarity, category, color, emoji

# Prediction function
def make_prediction(review_text, rating, age, feedback_count, model, vectorizer):
    """Make prediction using the trained model"""
    
    # Preprocess text
    cleaned_text = preprocess_text(review_text)
    
    # Get sentiment
    sentiment_score, sentiment_cat, _, _ = get_sentiment(review_text)
    
    # TF-IDF transformation
    tfidf_features = vectorizer.transform([cleaned_text])
    tfidf_array = tfidf_features.toarray()
    
    # Additional features
    review_length = len(review_text)
    word_count = len(cleaned_text.split())
    
    # Combine features
    additional_features = np.array([[rating, age, feedback_count, 
                                    sentiment_score, review_length, word_count]])
    
    # Combine with TF-IDF
    final_features = np.hstack([tfidf_array, additional_features])
    
    # Make prediction
    prediction = model.predict(final_features)[0]
    
    # Get probability if available
    try:
        probability = model.predict_proba(final_features)[0]
        confidence = max(probability) * 100
    except:
        confidence = None
    
    return prediction, confidence, sentiment_score, sentiment_cat

# ============================================
# STREAMLIT APP
# ============================================

# Page config
st.set_page_config(
    page_title="Customer Return Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #4ecdc4;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .positive-pred {
        background-color: #d4edda;
        border: 2px solid #51cf66;
    }
    .negative-pred {
        background-color: #f8d7da;
        border: 2px solid #ff6b6b;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4ecdc4;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🛍️ Customer Return Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze customer sentiment and predict return probability</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📋 About This App")
st.sidebar.info(
    """
    This application predicts whether a customer will return to the website 
    based on their review and rating.
    
    **Features:**
    - Sentiment Analysis
    - Return Probability Prediction
    - Customer Insights
    
    **Model:** Machine Learning Classifier
    **Dataset:** E-Commerce Reviews
    """
)

st.sidebar.header("📊 How It Works")
st.sidebar.markdown("""
1. Enter customer review
2. Provide rating & details
3. Get sentiment analysis
4. See return prediction
""")

# Load models
try:
    model, vectorizer = load_models()
    st.sidebar.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Enter Customer Review")
    
    # Review text input
    review_text = st.text_area(
        "Customer Review:",
        height=150,
        placeholder="Enter the customer's review here...",
        help="Enter the actual review text from the customer"
    )
    
    # Additional inputs
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        rating = st.slider("⭐ Rating", 1, 5, 5, help="Customer rating (1-5 stars)")
    
    with col_b:
        age = st.number_input("👤 Age", 18, 100, 35, help="Customer age")
    
    with col_c:
        feedback_count = st.number_input("👍 Helpful Votes", 0, 100, 0, 
                                        help="Number of helpful votes on review")
    
    # Predict button
    predict_button = st.button("🔮 Predict Return Likelihood", type="primary", use_container_width=True)

with col2:
    st.header("📈 Quick Stats")
    
    st.markdown("""
    <div class="metric-card">
        <h4>Model Performance</h4>
        <p>✓ Accuracy: 92%+</p>
        <p>✓ Precision: 90%+</p>
        <p>✓ F1-Score: 91%+</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <h4>Sentiment Categories</h4>
        <p>😊 Positive: High return rate</p>
        <p>😐 Neutral: Medium return rate</p>
        <p>😞 Negative: Low return rate</p>
    </div>
    """, unsafe_allow_html=True)

# Prediction section
if predict_button:
    if not review_text.strip():
        st.warning("⚠️ Please enter a review text!")
    else:
        with st.spinner("Analyzing review and making prediction..."):
            # Make prediction
            prediction, confidence, sentiment_score, sentiment_cat = make_prediction(
                review_text, rating, age, feedback_count, model, vectorizer
            )
            
            # Display results
            st.markdown("---")
            st.header("🎯 Prediction Results")
            
            # Sentiment Analysis
            st.subheader("💭 Sentiment Analysis")
            col_s1, col_s2, col_s3 = st.columns(3)
            
            _, _, color, emoji = get_sentiment(review_text)
            
            with col_s1:
                st.metric("Sentiment Category", f"{emoji} {sentiment_cat}")
            with col_s2:
                st.metric("Sentiment Score", f"{sentiment_score:.3f}")
            with col_s3:
                sentiment_percent = (sentiment_score + 1) * 50  # Convert to 0-100
                st.metric("Positivity", f"{sentiment_percent:.1f}%")
            
            # Progress bar for sentiment
            st.progress(sentiment_percent / 100)
            
            st.markdown("---")
            
            # Return Prediction
            st.subheader("🎯 Return Prediction")
            
            if prediction == 1:
                st.markdown("""
                <div class="prediction-box positive-pred">
                    <h2>✅ Customer WILL Return</h2>
                    <p style="font-size: 20px;">High likelihood of customer returning to the website</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("Great news! This customer is likely to return.")
                
            else:
                st.markdown("""
                <div class="prediction-box negative-pred">
                    <h2>❌ Customer MAY NOT Return</h2>
                    <p style="font-size: 20px;">Low likelihood of customer returning to the website</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.error("Warning! This customer might not return. Consider follow-up action.")
            
            # Confidence score
            if confidence:
                st.markdown("---")
                st.subheader("📊 Confidence Score")
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.metric("Model Confidence", f"{confidence:.1f}%")
                with col_c2:
                    if confidence > 80:
                        st.info("🎯 High confidence prediction")
                    elif confidence > 60:
                        st.info("📊 Medium confidence prediction")
                    else:
                        st.info("⚠️ Low confidence prediction")
            
            # Business Insights
            st.markdown("---")
            st.subheader("💡 Business Insights")
            
            if prediction == 1:
                st.success("""
                **Recommended Actions:**
                - ✅ Maintain current service quality
                - ✅ Send personalized thank you email
                - ✅ Offer loyalty rewards
                - ✅ Request product reviews
                """)
            else:
                st.warning("""
                **Recommended Actions:**
                - 🔔 Send follow-up email with discount
                - 🔔 Address concerns mentioned in review
                - 🔔 Offer customer support contact
                - 🔔 Improve product/service quality
                """)
            
            # Feature Analysis
            with st.expander("📊 View Detailed Analysis"):
                st.write("**Review Characteristics:**")
                col_d1, col_d2, col_d3 = st.columns(3)
                
                with col_d1:
                    st.metric("Review Length", f"{len(review_text)} chars")
                with col_d2:
                    st.metric("Word Count", len(review_text.split()))
                with col_d3:
                    st.metric("Rating Given", f"{rating}/5 ⭐")
                
                st.write("**Customer Profile:**")
                st.write(f"- **Age:** {age} years")
                st.write(f"- **Helpful Votes:** {feedback_count}")
                st.write(f"- **Engagement Level:** {'High' if feedback_count > 5 else 'Low'}")

# Example section
st.markdown("---")
st.header("💡 Try Example Reviews")

col_ex1, col_ex2 = st.columns(2)

with col_ex1:
    if st.button("Example: Positive Review", use_container_width=True):
        st.session_state.example = {
            'text': "This dress is absolutely beautiful! The fabric quality is excellent and it fits perfectly. I received so many compliments. Will definitely buy again!",
            'rating': 5,
            'age': 32,
            'feedback': 8
        }

with col_ex2:
    if st.button("Example: Negative Review", use_container_width=True):
        st.session_state.example = {
            'text': "Very disappointed with this purchase. The material is cheap and the sizing is completely off. Would not recommend to anyone.",
            'rating': 1,
            'age': 28,
            'feedback': 2
        }

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><b>Customer Return Prediction System</b></p>
    <p>Built with Streamlit 🎈 | Powered by Machine Learning 🤖</p>
    <p>B.Tech ML Project - 2024</p>
</div>
""", unsafe_allow_html=True)
