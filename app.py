"""BA Insight - Streamlit dashboard.
Run locally with:  streamlit run app.py
In the AWS design this app is deployed on Streamlit (see the Week 5
final report for the deployment documentation). It loads the trained
model from models/ba_satisfaction_model.joblib."""
import os
import pandas as pd
import streamlit as st
from src.model import predict_review
from src.recommend import recommend

def main():
    st.set_page_config(page_title='BA Insight', layout='wide')
    st.title('BA Insight - British Airways Customer Satisfaction AI')
    st.write('Predict passenger satisfaction from review text, see which service aspects drive it, and get a recommended action.')
    tab1, tab2 = st.tabs(['Try a review', 'Dashboard'])
    with tab1:
        text = st.text_area('Paste a British Airways review', 'The flight was delayed by three hours and nobody told us anything. The seats were cramped and the food was cold.', height=140)
        if st.button('Predict satisfaction'):
            result = predict_review(text)
            st.subheader('Prediction: ' + result['predicted_satisfaction'])
            st.write('VADER compound score: ' + str(result['vader_compound']) + ' (VADER label: ' + result['vader_label'] + ')')
            st.write('Aspects detected:')
            if result['aspects']:
                st.table(pd.DataFrame(list(result['aspects'].items()), columns=['Aspect', 'Sentiment']))
            else:
                st.write('No service aspect keywords were found in this review.')
            st.subheader('Recommended action')
            st.write(recommend(result))
    with tab2:
        st.subheader('Satisfaction across the 1,300 reviews')
        if os.path.exists('outputs/satisfaction_distribution.png'):
            st.image('outputs/satisfaction_distribution.png')
        st.subheader('Sentiment per service aspect')
        if os.path.exists('outputs/aspect_sentiment.png'):
            st.image('outputs/aspect_sentiment.png')
        st.subheader('Model comparison (held-out test set, n=260)')
        if os.path.exists('outputs/model_comparison.png'):
            st.image('outputs/model_comparison.png')
        if os.path.exists('outputs/aspect_summary.csv'):
            st.subheader('Aspect summary')
            st.dataframe(pd.read_csv('outputs/aspect_summary.csv'))

if __name__ == '__main__':
    main()
