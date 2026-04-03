import streamlit as st
import joblib


vectorizer=joblib.load('vectorizer.joblib')
model=joblib.load('model.joblib')

st.title("Fake News Detection")
st.write("Enter the News Article to check if it's real or fake")

news_input=st.text_area("News Article:","")

if st.button("Check News"):
    if news_input.strip():
        transform_input=vectorizer.transform([news_input])
        prediction=model.predict(transform_input)

        if prediction[0]==1:
            st.success("The news article is Real.")
        else:
            st.error("The news article is Fake.")    
    else:
        st.warning("Please enter a news article to check.")
                
