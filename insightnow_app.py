import streamlit as st
from transformers import pipeline

# Load models
@st.cache_resource
def load_pipelines():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")  # EN → FR
    return summarizer, translator

summarizer, translator = load_pipelines()

# Web App UI
st.title("InsightNow - Breaking Barriers with Smart Summaries")
st.subheader("Break language barriers with AI-powered summaries and translations.")

# Input section
user_input = st.text_area("Enter text to summarize and translate:", height=200)

# Checkbox for translation
translate_option = st.checkbox("Translate summarized text to French?")

if st.button("Process"):
    if user_input.strip():
        with st.spinner("Summarizing..."):
            # Step 1: Summarize
            summary = summarizer(user_input, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            st.success("Summary:")
            st.write(summary)
        
        # Step 2: Translate if user selected the option
        if translate_option:
            with st.spinner("Translating to French..."):
                translated = translator(summary)[0]['translation_text']
                st.success("Translated Summary (French):")
                st.write(translated)

    else:
        st.warning("Please enter some text.")

# Footer
st.markdown("---")
st.caption("Powered by Streamlit and Hugging Face Transformers.")
