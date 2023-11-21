import streamlit as st
import pandas as pd
from imports import *

header = st.container()
option = st.container()

with header:    
    st.title('Integrating Transformers with Pointer Generator Networks to Generate News snippets')
    st.text('By Extractors')
    
with option:
    st.text('Choose a model to run inference on : ')
    
    if "model_name" not in st.session_state:
        st.session_state.model_name = "No model selected"
        
    if "summary" not in st.session_state:
        st.session_state.summary = ""
        
    if "pgen" not in st.session_state:
        st.session_state.pgen = []
        
    if "tokens" not in st.session_state:
        st.session_state.tokens = []
    
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
    
    if col1.button('T5S - PTR'):
        st.session_state.model_name = "T5-Ptr-Gen"  
           
    if col2.button('T5S - INS'):
        st.session_state.model_name = "T5-Inshorts"
          
    if col3.button('T5S - CNN'):
        st.session_state.model_name = "T5-CNN" 
        
    if col4.button('BRT - PTR'):
        st.session_state.model_name = "Bart-Ptr-Gen"  
        
    if col5.button('BRT - INS'):
        st.session_state.model_name = "Bart-Inshorts"
        
    if col6.button('BRT - CNN'):
        st.session_state.model_name = "Bart-CNN"
        
    st.text(st.session_state.model_name)
    
    article = st.text_area('Enter the article to summarize : ', value="", height=200)        
    if st.button('Summarize'):
        if article:
            if st.session_state.model_name == "T5-Ptr-Gen":
                with st.spinner('Summarizing...'):
                    summary = generate_summary(st.session_state.model_name, article)
                    st.write(summary)
                    
            elif st.session_state.model_name == "T5-Inshorts":
                with st.spinner('Summarizing...'):
                    summary = generate_summary(st.session_state.model_name, article)
                    st.write(summary)
                    
            elif st.session_state.model_name == "T5-CNN":
                with st.spinner('Summarizing...'):
                    summary = generate_summary(st.session_state.model_name, article)
                    st.write(summary)
                    
            elif st.session_state.model_name == "Bart-Ptr-Gen":
                with st.spinner('Summarizing...'):
                    summary = generate_summary(st.session_state.model_name, article)
                    st.write(summary)
                    
            elif st.session_state.model_name == "Bart-Inshorts":
                with st.spinner('Summarizing...'):
                    summary = generate_summary(st.session_state.model_name, article)
                    st.write(summary)
                    
            elif st.session_state.model_name == "Bart-CNN":
                with st.spinner('Summarizing...'):
                    summary = generate_summary(st.session_state.model_name, article)
                    st.write(summary)
                    
            else:
                st.write("Please choose a model to run inference on")
               
            # Generate analysis on the pgen scores vs the summary 
            if st.session_state.model_name == "T5-Ptr-Gen" or st.session_state.model_name == "Bart-Ptr-Gen":
                with st.spinner('Generating pgen scores...'):
                    st.session_state.tokens, st.session_state.pgen = generate_pgen_scores(st.session_state.model_name, article)
                    df = pd.DataFrame(
                        {
                            "Tokens": st.session_state.tokens,
                            "Pgen": st.session_state.pgen
                        }
                    )
                    
                    st.write(df)
                    
        else:
            st.write("Please enter an article to summarize")                
    