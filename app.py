import streamlit as st
from crew import run_crew

st.title("Honey Badger Swarm")
topic = st.text_input("Enter topic to research:")

if st.button("Deploy Swarm"):
    with st.spinner("Agents are working..."):
        result = run_crew(topic)
        st.write(result)
