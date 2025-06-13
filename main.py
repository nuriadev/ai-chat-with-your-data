import streamlit as st
from loaders import get_response

def main():
    st.title("AI Chat with Your Data")
    st.write("Upload a PDF or TXT file and ask questions about its content.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])
    if uploaded_file:
        question = st.text_input("Your question:")
        if question:
            with st.spinner("Thinking..."):
                answer = get_response(uploaded_file, question)
            st.markdown(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()