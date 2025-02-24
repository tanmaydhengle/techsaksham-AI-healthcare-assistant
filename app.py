import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


# Load a pre-trained Hugging Face model
chatbot = pipeline("questuion-answering", model="deepset/bert-base-cased-squad2")


# Define healthcare-specific response logic (or use a model to generate responses)
def preprocess_input(user_input):
    stop_words= set(stopwords.word("english"))
    words = word_tokenize(user_input)
    filtered_words= [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

    # Simple rule-based keywords to respond
    def healthcare_chatbot(user_input):
        user_input = preprocess_input(user_input).lower()
    if "sneeze" in user_input:
        return "Frequent sneezing my indicate allergies or cold. Consult doctor if symptoms persist"
    elif "symptom" in user_input:
        return "it seems like you are experiencing symptoms. Please consult a doctor for accurate advice"
    elif "appointment" in user_input:
        return "would you like me to schedule an appointment with a doctor?"
    elif "medication" in user_input:
        return "It's important to take your prescribed medications regularly. If you have concerns, consult your doctor."
    else:
        context="""
        Common healthcare related scenarios include Symptoms of colds, flu, and allergies,
        along with medication guidance and appointment scheduling.
        """
        response = chatbot(question=user_input , context= context)
        return response['answer']


# Streamlit web app interface
def main():
    # Set up the web app title and input area
    st.title("Healthcare Assistant Chatbot")
    
    # Display a simple text input for user queries
    user_input = st.text_input("How can I assist you today?", "")
    
    # Display chatbot response
    if st.button("Submit"):
        if user_input:
            st.write("User: ", user_input)
            response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant: ", response)
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    main()
