import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="Shakespeare Word Predictor", page_icon="🎭", layout="centered")

# -------------------------------------------------
# Load Models + Tokenizer
# -------------------------------------------------
@st.cache_resource
def load_assets():
    try:
        lstm = load_model("final_next_word_lstm.keras")
        gru = load_model("final_next_word_gru.keras")

        with open("tokenizer_final.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)

        # Create reverse lookup dictionary (faster than looping)
        index_to_word = {index: word for word, index in tokenizer.word_index.items()}

        return lstm, gru, tokenizer, index_to_word

    except Exception as e:
        st.error("Error loading models or tokenizer. Ensure .keras models and tokenizer file are in the same folder.")
        st.error(f"Details: {e}")
        st.stop()

lstm_model, gru_model, tokenizer, index_to_word = load_assets()

# -------------------------------------------------
# Helper Function
# -------------------------------------------------
def preprocess_text(text, tokenizer, max_sequence_len):
    """Tokenizes, truncates, and pads the input text to match model shape."""
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
        
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    return token_list

# -------------------------------------------------
# Prediction Functions
# -------------------------------------------------
def predict_next_word(model, tokenizer, index_to_word, text, max_sequence_len):
    token_list = preprocess_text(text, tokenizer, max_sequence_len)
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]
    return index_to_word.get(predicted_index, None)

def predict_top_words(model, tokenizer, index_to_word, text, max_sequence_len, top_n=3):
    token_list = preprocess_text(text, tokenizer, max_sequence_len)
    predicted_probs = model.predict(token_list, verbose=0)[0]
    
    # Get indices of the top N probabilities
    top_indices = predicted_probs.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        word = index_to_word.get(idx)
        prob = predicted_probs[idx]
        if word:
            results.append((word, prob))
            
    return results

def generate_text(model, tokenizer, index_to_word, seed_text, next_words, max_sequence_len):
    output_text = seed_text
    for _ in range(next_words):
        next_word = predict_next_word(model, tokenizer, index_to_word, output_text, max_sequence_len)
        if next_word is None:
            break
        output_text += " " + next_word
    return output_text

# -------------------------------------------------
# UI Design
# -------------------------------------------------
st.title("🎭 Shakespearean Next Word Predictor")

st.markdown(
"""
This application uses **Deep Learning language models** trained on  
Shakespeare's *Hamlet* to predict the **next word in a sequence**.

You can compare predictions from:
- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Unit)**
"""
)

# Sidebar
st.sidebar.header("Model Configuration")

model_choice = st.sidebar.radio(
    "Select Model Architecture",
    ("LSTM", "GRU")
)

num_words_generate = st.sidebar.slider("Words to Generate (For Sentence Mode)", 1, 5, 1)

active_model = lstm_model if model_choice == "LSTM" else gru_model
max_sequence_len = active_model.input_shape[1] + 1

# -------------------------------------------------
# Input
# -------------------------------------------------
st.subheader("Enter a phrase")
input_text = st.text_input("Starting phrase:", "To be or not to")

# Layout columns for buttons
col1, col2 = st.columns(2)

# -------------------------------------------------
# Action: Predict Single Word
# -------------------------------------------------
with col1:
    if st.button("Predict Next Word", type="primary", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner(f"Consulting the {model_choice} model..."):
                predicted_word = predict_next_word(active_model, tokenizer, index_to_word, input_text, max_sequence_len)
                
                if predicted_word:
                    st.success(f"Predicted Next Word: **{predicted_word}**")
                    st.info(f"Complete Phrase: **{input_text} {predicted_word}**")
                    
                    st.subheader("Top 3 Predictions")
                    top_words = predict_top_words(active_model, tokenizer, index_to_word, input_text, max_sequence_len)
                    for word, prob in top_words:
                        # Use a progress bar to visually represent the probability
                        st.write(f"**{word}** ({prob:.1%})")
                        st.progress(float(prob))
                else:
                    st.error("Prediction failed.")

# -------------------------------------------------
# Action: Generate Sentence
# -------------------------------------------------
with col2:
    if st.button("Generate Sentence", use_container_width=True):
        if not input_text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner(f"Generating {num_words_generate} words..."):
                generated_text = generate_text(active_model, tokenizer, index_to_word, input_text, num_words_generate, max_sequence_len)
                st.success(generated_text)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption(
"""
Built by **Sudhanshu** with **TensorFlow / Keras + Streamlit** Models: **LSTM vs GRU comparison**
"""
)