import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os
import time
import io
import traceback
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder

# Set page configuration
st.set_page_config(
    page_title="Emotion Analysis Chatbot",
    page_icon="😊",
    layout="centered"
)

# Add custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .emotion-label {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    .title {
        text-align: center;
        padding-bottom: 1rem;
    }
    .emotion-angry {
        color: #e53935;
    }
    .emotion-happy {
        color: #43a047;
    }
    .emotion-love {
        color: #d81b60;
    }
    .emotion-sad {
        color: #1e88e5;
    }
</style>
""", unsafe_allow_html=True)

# Set title
st.markdown("<h1 class='title'>Emotion Analysis Chatbot</h1>", unsafe_allow_html=True)

# Define emotion color mapping
EMOTION_CLASSES = {}  # Will be dynamically populated based on model's emotions

# Function to safely get classes from model data pickle file
def extract_classes_from_pickle(filepath):
    try:
        with open(filepath, 'rb') as f:
            # Read the whole file
            content = f.read()
            
            # Search for class names pattern in binary content
            import re
            
            # Look for common emotion words
            emotions = []
            for emotion in ["anger", "happy", "happiness", "joy", "love", "sad", "sadness", "fear", "surprise"]:
                if emotion.encode() in content:
                    emotions.append(emotion)
            
            # If we found emotions, use them
            if emotions:
                return emotions
            else:
                return ["anger", "happy", "love", "sad"]  # Default fallback
    except:
        # Return default emotions if extraction fails
        return ["anger", "happy", "love", "sad"]

# Load model data from files
@st.cache_resource
def load_model_resources():
    try:
        start_time = time.time()
        
        # Check if emotion_bert_model directory exists
        if not os.path.exists('./emotion_bert_model'):
            return None, None, None, "Model directory not found", 0
        
        # Try to load model and tokenizer
        try:
            model = BertForSequenceClassification.from_pretrained('./emotion_bert_model')
            tokenizer = BertTokenizer.from_pretrained('./emotion_bert_model')
        except Exception as e:
            return None, None, None, f"Failed to load model: {str(e)}", 0
        
        # Move model to CPU (safer for Streamlit deployment)
        model.to('cpu')
        
        # Try multiple methods to get label encoder classes
        label_encoder = LabelEncoder()
        
        # Method 1: Try to get classes from JSON (most reliable)
        if os.path.exists('./emotion_bert_model/label_encoder_classes.json'):
            with open('./emotion_bert_model/label_encoder_classes.json', 'r') as f:
                label_encoder.classes_ = np.array(json.load(f))
                
        # Method 2: Try to load numpy file with pickle
        elif os.path.exists('./emotion_bert_model/label_encoder_classes.npy'):
            try:
                label_encoder.classes_ = np.load('./emotion_bert_model/label_encoder_classes.npy', allow_pickle=True)
            except Exception:
                # If numpy load fails, try to extract classes from pickle
                if os.path.exists('./emotion_bert_model/model_data.pkl'):
                    classes = extract_classes_from_pickle('./emotion_bert_model/model_data.pkl')
                    label_encoder.classes_ = np.array(classes)
                else:
                    # Final fallback - use standard emotions
                    label_encoder.classes_ = np.array(["anger", "happy", "love", "sad"])
        
        # Method 3: Use config.json from the model to determine number of labels
        elif os.path.exists('./emotion_bert_model/config.json'):
            with open('./emotion_bert_model/config.json', 'r') as f:
                config = json.load(f)
                num_labels = config.get('num_labels', 4)
                # Create generic emotion names based on number of labels
                generic_emotions = [f"emotion_{i}" for i in range(num_labels)]
                label_encoder.classes_ = np.array(generic_emotions)
        
        else:
            # Absolute fallback - use standard emotions
            label_encoder.classes_ = np.array(["anger", "happy", "love", "sad"])
        
        # Save classes as JSON for future use (more reliable)
        try:
            with open('./emotion_bert_model/label_encoder_classes.json', 'w') as f:
                json.dump(label_encoder.classes_.tolist(), f)
        except:
            pass
            
        load_time = time.time() - start_time
        return model, tokenizer, label_encoder, None, load_time
    
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}\n{traceback.format_exc()}"
        return None, None, None, error_msg, 0

# Function to predict emotion
def predict_emotion(text, model, tokenizer, label_encoder, max_length=128):
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Tokenize and encode text
        encodings = tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Move tensors to the same device as model
        input_ids = encodings['input_ids'].to(model.device)
        attention_mask = encodings['attention_mask'].to(model.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]
        
        # Check if prediction is in range of label_encoder classes
        if prediction >= len(label_encoder.classes_):
            prediction = prediction % len(label_encoder.classes_)
        
        # Convert numeric prediction to emotion label
        emotion = label_encoder.classes_[prediction]
        confidence = torch.softmax(logits, dim=1).cpu().numpy()[0][prediction]
        
        return emotion, confidence
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'resources_loaded' not in st.session_state:
    st.session_state.resources_loaded = False
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.label_encoder = None
    st.session_state.max_length = 128
    st.session_state.emotions_map = {}

# Sidebar for model information
with st.sidebar:
    st.title("Emotion Detection Model")
    
    # Load model button
    if not st.session_state.resources_loaded:
        if st.button("Load BERT Emotion Model"):
            with st.spinner("Loading model resources..."):
                model, tokenizer, label_encoder, error, load_time = load_model_resources()
                
                if model and tokenizer and label_encoder:
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.label_encoder = label_encoder
                    st.session_state.resources_loaded = True
                    
                    # Create emotion mapping
                    for i, emotion in enumerate(label_encoder.classes_):
                        st.session_state.emotions_map[i] = emotion
                        # Create CSS classes for emotions
                        emotion_lower = str(emotion).lower()
                        if 'anger' in emotion_lower or 'angry' in emotion_lower:
                            EMOTION_CLASSES[emotion] = "emotion-angry"
                        elif 'happ' in emotion_lower or 'joy' in emotion_lower:
                            EMOTION_CLASSES[emotion] = "emotion-happy"
                        elif 'love' in emotion_lower:
                            EMOTION_CLASSES[emotion] = "emotion-love"
                        elif 'sad' in emotion_lower:
                            EMOTION_CLASSES[emotion] = "emotion-sad"
                        else:
                            EMOTION_CLASSES[emotion] = ""
                    
                    st.success(f"Model loaded successfully in {load_time:.2f} seconds!")
                else:
                    st.error(f"Failed to load model: {error}")
    else:
        st.success("Model is loaded and ready!")
        # Show model info
        st.write(f"Model type: {type(st.session_state.model).__name__}")
        st.write(f"Device: {st.session_state.model.device}")
        
        # Option to unload model
        if st.button("Unload Model"):
            st.session_state.resources_loaded = False
            st.session_state.model = None
            st.session_state.tokenizer = None
            st.session_state.label_encoder = None
            st.experimental_rerun()
    
    # Show emotion classes
    if st.session_state.resources_loaded:
        st.title("Emotion Classes")
        st.write("This model detects the following emotions:")
        
        for class_idx, emotion in st.session_state.emotions_map.items():
            css_class = EMOTION_CLASSES.get(emotion, "")
            st.markdown(f"<div class='{css_class}'>Class {class_idx}: {emotion}</div>", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if 'emotion' in message and message["role"] == "assistant":
            emotion = message['emotion']
            confidence = message.get('confidence', 0)
            css_class = EMOTION_CLASSES.get(emotion, "")
            st.markdown(f"<div class='emotion-label {css_class}'>Detected emotion: {emotion} ({confidence:.1%} confidence)</div>", unsafe_allow_html=True)

# Input for user message
if prompt := st.chat_input("What's on your mind today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if model is loaded
    if st.session_state.resources_loaded:
        # Get prediction
        with st.spinner("Analyzing emotion..."):
            emotion, confidence = predict_emotion(
                prompt, 
                st.session_state.model, 
                st.session_state.tokenizer, 
                st.session_state.label_encoder,
                st.session_state.max_length
            )
        
        # Create personalized responses based on emotion
        responses = {}
        
        # Dynamically create responses based on detected emotions
        for emotion_name in st.session_state.label_encoder.classes_:
            emotion_lower = str(emotion_name).lower()
            
            if 'anger' in emotion_lower or 'angry' in emotion_lower:
                responses[emotion_name] = "I notice anger in your message. Taking a deep breath might help."
            elif 'joy' in emotion_lower or 'happy' in emotion_lower:
                responses[emotion_name] = "You seem happy! That's great to hear!"
            elif 'love' in emotion_lower:
                responses[emotion_name] = "I sense love and affection in your words. That's heartwarming!"
            elif 'sad' in emotion_lower:
                responses[emotion_name] = "I detect sadness in your message. Remember that tough times are temporary."
            else:
                responses[emotion_name] = f"I detect {emotion_name} in your message."
        
        # Get appropriate response
        response = responses.get(emotion, f"I detect {emotion} in your message.")
        
        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response, 
            "emotion": emotion,
            "confidence": confidence
        })
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
            css_class = EMOTION_CLASSES.get(emotion, "")
            st.markdown(f"<div class='emotion-label {css_class}'>Detected emotion: {emotion} ({confidence:.1%} confidence)</div>", unsafe_allow_html=True)
    else:
        # Display error if model not loaded
        with st.chat_message("assistant"):
            st.error("Please load the model first using the sidebar.")
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Please load the model first using the sidebar."
        })

# Add instructions and about section
with st.sidebar:
    st.title("Instructions")
    st.write("1. Click 'Load BERT Emotion Model' button")
    st.write("2. Type a message in the chat box")
    st.write("3. The AI will analyze the emotion in your text")
    st.write("4. View the detected emotion and confidence level")
    
    st.title("About")
    st.write("This chatbot uses a fine-tuned BERT model to analyze emotions in text.")
    st.write("The model was trained on a dataset of texts with emotion labels.")
    
    # Model structure information
    if st.session_state.resources_loaded:
        if st.checkbox("Show Model Details"):
            st.write("Model Architecture: BERT-base-uncased")
            st.write(f"Number of emotion classes: {len(st.session_state.emotions_map)}")
            st.write(f"Max sequence length: {st.session_state.max_length}")