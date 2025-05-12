import os
import pickle
# Set environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
import numpy as np
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean, remove_stopwords
from user_interface import create_contact_section

def load_model_and_params():
    """Load the model and its parameters"""
    try:
        # Get absolute path to src directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to project root, then into models directory
        project_root = os.path.dirname(script_dir)
        models_dir = os.path.join(project_root, 'models')
        
        # Check if models directory exists
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            raise Exception(f"Models directory created at {models_dir}. Please add model files.")
            
        # Load model parameters
        params_path = os.path.join(models_dir, 'model_params.json')
        if not os.path.exists(params_path):
            raise Exception(f"model_params.json not found at {params_path}")
            
        with open(params_path, 'r') as f:
            model_params = json.load(f)
        
        # Load available models with absolute paths
        models = {
            'Genetic LSTM': os.path.join(models_dir, 'genetic_lstm.h5'),
            'PSO LSTM': os.path.join(models_dir, 'pso_lstm.h5')
        }
        
        available_models = {}
        for name, path in models.items():
            if os.path.exists(path):
                try:
                    if path.endswith('.h5'):
                        model = tf.keras.models.load_model(
                            path,
                            custom_objects=None,
                            compile=False
                        )
                    else:
                        model = tf.saved_model.load(path)
                    available_models[name] = model
                    print(f"Successfully loaded model: {name}")
                except Exception as model_error:
                    print(f"Error loading model {name} from {path}: {str(model_error)}")
                    continue
            else:
                print(f"Model file not found: {path}")
        
        if not available_models:
            raise Exception(f"No models found in {models_dir}")
            
        return available_models, model_params
    except Exception as e:
        raise Exception(f"Error loading models: {str(e)}")

def main():
    st.title("Text Category Classifier🤖")
    
    try:
        # Load models first
        models, model_params = load_model_and_params()
        
        # Add model selection
        model_name = st.selectbox("Select Model", list(models.keys()))
        model = models[model_name]
        
        # Then load the appropriate tokenizer
        try:
            # Get the models directory path again
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            models_dir = os.path.join(project_root, 'models')
            
            tokenizers = {
                'Genetic LSTM': os.path.join(models_dir, 'genetic_tokenizer.pickle'),
                'PSO LSTM': os.path.join(models_dir, 'pso_tokenizer.pickle')
            }
            with open(tokenizers[model_name], 'rb') as handle:
                tokenizer = pickle.load(handle)
        except Exception as e:
            raise Exception(f"Error loading tokenizer: {str(e)}")
        
        categories = {
            0: "Politics",
            1: "Sport",
            2: "Technology",
            3: "Entertainment",
            4: "Business"
        }
        
        input_text = st.text_area("Enter your text:", height=150)
        
        if st.button("Classify"):
            if input_text:
                # Preprocess input
                processed_text = remove_stopwords(clean(input_text))
                sequence = tokenizer.texts_to_sequences([processed_text])
                padded_sequence = pad_sequences(sequence, maxlen=model_params['input_shape'])
                
                # Make prediction
                if isinstance(model, tf.keras.Model):
                    prediction = model.predict(padded_sequence)
                else:
                    # For SavedModel format
                    prediction = model(tf.constant(padded_sequence))
                    if isinstance(prediction, dict):
                        prediction = prediction['output_0']
                
                predicted_class = np.argmax(prediction[0])
                confidence = float(prediction[0][predicted_class]) * 100
                
                # Display results
                st.success(f"Category: {categories[predicted_class]}")
                st.info(f"Confidence: {confidence:.2f}%")
                
                # Display probability distribution
                st.write("Probability Distribution:")
                for i, prob in enumerate(prediction[0]):
                    prob_value = float(prob) * 100
                    st.progress(prob_value/100)
                    st.write(f"{categories[i]}: {prob_value:.2f}%")
                
            else:
                st.warning("Please enter text to classify.")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please make sure all model files are in the correct location.")
            
    # Add contact section
    create_contact_section()

if __name__ == "__main__":
    main()