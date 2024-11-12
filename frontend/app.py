import streamlit as st
import requests
from PIL import Image
import io
import time
import plotly.graph_objects as go
import numpy as np
import json

st.set_page_config(
    page_title="Chess Piece Classifier",
    page_icon="♟️",
    layout="wide"
)

def main():
    st.title("Chess Piece Classification")
    st.write("Upload an image of a chess piece to classify it!")

    # Sidebar with metrics
    st.sidebar.title("Performance Metrics")
    latency_placeholder = st.sidebar.empty()
    total_predictions = st.sidebar.empty()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        start_time = time.time()
        
        files = {"file": ("image.jpg", uploaded_file, "image/jpeg")}
        response = requests.post("http://localhost:8000/predict", files=files)
        
        end_time = time.time()
        latency = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Display prediction results
            col2.markdown("### Prediction Results")
            col2.write(f"**Predicted Class:** {result['class'].upper()}")
            col2.write(f"**Confidence:** {result['confidence']*100:.2f}%")
            
            # Create confidence bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(result['probabilities'].values()),
                    y=list(result['probabilities'].keys()),
                    orientation='h'
                )
            ])
            
            fig.update_layout(
                title="Confidence Scores",
                xaxis_title="Confidence",
                yaxis_title="Class",
                yaxis={'categoryorder':'total ascending'}
            )
            
            col2.plotly_chart(fig)
            
            # Update metrics
            latency_placeholder.metric("Prediction Latency", f"{latency*1000:.0f}ms")
            
            # Increment prediction counter (stored in session state)
            if 'prediction_count' not in st.session_state:
                st.session_state.prediction_count = 0
            st.session_state.prediction_count += 1
            
            total_predictions.metric("Total Predictions", st.session_state.prediction_count)
        else:
            st.error("Error making prediction. Please try again.")

if __name__ == "__main__":
    main()