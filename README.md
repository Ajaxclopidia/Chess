# Chess Piece Classification System

A deep learning-based system for classifying chess pieces using computer vision. This project implements an advanced image classification model using the VGG19 architecture, deployed through a REST API and Streamlit interface.

## Project Structure
```
chess_classifier/
├── notebooks/
│   └── model_development.ipynb
├── api/
│   ├── Dockerfile
│   ├── api.py
│   └── requirements.txt
├── frontend/
│   ├── Dockerfile.streamlit
│   ├── app.py
│   └── requirements.txt
├── models/
│   └── Checkpoint.keras
├── report/
│   └── technical_report.pdf
├── docker-compose.yml
└── README.md
```

## Features
- Deep learning model based on VGG19 architecture
- Data augmentation and preprocessing pipeline
- REST API for model inference
- Interactive Streamlit web interface
- Docker containerization for easy deployment
- Real-time performance monitoring

## Model Architecture
- Base model: VGG19 (pre-trained on ImageNet)
- Custom top layers with dropout for regularization
- Advanced data augmentation pipeline
- Achieved accuracy of 0.8984% on test set

## Prerequisites
- Docker and Docker Compose
- Python 3.9+
- GPU support (optional, but recommended)

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/chess-classifier.git
cd chess-classifier
```

2. Build and run the Docker containers:
```bash
docker-compose up --build
```

3. Access the applications:
- Streamlit Interface: http://localhost:8501
- FastAPI Documentation: http://localhost:8000/docs

## Usage

### Using the Streamlit Interface
1. Open http://localhost:8501 in your browser
2. Upload an image of a chess piece
3. View the classification results and confidence scores
4. Monitor real-time performance metrics

### Using the REST API
```python
import requests

# Upload an image
files = {'file': open('path_to_image.jpg', 'rb')}
response = requests.post('http://localhost:8000/predict', files=files)
prediction = response.json()
```

## API Documentation
The REST API provides the following endpoints:

- `GET /`: Health check endpoint
- `POST /predict`: Image classification endpoint
  - Input: Image file (JPG, PNG)
  - Output: JSON with class predictions and confidence scores

## Model Training
The model was trained using the following approach:
1. Data preprocessing and augmentation
2. Transfer learning with VGG19 base
3. Fine-tuning of top layers
4. Hyperparameter optimization

Detailed training process and results are available in `notebooks/Assessment ML Chess Submission.ipynb`.

## Performance Metrics
- Accuracy: 0.8984%

## Troubleshooting

### Common Issues

1. Docker Container Startup Issues
```bash
# Check container logs
docker-compose logs
```

2. Model Loading Errors
- Ensure model file is present in models/
- Check file permissions
- Verify TensorFlow version compatibility

3. Image Upload Issues
- Check image format (JPG/PNG supported)
- Verify image size and resolution
- Ensure proper file permissions

### Performance Optimization
- Use GPU if available
- Adjust batch size for your hardware
- Monitor memory usage

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- VGG19 model architecture
- TensorFlow/Keras team
- FastAPI framework
- Streamlit team

## Contact
For any queries or issues, please open an issue on GitHub or contact [Muhammed](mailto:ajaxclopidia77@gmail.com).