from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Chess Piece Classification API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a base model without data augmentation
def create_model():
    base_model = tf.keras.applications.VGG19(include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(6),
        tf.keras.layers.Activation('softmax')
    ])
    
    return model

# Initialize the model
model = create_model()

# Load the weights
try:
    model.load_weights('/app/models/Checkpoint.keras')
except:
    raise HTTPException(status_code=500, detail="Failed to load model weights")

# Class names
class_names = ["bishop", "king", "knight", "pawn", "queen", "rook"]

def preprocess_image(image: Image.Image):
    # Resize image to 224x224
    image = image.resize((224, 224))
    # Convert to array and expand dimensions
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)
    # Normalize pixel values
    image_array = image_array / 255.0
    return image_array

@app.get("/")
async def root():
    return {"message": "Chess Piece Classification API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and verify the image
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))
        
        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                class_name: float(prob) 
                for class_name, prob in zip(class_names, prediction[0])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)