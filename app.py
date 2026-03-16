import numpy as np
import os
import cv2
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from flask import Flask, render_template, Response

# =============== TRAINING SECTION ===============
def train_model():
    """Train the CNN model if it doesn't exist"""
    print("\n=== Starting Model Training ===")
    
    train_data = ImageDataGenerator(
        rescale=1./255,
        rotation_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)
    
    traindata_load = train_data.flow_from_directory(
        "C:/Users/D.RAHUL/Downloads/datasets",
        target_size=(180, 180),
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        subset="training",
        seed=42
    )
    
    valdata_load = train_data.flow_from_directory(
        "C:/Users/D.RAHUL/Downloads/datasets",
        target_size=(180, 180),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
        subset="validation",
        seed=42
    )
    
    num_classes = traindata_load.num_classes
    
    # Build model
    model = models.Sequential([
        layers.Input(shape=(180, 180, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2), 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2), 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2), 2),
        layers.Flatten(),
        layers.Dense(units=128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train model
    model.fit(traindata_load,
              epochs=10,
              validation_data=valdata_load)
    
    # Save model
    model.save("cnn_model.keras")
    print("Training complete & model saved!")
    
    # Evaluate on validation data
    loss, acc = model.evaluate(valdata_load)
    print("Accuracy Score:", acc)
    
    return model, traindata_load

# =============== TESTING SECTION (IMAGE INPUT) ===============
def test_image(model):
    """Test model with single image input"""
    print("\n=== Image Testing Mode ===")
    path = input("Enter image path: ")
    
    if not os.path.exists(path):
        print("Image path not found!")
        return
    
    img = image.load_img(path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)
    predicted_class_idx = np.argmax(pred)
    predicted_class = ["cat", "dog"][predicted_class_idx]
    confidence = np.max(pred)
    
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

# =============== FLASK + OPENCV REAL-TIME SECTION ===============
app = Flask(__name__)
camera = None
model = None
class_labels = ["cat", "dog"]

def generate_frames():
    """Generate frames for video streaming"""
    global camera, model
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Prepare image for prediction
        img = cv2.resize(frame, (180, 180))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        pred = model.predict(img)
        label = class_labels[np.argmax(pred)]
        conf = np.max(pred)
        
        # Display prediction on frame
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render index page"""
    return render_template('index.html')

@app.route('/video')
def video():
    """Stream video feed"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_flask_app():
    """Start Flask web application for real-time inference"""
    global camera, model
    
    print("\n=== Starting Flask Web App ===")
    print("Opening camera...")
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        print("Error: Could not open camera!")
        return
    
    print("Flask app running at http://127.0.0.1:5000/")
    app.run(debug=False)

# =============== MAIN EXECUTION ===============
if __name__ == '__main__':
    print("\n" + "="*50)
    print("CNN Cat vs Dog Prediction System")
    print("="*50)
    
    model = None
    traindata_load = None
    
    # Check if model exists
    if not os.path.exists("cnn_model.keras"):
        print("\nModel not found! Training a new model...")
        model, traindata_load = train_model()
    else:
        print("\nModel found! Loading existing model...")
        model = load_model("cnn_model.keras")
    
    # Directly start Flask app with camera
    start_flask_app()

