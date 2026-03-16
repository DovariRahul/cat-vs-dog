# Cat vs Dog Live Prediction Project

## Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify images as either cats or dogs. It includes:
- A **training script** to build and train the CNN model on cat and dog datasets
- A **Flask web application** with real-time webcam integration for live predictions
- A **pre-trained model** ready for immediate inference

The system uses TensorFlow/Keras for deep learning, Flask for web serving, and OpenCV for real-time video processing. Users can see predictions displayed directly on their webcam feed with confidence scores.

---

## Project Structure

```
cat vs dog-prediction/
├── README.md                  # This file
├── cnn_demo.py               # Training script for the CNN model
├── cnn_demo2.py              # Flask app for live prediction with webcam
├── cnn_model.keras           # Pre-trained model (Keras format)
├── cnn_model.h5              # Pre-trained model (HDF5 format)
├── zip_cnn_model.py          # Utility script to compress the model
└── templates/
    └── index.html            # Web interface for live prediction
```

### File Descriptions

| File | Purpose |
|------|---------|
| `cnn_demo.py` | Training script that builds CNN, applies data augmentation, trains on cat/dog dataset, and saves the model |
| `cnn_demo2.py` | Flask web application serving live webcam predictions with real-time inference |
| `cnn_model.keras` | Trained model in Keras format (recommended format) |
| `cnn_model.h5` | Trained model in HDF5 format (alternative) |
| `zip_cnn_model.py` | Compresses the model file for easier distribution |
| `templates/index.html` | Simple HTML template displaying the live video stream |

---

## Prerequisites & Requirements

Before running this project, ensure you have:

### System Requirements
- **Python**: 3.8 or higher
- **Webcam**: Required for live prediction (cnn_demo2.py)
- **RAM**: At least 4GB (for training), 2GB (for inference)
- **GPU**: Optional but recommended for faster training (NVIDIA GPU with CUDA support)

### Python Libraries
Install the required packages using pip:

```bash
pip install tensorflow
pip install keras
pip install flask
pip install opencv-python
pip install numpy
pip install scikit-learn
```

Or install all at once:

```bash
pip install tensorflow keras flask opencv-python numpy scikit-learn
```

---

## Installation & Setup

### Step 1: Clone or Download the Project
```bash
cd "path\to\cat vs dog-prediction"
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually install (if requirements.txt is not available):
```bash
pip install tensorflow keras flask opencv-python numpy scikit-learn
```

### Step 4: Prepare Your Dataset (If Training From Scratch)
Create a dataset folder structure like:
```
datasets/
├── cat/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
└── dog/
    ├── 001.jpg
    ├── 002.jpg
    └── ...
```

Place your dataset images in the appropriate folders.

---

## Running the Project

### Option 1: Use Pre-trained Model (Recommended for Quick Start)

#### Running Live Prediction with Webcam:

```bash
python cnn_demo2.py
```

**Steps:**
1. Open your terminal/command prompt in the project directory
2. Run the command above
3. The Flask server will start (usually at `http://127.0.0.1:5000/`)
4. Open your web browser and navigate to `http://localhost:5000/`
5. You'll see your webcam feed with real-time cat/dog predictions
6. Each prediction includes:
   - **Label**: "cat" or "dog"
   - **Confidence**: Probability score (0-1)
7. To stop the server: Press `CTRL + C` in the terminal

**Expected Output:**
```
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

---

### Option 2: Train Your Own Model

#### Prerequisites for Training:
- Download a cat/dog dataset (e.g., from Kaggle or Microsoft)
- Update the dataset path in `cnn_demo.py` (line 24):
  ```python
  "C:/Users/D.RAHUL/Downloads/datasets"  # Change this path
  ```

#### Training Steps:

1. **Update Dataset Path:**
   Open `cnn_demo.py` and modify the dataset path to your location:
   ```python
   traindata_load = train_data.flow_from_directory(
       "YOUR_DATASET_PATH",  # Replace with your path
       target_size=(180,180),
       batch_size=32,
       ...
   )
   ```

2. **Run Training Script:**
   ```bash
   python cnn_demo.py
   ```

3. **Monitor Training:**
   - The script will display epoch-wise accuracy and loss
   - Training typically takes 10-30 minutes depending on dataset size and hardware
   - The trained model saves automatically

4. **Expected Output:**
   ```
   Found X images belonging to 2 classes.
   Found Y images belonging to 2 classes.
   Epoch 1/10
   [============================] - Xs - loss: 0.6234 - accuracy: 0.6543
   ...
   Epoch 10/10
   [============================] - Xs - loss: 0.1234 - accuracy: 0.9567
   ```

#### Model Architecture:

The CNN uses:
- **Input Layer**: 180×180×3 (RGB images)
- **Conv Block 1**: 32 filters, 3×3 kernel + Max Pooling
- **Conv Block 2**: 64 filters, 3×3 kernel + Max Pooling
- **Conv Block 3**: 128 filters, 3×3 kernel + Max Pooling
- **Flatten + Dense**: 128 neurons with ReLU
- **Output**: Softmax activation for 2 classes (cat/dog)

---

## Detailed Running Process

### Live Prediction Workflow (cnn_demo2.py):

```
1. Start Flask Server
   ↓
2. Load Pre-trained Model (cnn_model.keras)
   ↓
3. Initialize Webcam (OpenCV)
   ↓
4. For Each Frame:
   a. Capture frame from webcam
   b. Resize to 180×180 pixels
   c. Normalize pixel values (divide by 255)
   d. Add batch dimension
   e. Pass through CNN model
   f. Get prediction (cat or dog) and confidence
   g. Draw label and confidence on frame
   h. Stream encoded frame to browser
   ↓
5. Display in Web Browser (HTML)
```

### Training Workflow (cnn_demo.py):

```
1. Load Dataset with Image Augmentation
   - Rescaling (normalize to 0-1)
   - Rotation, shifts, zoom
   - Horizontal flipping
   ↓
2. Split into Training (80%) and Validation (20%)
   ↓
3. Build CNN Model
   ↓
4. Compile with Adam Optimizer
   - Loss: Categorical Crossentropy
   - Metric: Accuracy
   ↓
5. Train for Multiple Epochs
   - Each epoch: process all training samples
   - Validate on validation set
   - Adjust weights based on loss
   ↓
6. Save Trained Model
```

---

## Usage Examples

### Example 1: Run Live Predictions Immediately
```bash
# Navigate to project directory
cd "cat vs dog-prediction"

# Run the Flask app
python cnn_demo2.py

# Open browser and go to http://localhost:5000/
```

### Example 2: Train Model with Custom Dataset
```bash
# Update dataset path in cnn_demo.py
python cnn_demo.py

# After training completes, test with:
python cnn_demo2.py
```

### Example 3: Compress Model for Sharing
```bash
# Run this to create model.zip
python zip_cnn_model.py

# A model.zip file will be created
```

---

## Troubleshooting

### Issue: "No module named 'tensorflow'"
**Solution:**
```bash
pip install tensorflow --upgrade
```

### Issue: "Webcam not detected" or "OpenCV error"
**Solution:**
- Ensure your webcam is properly connected and working
- Check if another application is using the webcam
- Try granting camera permissions to Python
- Update OpenCV:
  ```bash
  pip install opencv-python --upgrade
  ```

### Issue: "Model file not found (cnn_model.keras)"
**Solution:**
- Ensure `cnn_model.keras` exists in the project root
- If missing, train a new model using `cnn_demo.py`
- Or download from the source repository

### Issue: "Port 5000 already in use"
**Solution:**
- Close other Flask applications
- Or modify the port in `cnn_demo2.py`:
  ```python
  app.run(debug=True, port=5001)  # Use different port
  ```

### Issue: "Low accuracy on training"
**Solution:**
- Increase training epochs in `cnn_demo.py`
- Use a larger, more diverse dataset
- Adjust learning rate in optimizer
- Ensure dataset is properly labeled

### Issue: "Slow predictions on CPU"
**Solution:**
- Use GPU if available (NVIDIA GPU with CUDA)
- Reduce model complexity if needed
- Optimize image preprocessing

---

## Advanced Customization

### Change Input Image Size:
In both files, modify the size (currently 180×180):
```python
target_size=(180,180)  # Change to (224,224), (256,256), etc.
```

### Change Class Labels:
In `cnn_demo2.py`:
```python
class_labels = ["cat", "dog"]  # Modify as needed
```

### Adjust Model Architecture:
In `cnn_demo.py`, modify the Sequential layers:
```python
layers.Conv2D(32, (3,3), activation='relu')  # Add/remove layers
layers.Dropout(0.5)  # Add dropout for regularization
```

---

## Key Technologies

- **TensorFlow/Keras**: Deep learning framework
- **Flask**: Web framework for serving predictions
- **OpenCV**: Real-time video processing
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning utilities

---

## Performance Notes

- **Training Time**: 10-30 minutes on CPU, 2-5 minutes on GPU (depends on dataset size)
- **Inference Speed**: Real-time (30+ FPS on modern hardware)
- **Model Size**: ~1-5 MB (depending on format)
- **Memory Usage**: 500 MB - 2 GB (training), 100-500 MB (inference)

---

## License

This project uses open-source libraries. Ensure compliance with their respective licenses.

---

## Support & Help

For issues or improvements:
1. Check the Troubleshooting section
2. Review code comments
3. Consult TensorFlow and Flask documentation
4. Ensure all dependencies are correctly installed

---

## Future Enhancements

- Add support for multiple animals
- Integrate model export to ONNX format
- Add batch prediction on image folders
- Create REST API endpoints
- Deploy on cloud platforms (AWS, Azure, GCP)
- Add model visualization and attention maps

---

**Last Updated:** March 2026  
**Project Type:** Computer Vision / Deep Learning
