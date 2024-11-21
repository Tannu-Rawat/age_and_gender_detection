
# **Age and Gender Detection**

This repository contains an implementation of a deep learning model designed to predict **age** and **gender** from facial images. It features both a **training pipeline** using the UTKFace dataset and a **real-time prediction module** via a live camera feed. The model has been tested on several examples and accurately predicts both age and gender.

---

## **Features**
1. **Age Prediction**:
   - Predicts the approximate age of a person based on facial features.
   - Tested with real-life images, achieving accurate predictions. For example:
     - **Screenshot 2024-11-20 161133.png**: Predicts the user’s age as **18 years old female**, which matches reality.
     - **Screenshot 2024-11-21 080512.png**: Predicts the age of a young Selena Gomez (estimated 20–25 years) as **24 years**, showcasing model precision.

2. **Gender Classification**:
   - Distinguishes between **Male** and **Female** with high accuracy, achieving ~95.51% during testing.

3. **Live Camera Predictions**:
   - Runs real-time age and gender detection on a webcam feed.
   - Displays predictions directly on the video feed.

---

## **Dataset**
The model is trained on the [UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new):
- A large-scale dataset containing facial images labeled with age, gender, and ethnicity.
- The dataset includes diverse age groups and genders, enabling robust training.

---

## **Model Architecture**
The **Convolutional Neural Network (CNN)** used in this project has:
- **Four convolutional layers** for extracting hierarchical spatial features.
- **Fully connected layers** for prediction, with dropout layers to reduce overfitting.
- Dual output layers:
  - `gender_out`: **Sigmoid activation** for binary classification (Male/Female).
  - `age_out`: **ReLU activation** for continuous regression-based age prediction.

### **Training Details**
- **Input Shape**: Grayscale facial images resized to 128x128 pixels.
- **Loss Functions**:
  - `binary_crossentropy` for gender classification.
  - `mean absolute error (MAE)` for age prediction.
- **Optimizer**: Adam.
- **Metrics**: Accuracy for gender and MAE for age.

```python
# Training the model
history = model.fit(x=X, y=[y_gender, y_age], batch_size=32, epochs=30, validation_split=0.2)
```

### **Performance**
- **Gender Classification Accuracy**: 95.51%.
- **Age Prediction Mean Absolute Error (MAE)**: 3.0859 years.

---

## **Real-Time Prediction**
This repository includes a script for real-time predictions using a webcam feed:
- Captures frames in real-time.
- Preprocesses frames (grayscale conversion, resizing, and normalization).
- Uses the trained model to predict age and gender.
- Displays predictions directly on the video.


---

### **Example Predictions**
1. **Screenshot 2024-11-20 161133.png**:  
   - **Input**: The user’s facial image.  
   - **Prediction**: Age **18**, Gender **Female** (Correct).  
   - ![Screenshot 2024-11-20 161133](https://github.com/Tannu-Rawat/age_and_gender_detection/blob/main/Screenshots/2024-11-20_161133.png)

2. **Screenshot 2024-11-21 080512.png**:  
   - **Input**: Selena Gomez’s younger photo (age ~20–25).  
   - **Prediction**: Age **24**, Gender **Female** (Accurate).  
   - ![Screenshot 2024-11-21 080512](https://github.com/Tannu-Rawat/age_and_gender_detection/blob/main/Screenshots/2024-11-21_080512.png)

3. **Screenshot 2024-11-21 081739.png**:  
   - **Input**: Abby’s facial image during training.  
   - **Prediction**: Almost exact, showcasing the model’s ability to generalize.  
   - ![Screenshot 2024-11-21 081739](https://github.com/Tannu-Rawat/age_and_gender_detection/blob/main/Screenshots/2024-11-21_081739.png)

---

### **How to Update**
Replace `yourusername` with your actual GitHub username, and ensure the file paths match your repository structure.

Let me know if further adjustments are needed! 😊
```python
# Running the live prediction script
python live_camera.py
```

---

## **How to Use**
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/age_and_gender_detection.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model (optional if using a pre-trained model):
   ```bash
   python train_model.py
   ```
4. Run the live prediction module:
   ```bash
   python live_camera.py
   ```

---

## **Requirements**
- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib

---

## **Future Improvements**
- Enhance age prediction for extreme age ranges (infants or seniors).
- Add emotion recognition to complement age and gender detection.
- Train on larger, more diverse datasets for improved generalization.

Feel free to contribute and share your feedback!


