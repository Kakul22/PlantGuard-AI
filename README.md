# 🌿 Plant Disease Detection Web Application

A complete Machine Learning based web application for detecting plant diseases using Convolutional Neural Networks (CNN). This application can identify 38 different types of plant diseases from leaf images and provide treatment recommendations.

## 📋 Features

- **Deep Learning Model**: CNN architecture trained on PlantVillage dataset
- **38 Disease Classes**: Detects diseases across multiple plant species
- **Real-time Prediction**: Upload image and get instant results
- **Confidence Scores**: Shows prediction confidence percentage
- **Treatment Recommendations**: Provides actionable treatment advice
- **Responsive UI**: Works on desktop, tablet, and mobile devices
- **Clean Interface**: Modern, user-friendly design

## 🗂️ Project Structure

```
plant_disease_detector/
│
├── model_training.py          # CNN model training script
├── app.py                      # Flask backend application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── models/                     # Trained model storage
│   ├── plant_disease_model.h5    # Trained model (generated)
│   └── class_indices.json         # Class mapping (generated)
│
├── templates/                  # HTML templates
│   └── index.html             # Frontend interface
│
├── static/                     # Static files
│   ├── css/
│   │   └── style.css          # Stylesheet
│   └── uploads/               # Uploaded images (temporary)
│
└── PlantVillage/              # Dataset (to be downloaded)
    ├── train/                 # Training images
    └── valid/                 # Validation images
```

## 🚀 Installation & Setup

### Step 1: Clone or Download the Project

Download all the project files to your local machine.

### Step 2: Install Python

Make sure you have Python 3.8 or higher installed. Check with:

```bash
python --version
```

### Step 3: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow/Keras (Deep Learning)
- Flask (Web Framework)
- NumPy (Numerical Computing)
- Pillow (Image Processing)
- Matplotlib & Seaborn (Visualization)
- scikit-learn (ML Utilities)

### Step 5: Download Dataset

1. Download the PlantVillage dataset from Kaggle:
   - URL: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
   - Or search "PlantVillage dataset" on Kaggle

2. Extract the downloaded ZIP file

3. Place the extracted folder in the project directory with this structure:
   ```
   PlantVillage/
   ├── train/
   │   ├── Apple___Apple_scab/
   │   ├── Apple___Black_rot/
   │   ├── ...
   └── valid/
       ├── Apple___Apple_scab/
       ├── Apple___Black_rot/
       └── ...
   ```

4. Update the `DATASET_PATH` in `model_training.py` if your folder name is different

## 🎓 Training the Model

### Step 6: Train the CNN Model

```bash
python model_training.py
```

**Training Details:**
- **Duration**: 2-4 hours (depending on hardware)
- **GPU Recommended**: Training will be much faster with GPU
- **Output Files**:
  - `models/plant_disease_model.h5` - Final trained model
  - `models/best_model.h5` - Best model checkpoint
  - `models/class_indices.json` - Class name mappings
  - `training_history.png` - Training curves
  - `confusion_matrix.png` - Confusion matrix
  - `classification_report.txt` - Detailed metrics

**Training Progress:**
The script will show:
- Data loading progress
- Model architecture summary
- Training progress (epoch by epoch)
- Validation accuracy and loss
- Final evaluation metrics

**Expected Results:**
- Training Accuracy: ~95-99%
- Validation Accuracy: ~90-95%
- Precision/Recall: ~90%+

## 🌐 Running the Web Application

### Step 7: Start the Flask Server

```bash
python app.py
```

The server will start on `http://127.0.0.1:5000`

### Step 8: Access the Application

1. Open your web browser
2. Navigate to: `http://127.0.0.1:5000` or `http://localhost:5000`
3. You should see the Plant Disease Detection interface

## 📱 Using the Application

1. **Upload Image**:
   - Click on "Choose Image" button
   - Select a clear image of a plant leaf
   - Supported formats: PNG, JPG, JPEG
   - Max size: 16MB

2. **View Preview**:
   - The selected image will appear as a preview

3. **Analyze**:
   - Click "Analyze Image" button
   - Wait for processing (usually 1-2 seconds)

4. **View Results**:
   - **Detected Disease**: Name of the identified disease
   - **Confidence Level**: Prediction confidence (0-100%)
   - **Treatment Recommendation**: Suggested treatment steps

5. **Analyze Another**:
   - Click "Analyze Another Image" to start over

## 🔬 Model Architecture

The CNN model consists of:
- **Input Layer**: 224x224x3 (RGB images)
- **4 Convolutional Blocks**: Each with:
  - 2 Conv2D layers
  - Batch Normalization
  - MaxPooling
  - Dropout
- **Dense Layers**: 512 → 256 → 38 (output classes)
- **Activation**: ReLU (hidden), Softmax (output)
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy

## 🌱 Supported Plant Diseases

The model can detect 38 different conditions including:

**Apple**: Apple Scab, Black Rot, Cedar Apple Rust, Healthy

**Cherry**: Powdery Mildew, Healthy

**Corn**: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy

**Grape**: Black Rot, Esca, Leaf Blight, Healthy

**Peach**: Bacterial Spot, Healthy

**Pepper**: Bacterial Spot, Healthy

**Potato**: Early Blight, Late Blight, Healthy

**Strawberry**: Leaf Scorch, Healthy

**Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

And more...

## 🛠️ Troubleshooting

### Issue: Model file not found
**Solution**: Make sure you've run `model_training.py` first to generate the model files.

### Issue: Dataset not found
**Solution**: Download the PlantVillage dataset and place it in the correct location.

### Issue: Out of memory during training
**Solution**: 
- Reduce `BATCH_SIZE` in `model_training.py` (try 16 or 8)
- Close other applications
- Use a machine with more RAM/VRAM

### Issue: Low prediction accuracy
**Solution**:
- Ensure images are clear and well-lit
- Use images of diseased leaves (not whole plants)
- Try different angles or lighting

### Issue: Port 5000 already in use
**Solution**: 
- Change the port in `app.py`: `app.run(port=5001)`
- Or kill the process using port 5000

### Issue: TensorFlow installation fails
**Solution**:
- For Windows: Make sure you have Microsoft Visual C++ Redistributable installed
- For macOS with M1/M2: Use `tensorflow-macos` instead
- Update pip: `pip install --upgrade pip`

## 📊 Model Performance

After training, you'll see:
- **Accuracy**: Overall classification accuracy
- **Precision**: How many predictions were correct
- **Recall**: How many actual cases were found
- **Confusion Matrix**: Visual representation of predictions
- **Training Curves**: Loss and accuracy over epochs

## 🔒 Security Notes

- The application stores uploaded images temporarily in `static/uploads/`
- Consider adding authentication for production use
- Implement rate limiting for the API endpoint
- Use HTTPS in production
- Regularly update dependencies for security patches

## 🚀 Deployment (Optional)

To deploy this application:

1. **Heroku**:
   - Add `Procfile`: `web: gunicorn app:app`
   - Add `gunicorn` to requirements.txt
   - Deploy using Heroku CLI

2. **AWS/GCP/Azure**:
   - Use their respective ML deployment services
   - Consider using Docker containers

3. **Docker**:
   - Create a Dockerfile
   - Build and run container

## 📈 Future Enhancements

Potential improvements:
- Add more plant species and diseases
- Implement disease severity detection
- Add multi-language support
- Include disease progression tracking
- Add weather-based recommendations
- Implement user authentication
- Store prediction history
- Add mobile app version

## 📝 Notes

- This is an AI-based diagnostic tool
- For severe cases, consult with agricultural experts
- Model accuracy depends on image quality
- Regular model updates recommended with new data
- The application is for educational and research purposes

## 🤝 Contributing

To improve this project:
1. Train with more data
2. Experiment with different architectures
3. Add more disease classes
4. Improve UI/UX
5. Add new features

## 📄 License

This project is open-source and available for educational purposes.

## 👨‍💻 Technical Stack

- **Backend**: Python, Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Framework**: TensorFlow/Keras
- **Image Processing**: Pillow (PIL)
- **Data Science**: NumPy, scikit-learn
- **Visualization**: Matplotlib, Seaborn

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure dataset is properly structured
4. Check Python version compatibility

## ✅ Checklist

Before running the application:
- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed from requirements.txt
- [ ] PlantVillage dataset downloaded and extracted
- [ ] Model training completed successfully
- [ ] Model files exist in `models/` directory
- [ ] Flask server starts without errors

---

**Happy Plant Disease Detection! 🌿**

For questions or improvements, feel free to contribute to this project.
