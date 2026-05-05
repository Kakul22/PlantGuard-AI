
# 🌿 Plant Disease Detection System

> AI-powered web app to detect plant leaf diseases using Deep Learning (CNN)

---

## 🚀 Features
- 📸 Upload or capture leaf images
- 🤖 CNN-based disease prediction
- 📊 Confidence score + probabilities
- 💡 Treatment recommendations
- 🔍 Grad-CAM visualization (model explainability)

---

## 🧠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-CNN-red?style=for-the-badge&logo=keras)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b?style=for-the-badge&logo=streamlit)


---


---

## 🧪 Model Details
- 📂 Dataset: 20,000+ images (15 classes)
- 🧠 Model: Convolutional Neural Network (CNN)
- ⚙️ Optimizer: RMSProp
- ⏱️ Early Stopping used
- 📈 Accuracy & Loss graphs monitored

---

## 🔍 Grad-CAM
- Highlights infected regions in leaf images
- Improves model interpretability

---

## 📊 Output
- Disease Class (Healthy / Early Blight / Late Blight)
- Confidence Score
- Class Probabilities
- Treatment Suggestions

---

## ▶️ How to Run

```bash
# install dependencies
pip install -r requirements.txt

# run flask app
python app.py
```

## 📌 Future Scope
Mobile app integration 📱
Multi-crop disease detection 🌾
Real-time farm monitoring 🚜

## 🧠 Model Architecture (CNN)

Input: 224 × 224 × 3 (RGB Image)

→ Conv2D (32 filters, 3×3) + ReLU  
→ MaxPooling (2×2)

→ Conv2D (64 filters, 3×3) + ReLU  
→ MaxPooling (2×2)

→ Conv2D (128 filters, 3×3) + ReLU  
→ MaxPooling (2×2)

→ Flatten

→ Dense (128 units) + ReLU  
→ Dropout (0.5)

→ Dense (3 units) + Softmax

Output:  
- Healthy  
- Early Blight  
- Late Blight
