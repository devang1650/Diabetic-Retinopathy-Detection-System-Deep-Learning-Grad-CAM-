# Diabetic-Retinopathy-Detection-System-Deep-Learning-Grad-CAM-

## 📌 Overview  
This project focuses on the automated detection and classification of **Diabetic Retinopathy (DR)** from retinal fundus images using **Deep Learning** techniques. Diabetic Retinopathy is a diabetes-related eye disease that can cause vision loss if not detected early. The goal of this system is to provide an **AI-powered tool** that assists ophthalmologists and healthcare professionals by classifying images into different severity levels and providing visual explanations.  

The project implements **state-of-the-art CNN architectures (DenseNet201/MobileNetV2)** and integrates **Grad-CAM** for explainability. The solution is deployed through a **Flask/Tkinter interface**, enabling real-time predictions and patient-friendly reports.  

---

## 🎯 Objectives  
- To build a reliable deep learning model for **classifying retinal fundus images** into 5 severity levels of Diabetic Retinopathy:  
  - 0: No DR  
  - 1: Mild  
  - 2: Moderate  
  - 3: Severe  
  - 4: Proliferative DR  
- To enhance **model interpretability** using **Grad-CAM heatmaps**.  
- To provide an easy-to-use **GUI/web interface** for doctors and patients.  
- To generate **automated reports** that summarize the diagnosis.  
---

---

## ⚙️ Implementation  

1. **Dataset**  
   - Used **APTOS/IDRiD dataset** containing thousands of labeled retinal images.  
   - Images were resized and preprocessed to improve clarity.  
   - Labels were mapped from `trainLabels.csv` into 5 severity categories.  

2. **Model Architecture**  
   - Base Model: **DenseNet201** with fine-tuning.  
   - Additional layers: GlobalAveragePooling, Dense, Dropout, Softmax classifier.  
   - Loss Function: **Categorical Crossentropy**.  
   - Optimizer: **Adam**.  

3. **Explainability**  
   - **Grad-CAM** implemented to visualize **regions of interest** in retinal images.  
   - Heatmaps help in understanding **why the model made a prediction**.  

4. **Deployment**  
   - Built a **Tkinter GUI** for desktop use.  
   - Option to upload retinal images → get prediction + Grad-CAM output.  
  

---

## 📊 Results  
- **Accuracy Achieved:** ~92% on validation set.  
- **Confusion Matrix:** Balanced detection across all DR classes.  
- **Grad-CAM Heatmaps:** Showed clear focus on retinal lesions and affected areas.  
- **User Interface:** Successfully predicts DR stage and displays explanation side-by-side with original image.  

---

## 🚀 How to Run  

1. Clone the repository:  
   ```bash
   git clone --depth 1 https://github.com/devang1650/Diabetic-Retinopathy-Detection-System-Deep-Learning-Grad-CAM-.git
   cd Model_file
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the application:
   a) For Tkinter GUI:
   ```bash
   python main2.py
4. Upload a retinal image → View prediction + Grad-CAM heatmap.
---
📌 Future Scope

1. Integrating with cloud-based healthcare systems for real-time use.
2. Expanding dataset for generalized global predictions.
3. Deploying as a mobile application for rural healthcare workers.
