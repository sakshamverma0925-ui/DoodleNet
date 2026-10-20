# 🎨 DoodleNet: Real-Time Sketch Recognition

A complete, end-to-end Machine Learning prototype that recognizes hand-drawn sketches in real time using a Convolutional Neural Network (CNN) and an interactive web canvas.

---

## 👥 Project Team & Academic Credentials

- **Institution**: SRM Institute of Science and Technology (SRMIST), Vadapalani Campus
- **Project**: College Machine Learning Project Prototype
- **Team Members**:
  - **S Adithya Ganesh**
  - **Sai Shrikar M K**
  - **Saksham Verma**

---

## 🏛️ System Architecture

```
[ Quick, Draw! Dataset ] ── (10 classes × 10K drawings)
         │
         ▼
 [ Preprocessing ] ───────> (28×28 resize, bounding box centering, normalization [0, 1])
         │
         ▼
 [ CNN Training ] ────────> (Conv2D, MaxPool, Dropout, Dense, Adam, 10 epochs)
         │
         ▼
 [ Saved Model ] ─────────> (model.h5 + classes.json) ──┐
                                                       │
                                                       ▼
 [ User Draws on Canvas ] ─> [ Streamlit App ] ───> [ Trained CNN ] ──> [ Top-3 Predictions ]
 (Interactive Web Canvas)    (Resizes to 28×28)     (Loads model.h5)      (+ Confidence %)
```

### 🎯 Supported Classes (10 Categories)
- 🍎 Apple
- 🚌 Bus
- 🐱 Cat
- 🐶 Dog
- 🐟 Fish
- 🌸 Flower
- 🏠 House
- 🔑 Key
- ✏️ Pencil
- ⭐ Star

---

## 🛠️ Project Structure

```
DoodleNet/
├── app.py                 # Streamlit web application with "Artist's Studio" UI
├── train.py               # Dataset ingestion, CNN training, evaluation & model saving
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation & execution guide
├── data/                  # Downloaded Quick, Draw! .npy bitmap files (cached)
├── model.h5               # Trained CNN model weights & architecture
├── classes.json           # Class labels metadata mapping
└── training_history.png   # Accuracy & loss curves generated during training
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup

Open **PowerShell** or **Command Prompt** in the project directory:

```bash
cd d:\college\DoodleNet
```

*(Recommended)* Create and activate a Python virtual environment:
```bash
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows Command Prompt:
.\venv\Scripts\activate.bat
```

Install required packages:
```bash
pip install -r requirements.txt
```

---

### 2. Train the CNN Model

You have two convenient options to train the model:

#### Option A: Full Training on Quick, Draw! Dataset (10 classes × 10,000 samples)
Downloads official `.npy` files from Google Cloud Storage and trains the CNN for 10 epochs:
```bash
python train.py --epochs 10 --samples-per-class 10000
```
> **Tip**: You can use `--samples-per-class 2000` for faster initial training.

#### Option B: Instant Offline / Synthetic Mode (No Big Downloads)
Ideal for instant testing, grading verification, or if internet bandwidth is restricted:
```bash
python train.py --synthetic --epochs 5 --samples-per-class 1000
```

Both options generate:
1. `model.h5` - The saved neural network.
2. `classes.json` - Target class definitions.
3. `training_history.png` - Plots showing training & validation loss and accuracy curves.

---

### 3. Launch the Interactive Sketchbook Studio

Run the Streamlit web application:

```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`.

---

## 🎨 UI & Aesthetic Highlights

- **Theme**: *Artist's Studio / Sketchbook*
  - Warm cream background (`#FDFBF7`)
  - Charcoal typography (`#2C2C2C`)
  - Burnt Sienna / Terracotta accents (`#D35400`)
- **Interactive Easel**: Adjustable brush stroke size, ink tone picker, and quick clear action.
- **Smart Preprocessing Pipeline**: Automatically finds the sketch bounding box, centers it, inverts the polarity to match Quick, Draw! standards, and normalizes it to a 28×28 grayscale tensor.
- **Explainability**: Expand the "🔍 Model View" panel in the app to see the exact 28×28 image passed into the convolutional feature maps.
