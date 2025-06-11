# 🚀 Getting Started

## 📚 Index

- [🔧 Setup Virtual Environment](#-setup-virtual-environment)
- [📂 Navigate to Project Directory](#-navigate-to-project-directory)
- [📦 Install Dependencies](#-install-dependencies)
- [▶️ Run the Server](#️-run-the-server)
- [🧠 Train Face Recognition Model](#-train-face-recognition-model)


## 🔧 Setup Virtual Environment
### Step 1: Delete the old virtual environment (if exists)
```bash
Remove-Item -Recurse -Force .venv
```

### Step 2: Create a new virtual environment
```bash
python -m venv .venv
```

### Step 3: Activate the virtual environment
```bash
.venv\Scripts\activate
```

## 📂 Navigate to Project Directory
### Step 4: Go to the main project directory
```bash
cd gr-project
```

## 📦 Install Dependencies
### Step 5: Install required Python packages
```bash
pip install -r requirements.txt
```

## ▶️ Run the Server
### Step 6: Start the Django development server
```bash
python manage.py runserver
```
## 🧠 Train Face Recognition Model
### Step 1: Navigate to the face recognition module
```bash
cd face_recognition_project
```

### Step 2: Train and save the model
```bash
python train_and_save_model.py
```
