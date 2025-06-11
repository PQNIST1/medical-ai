# 🚀 Getting Started

## 📚 Index


- [⚙️ Clone The Project](#clone-the-project)
- [🔧 Setup Virtual Environment](#-setup-virtual-environment)
- [📂 Navigate to Project Directory](#-navigate-to-project-directory)
- [📦 Install Dependencies](#-install-dependencies)
- [▶️ Run the Server](#️-run-the-server)
- [🧠 Train Face Recognition Model](#-train-face-recognition-model)

## ⚙️ Clone The Project <a name="clone-the-project"></a>
### Step 1: Clone
```bash
git clone https://github.com/PQNIST1/Medical-Ai.git
```

## 🔧 Setup Virtual Environment
### Step 2: Delete the old virtual environment (if exists)
```bash
Remove-Item -Recurse -Force .venv
```

### Step 3: Create a new virtual environment
```bash
python -m venv .venv
```

### Step 4: Activate the virtual environment
```bash
.venv\Scripts\activate
```

## 📂 Navigate to Project Directory
### Step 5: Go to the main project directory
```bash
cd Medical-Ai
```

## 📦 Install Dependencies
### Step 6: Install required Python packages
```bash
pip install -r requirements.txt
```

## ▶️ Run the Server
### Step 7: Start the Django development server
```bash
python manage.py runserver
```
### Step 7: Open the server
http://127.0.0.1:8000

## 🧠 Train Face Recognition Model
### Step 1: Navigate to the face recognition module
```bash
cd face_recognition_project
```

### Step 2: Train and save the model
```bash
python train_and_save_model.py
```
