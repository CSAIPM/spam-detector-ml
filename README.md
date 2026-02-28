# 📩 SMS / Email Spam Detector (FastAPI + ML)



This project is a machine learning API that classifies messages as Spam or Not Spam.



## 🚀 Features

\- Trained ML model using scikit-learn

\- REST API built with FastAPI

\- JSON input/output

\- Ready for cloud deployment


##Project Structure
spam-detector-ml/
├── src/
│   ├── train.py          
│   ├── predict.py        
│   └── api.py            
├── model.joblib          
├── requirements.txt     
├── start.sh              
├── README.md            
└── .gitignore



## 🧠 Tech Stack

\- Python

\- scikit-learn

\- FastAPI

\- Uvicorn



## 📦 How to Run Locally



```bash

python -m venv .venv

source .venv/Scripts/activate   # Windows Git Bash

pip install -r requirements.txt

uvicorn src.api:app --reload

