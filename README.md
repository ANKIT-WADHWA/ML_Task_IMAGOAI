📊 ImagoAI Assignment
This repository contains the implementation of spectral data analysis and regression modeling for the ImagoAI Assignment. It includes data preprocessing, dimensionality reduction (PCA & t-SNE), model training, evaluation, and visualization.

---------------------------------



Files are present in the SRC folder
files such as 
-train_model
-preprocess_model



-------------------------------------------

ML_Task_ImagoAI/
│── main.py                # Main script for training & evaluation
│── requirements.txt       # List of dependencies
│── data/                  # Folder for datasets
│── models/                # Trained models
│── reports/               # Analysis & results
│── README.md              # Documentation (this file)
│── .gitignore             # Ignored files
│── notebooks/             # Jupyter notebooks for EDA and analysis
│── src/                   # Source code for preprocessing, training, etc.
│   │── preprocess.py      # Data preprocessing script
│   │── train_model.py     # Model training script
│   │── __init__.py        # Initialization file
│   │── __pycache__/       # Python cache files
│── catboost_info/         # CatBoost training logs and info
│── ml_env/                # Virtual environment for the project





Instructions to Install Dependencies and Run the Code


1. Clone the Repository:
    git clone <repository-url>
    cd ML_Task_ImagoAI



2. Set Up the Virtual Environment:
    Create a virtual environment:

        python -m venv ml_env




Activate the virtual environment:   
       ml_env\Scripts\activate




3. Install Dependencies:
        pip install -r requirements.txt


4. Execute the main script:
    python main.py
