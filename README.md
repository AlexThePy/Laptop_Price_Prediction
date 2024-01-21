# Laptop_Price_Prediction


# Problem Description

The main objective is to understand the factors that contribute to the pricing of laptops. This could be done by analyzing the correlation between laptop specifications and their prices. Additionally, one could aim to develop a predictive model that estimates the price of a laptop based on its specifications. This model could be useful for potential buyers to gauge the market value of a laptop and for sellers or manufacturers to price their products competitively.

# How the Model Can be Used
A predictive model is built using machine learning techniques. The features like Company, Product, TypeName, Inches, ScreenResolution, CPU, RAM, Memory, GPU, Operating System (OpSys), and Weight could serve as the independent variables, while Price in euros would be the dependent variable (the one we aim to predict).

Upon testing the Linear, Random Forest and XGB Regressors and performing hyperparameter tuning, the best model used is the XGB Regressor.

# INSTRUCTIONS:

Download train.py script and run it in your Python environment to ensure that it works correctly. You can run the script from the command line:

python train_model.py



# Setting Up the Virtual Environment:

1.Open a terminal or command prompt and navigate to the directory where your project is located.

cd path/to/your/project

2.Create a Virtual Environment

On Windows:
python -m venv venv

On macOS/Linux:
python3 -m venv venv

This command creates a new virtual environment named venv within your project directory. You can name it differently if you prefer.


Step 2: Activating the Virtual Environment
Before installing any packages, you need to activate the virtual environment:



Windows:
.\venv\Scripts\activate

macOS/Linux:
source venv/bin/activate

Your command prompt should now indicate that you're working inside the virtual environment. It’s common to see the name of the virtual environment (e.g., (venv)) prefixed to your command prompt.

Step 3: Installing Dependencies

With the virtual environment activated, you can now install the packages listed in your requirements.txt file:

pip install -r requirements.txt

This command reads the requirements.txt file and installs all the packages listed there, along with their specified versions.


Building the docker image:

1. Open the cmd prompt and navigate to the directory with the dockerfile:
cd "path\to\folder"

2.Run the docker build command:
docker build -t dockerfile .


Instructions for Use:
1.Run the Training Script: Execute python train.py to train and save the model.

2.Run the Prediction Service: Execute python predict.py to start the Flask server.
