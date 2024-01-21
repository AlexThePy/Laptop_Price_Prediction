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
