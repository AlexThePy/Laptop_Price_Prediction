# Laptop Price Prediction


# Problem Description

The main objective is to understand the factors that contribute to the pricing of laptops. This is done by analyzing the correlation between laptop specifications and their prices and developeing a predictive model that estimates the price of a laptop based on its specifications. This project is created for learning purposes and can be used for potential buyers to gauge the market value of a laptop.

The dataset was downloaded from Kaggle: https://www.kaggle.com/datasets/muhammetvarl/laptop-price



# Exploratory Data Analysis

For the EDA, I have performed:

Statistical Summary by looking at the distribution of the numeric features, the mean, median, and outliers.

Correlation Analysis by determining how the different features relate to the price and to each other.

Visualization by creating plots to visualize distributions and relationships in the data.

Important Features Analysis by identifying which features have the most influence on laptop prices using feature importance from model or other statistical methods.

You can also check notebook.ipynb from this project.


![prices_ram](https://github.com/AlexThePy/Laptop_Price_Prediction/assets/106477870/1deff6dd-f9b7-40a6-b21c-63af934d1ae1)

![ram_price](https://github.com/AlexThePy/Laptop_Price_Prediction/assets/106477870/a21f062f-c7d3-462a-ac02-11987b84ff7e)


# How the Model Can be Used
A predictive model is built using machine learning techniques. The features like Company, Product, TypeName, Inches, ScreenResolution, CPU, RAM, Memory, GPU, Operating System (OpSys), and Weight could serve as the independent variables, while Price in euros would be the dependent variable (the one we aim to predict).

Upon using the Linear, Random Forest and XGB Regressors and performing hyperparameter tuning I chose the XGB Regressor as the best model.

# This is the prediction based on the XGB Regressor (which is the chosen model):
![prediction](https://github.com/AlexThePy/Laptop_Price_Prediction/assets/106477870/505a2d09-9025-49d6-98a9-ce434dd13f29)


# This is the prediction based on the Random Forest Regressor:
![rf_predictions](https://github.com/AlexThePy/Laptop_Price_Prediction/assets/106477870/2d961aee-d3d6-47f5-8195-9a022c6dec12)

# And this is the prediction based on the Linear Regression:
![lr_predictions](https://github.com/AlexThePy/Laptop_Price_Prediction/assets/106477870/1959dcc2-f34c-4c53-abd8-373b7fbf7f83)

# The MSE and R2 for the models are the following:

Linear Regression: (103513.41657271476, 0.7962039556407352)

Random Forest Regressor:(101900.37499209472, 0.7993796936696936)

XGB Regressor: (98125.51348189873,0.806811598337255) -> this model was chosen for its' best results

# INSTRUCTIONS:

# I. Download the whole project from my github and unzip it.


# II. Setting Up the Virtual Environment:

1.Open a terminal or command prompt and navigate to the directory where you have downloaded the project and run:

```
cd path/to/your/project
```

2.Create a Virtual Environment

On Windows:

```
python -m venv venv
```

On macOS/Linux:

```
python3 -m venv venv
```

This command creates a new virtual environment named venv within your project directory. You can name it differently if you prefer.


3. Activating the Virtual Environment:

Windows:

```
.\venv\Scripts\activate
```

macOS/Linux:

```
source venv/bin/activate
```

Your command prompt should now indicate that you're working inside the virtual environment.

Step 3: Installing Dependencies

With the virtual environment activated, you can now install the packages listed in your requirements.txt file:

```
pip install -r requirements.txt
```

This command reads the requirements.txt file and installs all the packages listed there, along with their specified versions.


# Building the docker image:

1. Open the cmd prompt and navigate to the directory with the dockerfile:

```
cd path\to\folder
```

2.Run the docker build command:

```
docker build -t dockerfile .
```

# Instructions for Using the Model:
1.Run the Training Script: 
```
python train.py 
```
to train and save the model.

2.Run the Prediction Service:
```
python predict.py
```
to start the Flask server.

3. Access http://127.0.0.1:6969/

4. Check the laptop_price.csv dataset to predict the price of any laptop there!

5. Thank you for checking out my project!
