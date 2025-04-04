# Daily_Weather_Analysis

**Project Overview**
The aim is to forecast future climate conditions based on past weather records using regression-based models.
Files in the Repository
File Name	Description
weather_prediction.py	----Main Python script for weather prediction
weather_dataset.csv	----- Historical climate data
README.md	-----Project documentation (this file)

**Dataset**
Historical data includes:

Minimum and Maximum Temperature

Humidity at 9 AM

Pressure at 9 AM

Rainfall levels

**Model & Features**

Model Used: Random Forest Regressor

**Features Used:**

Minimum Temperature

Maximum Temperature

Humidity at 9 AM

Pressure at 9 AM

Rainfall

**Process**
Load and clean the dataset

Handle missing values

Split data into training and testing sets

Train the Random Forest Regressor

Predict future weather patterns

**Requirements**

Python 3.x
Pandas
NumPy
scikit-learn
Matplotlib / Seaborn (optional for visualization)

**How to Run**
Install dependencies:

pip install pandas numpy scikit-learn matplotlib

**Run the script:**
python weather_prediction.py

**Output**

Predicted values for temperature and rainfall

Visual comparison of actual vs. predicted trends

Evaluation metric: Mean Squared Error (MSE)

The model performs well on normal weather trends

Slightly less accurate in extreme weather shifts

Can be extended with real-time weather APIs

