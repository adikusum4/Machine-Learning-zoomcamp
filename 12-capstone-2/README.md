# Estimation of Obesity Levels Based on Eating Habits and Physical Condition

This repository was created as part of [the DataTalks.Club's Machine Learning Zoomcamp](https://github.com/alexeygrigorev) by [Alexey Grigorev](https://github.com/alexeygrigorev).

This project was submitted as a Project Capstone 2 for the course.

---

## Introduction

Obesity is a pressing global issue, contributing significantly to numerous health problems such as heart disease, diabetes, and certain cancers. An individual's eating habits and physical condition play a crucial role in determining their overall health. Poor dietary practices combined with insufficient physical activity are key factors that often lead to obesity, which in turn can result in severe health complications, sometimes even fatal ones. Early identification and prevention of obesity are vital to improving public health outcomes.

### What is the problem?

The problem lies in effectively predicting obesity levels in individuals based on their eating habits and physical condition. With the rise in obesity rates worldwide, it has become increasingly important to develop systems that can assess obesity risk in a timely and accurate manner. However, many people are unaware of their risk until it's too late, often because they lack the necessary data or tools to monitor their health effectively.

### How does the solution work?

This project aims to address this problem by developing a machine learning model that classifies obesity levels based on data from individuals in Colombia, Peru, and Mexico. By analyzing key attributes related to eating habits and physical activity, the model can predict the likelihood of an individual being overweight or obese. The dataset used in this project comes from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition), consisting of 17 attributes across 2,111 records. Each record is classified based on the `NObesity` variable (Obesity Level), which includes the following categories:
+ Insufficient Weight  
+ Normal Weight  
+ Overweight Level I  
+ Overweight Level II  
+ Obesity Type I  
+ Obesity Type II  
+ Obesity Type III  

To address potential imbalances in the dataset, 77% of the records were synthetically generated using the Weka tool with the SMOTE filter, while the remaining 23% were collected from actual users via a web platform. This combined approach ensures that the model has sufficient data to train and make accurate predictions.

The solution involves building a predictive model using machine learning algorithms, which can help identify individuals at risk of obesity, thus enabling early intervention. By providing individuals with insights into their health based on their behaviors, this tool can play a significant role in managing and preventing obesity.

--- 

## Downloading the Dataset

The dataset can be downloaded using the following Python commands:

```bash
wget https://archive.ics.uci.edu/static/public/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip
unzip -o /content/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip
```

Alternatively, you can download it from this repository:  

```bash
wget 
https://github.com/adikusum4/Machine-Learning-zoomcamp/tree/main/12-capstone-2/estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition/estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition
```

## Dataset Analysis and Model Training

The dataset was analyzed and used to train models in a Jupyter Notebook. The analysis file is available in this [repository folder](https://github.com/adikusum4/Machine-Learning-zoomcamp/tree/main/12-capstone-2/notebook.ipynb).  

The training script for the selected model can be found [here](https://github.com/adikusum4/Machine-Learning-zoomcamp/tree/main/12-capstone-2/s/train.py).  

The preprocessing pipeline, label encoder, and final trained model were saved in a file named [obesity-levels-model_catboostl.bin](https://github.com/adikusum4/Machine-Learning-zoomcamp/blob/main/12-capstone-2/obesity-levels-model_catboost.bin).

## Running the Project Locally

### Using Flask

The Flask deployment script is [predict.py](https://github.com/adikusum4/Machine-Learning-zoomcamp/blob/main/12-capstone-2/predict.py).  

Dependencies are managed using [Pipfile](https://github.com/adikusum4/Machine-Learning-zoomcamp/blob/main/12-capstone-2/Pipfile). Install them by running:  

```bash
pipenv install
```

Activate the virtual environment using:  
```bash
pipenv shell
```

To start the Flask application, execute:  
```bash
cd scripts
python predict.py
```

You can test the API by running:  
```bash
python scripts/test.py
```

Ensure that the `url` variable in [test.py](https://github.com/adikusum4/Machine-Learning-zoomcamp/blob/main/12-capstone-2/test.py) is set to:  
```python
url = "http://localhost:9696/predict"
```

### Using Waitress as a WSGI Server

Within the virtual environment, start the application with:  
```bash
cd scripts
waitress-serve --listen=0.0.0.0:9696 predict:app
```

Test the API by running:  
```bash
python scripts/test.py
```

Remember to update the `url` variable in [test.py](https://github.com/adikusum4/Machine-Learning-zoomcamp/blob/main/12-capstone-2/test.py) to:  
```python
url = "http://localhost:9696/predict"
```

### Local Deployment with Docker

A [Dockerfile](https://github.com/adikusum4/Machine-Learning-zoomcamp/blob/main/12-capstone-2/Dockerfile) is included for containerization.  

Build the Docker image:  
```bash
docker build -t estimation-obesity-levels .   
```

Run the container:  
```bash
docker run -p 9696:9696 -it estimation-obesity-levels:latest
```

Test the Deployment:
   - Once the deployment is finished, run the following command to test your application:
     ```bash
     python test.py
     ```
Remember to update the `url` variable in [test.py](https://github.com/adikusum4/Machine-Learning-zoomcamp/blob/main/12-capstone-2/test.py) to:  
```python
url = "http://localhost:9696/predict"
```

## Running the Project Remotely

**Test my application:**
   - First, log in to GitHub, open your Codespace and run:
     ```bash
     python test_railway.py
     ```

### Remote Deployment with Docker and Railway

1. **Prepare Your Environment:**
   - First, log in to GitHub and open your Codespace.

2. **Sign up or Log in to Railway:**
   - Visit [https://railway.com/](https://railway.com/) and sign up or log in.

3. **Install Railway CLI in Your Codespace:**
   - In the terminal, run the following commands to install the Railway CLI:
     ```bash
     npm install -g railway
     npm uninstall -g railway
     npm install -g @railway/cli
     ```

4. **Login to Railway:**
   - After installing the Railway CLI, log in by running:
     ```bash
     railway login
     ```
   - This will open a browser for authentication. Follow the prompts to log in.

5. **Continue with the Deployment:**

   - **Initialize Your Project:**
     - In the terminal, run:
       ```bash
       railway init
       ```

   - **Deploy Your Application:**
     - To deploy your application, run:
       ```bash
       railway up
       ```

6. **Update Your Project URL:**
   - Once the deployment is finished, go to your project on [Railway.com](https://railway.com/).
   - Navigate to the **Settings** section of your project and click on **Generate Public Networking**.
   - Copy the generated link and update the `url` variable in the [test_railway.py](https://github.com/adikusum4/Machine-Learning-zoomcamp/blob/main/12-capstone-2/test_railway.py) file to the copied link:
     ```python
     url = "<your copied link>"
     ```

7. **Test the Deployment:**
   - After updating the URL, run the following command to test if your application is working:
     ```bash
     python test_railway.py
     ```

---
