# Home Price Prediction Project  

This project focuses on predicting residential home prices in Durham, NC.  

## Dataset  

The dataset is sourced from the local multiple listing service (MLS) and contains information on homes sold in Durham, NC, between January 1, 2021, and October 31, 2021.  

## How to Run the Application  

1. Clone this repository to your workspace.  
2. Update the `home` dictionary in the `test.py` file as needed.  
3. Run the application using the following command:  

   ```bash  
   python test.py  
   ```  

Alternatively, you can build and run the application using Docker with the following steps:  

1. Build the Docker image:  
   ```bash  
   docker build -t project-capstone-1 .  
   ```  

2. Run the Docker container:  
   ```bash  
   docker run -it --rm -p 9696:9696 project-capstone-1  
   ```  

---  
