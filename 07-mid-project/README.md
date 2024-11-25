### Problem Overview

This machine learning project aims to model air quality data and predict asthma prevalence across various regions in the USA. The hypothesis is that air quality, as indicated by the Air Quality Index (AQI), correlates with the prevalence of asthma within a given city or region.

The solution from this project can help individuals make informed decisions on where to live, especially those with asthma, by identifying regions with poorer air quality that may negatively affect respiratory health. By avoiding areas with high pollution levels, people can potentially improve their health outcomes.

### Data Used

Two datasets are combined in this project: "Air Quality" and "Asthma Prevalence."

1. **1980-2021 Yearly Air Quality Index Data from the US Environmental Protection Agency (EPA):**  
   The data is available [here](https://www.kaggle.com/threnjen/40-years-of-air-quality-index-from-the-epa-yearly). It contains yearly air quality reports from various metro areas across the United States, along with geographic details of the locations.  

   The columns in this dataset include:  
   - 'State'  
   - 'County'  
   - 'Year'  
   - 'Days with AQI'  
   - 'Good Days'  
   - 'Moderate Days'  
   - 'Unhealthy for Sensitive Groups Days'  
   - 'Unhealthy Days'  
   - 'Very Unhealthy Days'  
   - 'Hazardous Days'  
   - 'Median AQI'  
   - 'Days CO'  
   - 'Days NO2'  
   - 'Days Ozone'  
   - 'Days SO2'  
   - 'Days PM2.5'  
   - 'Days PM10'  
   - 'Latitude'  
   - 'Longitude'

2. **US County Data from the CDC (Centers for Disease Control and Prevention):**  
   Available [here](https://data.cdc.gov/500-Cities-Places/PLACES-County-Data-GIS-Friendly-Format-2020-releas/mssc-ksj7/about_data), [here](https://data.cdc.gov/500-Cities-Places/PLACES-County-Data-GIS-Friendly-Format-2021-releas/kmvs-jkvx/about_data), [here](https://data.cdc.gov/500-Cities-Places/PLACES-County-Data-GIS-Friendly-Format-2022-releas/xyst-f73f/about_data), this dataset provides information on asthma prevalence adjusted for age, as well as the population of each county.  
   Columns include:  
   - 'StateDesc'  
   - 'CountyName'  
   - 'TotalPopulation'  
   - 'CASTHMA_AdjPrev' (Asthma Prevalence, in percentage)

Note: This project uses data from the first dataset for 2018 - 2020, as it provides better results when matched with the 2018 - 2020 data from the second dataset. This suggests that changes in air quality impact asthma prevalence in the following year.

### Environment Setup and Execution

To install the required dependencies and set up the environment on Linux, execute:

```bash
# Install dependencies
pipenv install scikit-learn==1.0 numpy flask gunicorn

# Run the Flask app with gunicorn
pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
```

Alternatively, for a better user interface, you can use Jupyter Notebook:

```bash
# Start Jupyter notebook
jupyter-notebook
```

To clone the project repository from GitHub:

```bash
# Clone the repository
git clone https://github.com/adikusum4/Machine-Learning-zoomcamp/new/main/07-mid-project.git
```

Once the repository is cloned, open the Jupyter Notebook interface by visiting [http://localhost:8888/](http://localhost:8888/) in your browser.

### Containerization and Execution

This project includes a `Dockerfile` for containerization. To build and run the Docker container, follow these steps:

1. Build the Docker container:

   ```bash
   sudo docker build -t predict .
   ```

2. Run the container:

   ```bash
   sudo docker run -it --rm -p 9696:9696 predict
   ```

After running the container, the Flask app will be accessible at [http://localhost:9696](http://localhost:9696/).

Enjoy using the project at `/07-mid-project/notebook.ipynb`.
