# Sales Prediction with LightGBM

This project predicts weekly sales for Walmart stores using historical sales, store information, and economic indicators. The model is built with LightGBM, and the code is structured for reproducibility and deployment using Docker.

##Project Structure
```
Sales-Prediction/
├── data/                  # Place CSV data files here (not included in repo)
├── src/
│   ├── data_preprocessing.py  # Data loading and preprocessing
│   └── train.py              # Model training and validation
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```
## Getting Started
Requirements
```
Docker
Git
```

## Run the project
1. Clone the repository:
```
git clone https://github.com/HamoAsatryan/Sales-Forecast.git
cd Sales-Forecast
```
2. Build the Docker image:
```
docker build -t walmart-forecast:latest .
```
3. Run the training inside the container:
```
docker run --rm walmart-forecast:latest
```
The training logs will display validation metrics: RMSE, MAE, and R².

## Dependencies
```
pandas==2.3.2
numpy==2.1.2
scikit-learn==1.7.1
lightgbm==4.6.0
```



