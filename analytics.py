import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from statsmodels.tsa.arima.model import ARIMA

import streamlit as st
from io import BytesIO
import base64
from cassandra.cluster import Cluster
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
from shapely.geometry import Point
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
from sklearn.decomposition import PCA
from keras import layers, models
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def clustering(df):
    # Convert Pandas DataFrame to PySpark DataFrame
    spark = SparkSession.builder.appName("ECommerceClustering").getOrCreate()
    spark_df = spark.createDataFrame(df)

    # Feature engineering
    assembler = VectorAssembler(inputCols=['Recency_Flag', 'Freq_Flag', 'Monetary_Flag'], outputCol='features')
    kmeans = KMeans(k=5, featuresCol='features', predictionCol='Cluster')
    pipeline = Pipeline(stages=[assembler, kmeans])
    
    # Fit the model
    model = pipeline.fit(spark_df)
    
    # Predict clusters
    clustered_df = model.transform(spark_df)

    # Visualize the clusters
    st.subheader('Clusters Visualization')
    clustered_pandas_df = clustered_df.toPandas()
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(clustered_pandas_df['Recency_Flag'], clustered_pandas_df['Freq_Flag'], c=clustered_pandas_df['Cluster'], cmap='plasma')
    ax.set_xlabel('Recency_Flag')
    ax.set_ylabel('Freq_Flag')
    ax.set_title('Clusters Visualization')

    # Add colorbar to represent cluster labels
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Cluster Labels')

    # Save the plot to a file
    plt.savefig("cluster_chart.png")

    # Show the plot in the Streamlit app
    st.pyplot(fig)
    
    # Close Spark session
    spark.stop()

def transactions_prediction_final(df):
    # ... (remains the same up to creating the PySpark DataFrame)

    # Convert Pandas DataFrame to PySpark DataFrame
    spark = SparkSession.builder.appName("ECommerceForecasting").getOrCreate()
    spark_df = spark.createDataFrame(df)

    # Group transactions by store_nbr and date, summing the transactions
    store_transactions = (
        spark_df.groupBy("store_nbr", F.window("date", "1 day"))
        .agg(F.sum("transactions").alias("transactions"))
        .select("store_nbr", "window.start", "transactions")
        .withColumnRenamed("start", "date")
        .toPandas()
    )

    # Perform ARIMA modeling for each store_nbr
    for store_nbr in spark_df.select("store_nbr").distinct().rdd.flatMap(lambda x: x).collect():
        store_data = store_transactions[store_transactions["store_nbr"] == store_nbr].set_index("date")
        
        # Fit ARIMA model
        model = ARIMA(store_data["transactions"], order=(1, 1, 1))
        fit_model = model.fit()

        # Make predictions for the next day
        forecast = fit_model.get_forecast(steps=1)
        predicted_value = forecast.predicted_mean.values[0]

        print(f"Forecast for Store {store_nbr}: {predicted_value}")

    # Stop the Spark session
    spark.stop()

    # ... (rest of the function remains the same)

if __name__ == "__main__":
    clustering("")
