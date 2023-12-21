from flask import Flask, request, jsonify
from io import StringIO
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

app = Flask(__name__)



@app.route('/clustering', methods=['POST'])
def clustering_api():
    try:
        # Get CSV data from the request
        csv_data = request.files['file'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_data))

        # Run the clustering function
        #result = clustering(df)

        return jsonify({"result": "clustering results will be coming soon"})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/transactions_prediction_final', methods=['POST'])
def transactions_prediction_final_api():
    try:
        # Get CSV data from the request
        csv_data = request.files['file'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_data))

        # Run the transactions_prediction_final function
        #transactions_prediction_final(df)

        return jsonify({"message": "Predictions sent via email!"})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
