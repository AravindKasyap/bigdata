import streamlit as st
from io import BytesIO
import base64
# from cassandra.cluster import Cluster
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
from shapely.geometry import Point
# from pyspark.sql import SparkSession
# from pyspark.sql import functions as F
# from pyspark.ml.feature import VectorAssembler
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


def send_email(results_df):
    # Email credentials
    sender_email = "aravindkasyap317@gmail.com"
    sender_password = "uvoa gphx jxrj hczv"
    receiver_email = "aravindkasyap317@gmail.com"  # Replace with the recipient's email

    # Email content
    subject = "Predicted Demand details for each store"
    body = "Please maintain the inventory at each store based on the prediction results attached to this email."

    # Create the MIME object
    msg = MIMEMultipart()
    msg.attach(MIMEText(body, 'plain'))

    # Attachment (CSV file)
    attachment = MIMEText(results_df.to_csv(index=False), 'csv')
    attachment.add_header('Content-Disposition', 'attachment', filename='predicted_results.csv')
    msg.attach(attachment)

    # Establish a connection to the SMTP server
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())



def clustering(df):
     # loading the dataset:
        # df = pd.read_csv("C://Kasyap//projects//bigdata//data//E-commerce_Dataset_STP.csv")

        # print(df.head(10))
        # looking at the shape of the dataset
        df.shape

        df["Total_Price"] = df["UnitPrice"] * df["Quantity"]
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

        # Group by 'InvoiceDate' and 'InvoiceNo', then sum the 'Total_Price'
        grouped_df = (
            df.groupby(["InvoiceDate", "InvoiceNo"])
            .agg({"Total_Price": "sum"})
            .reset_index()
        )

        # Create a new column 'date' based on the year and month
        df["date"] = (
            grouped_df["InvoiceDate"].dt.year * 100 + grouped_df["InvoiceDate"].dt.month
        )

        # Display the result
        print(grouped_df)

        # checking country-wise sales
        Cust_country = df[["Country", "CustomerID"]].drop_duplicates()

        # Calculating the distinct count of customer for each country
        Cust_country_count = (
            Cust_country.groupby(["Country"])["CustomerID"]
            .aggregate("count")
            .reset_index()
            .sort_values("CustomerID", ascending=False)
        )

        # Plotting the count of customers
        country = list(Cust_country_count["Country"])
        Cust_id = list(Cust_country_count["CustomerID"])
        # plt.figure(figsize=(12,8))
        # sns.barplot(country, Cust_id, alpha=0.8, color=color[2])
        # plt.xticks(rotation='60')
        # plt.show()

        Cust_date_UK = df[df["Country"] == "United Kingdom"]
        Cust_date_UK = Cust_date_UK[["CustomerID", "date"]].drop_duplicates()

        def recency(row):
            if row["date"] > 201110:
                val = 5
            elif row["date"] <= 201110 and row["date"] > 201108:
                val = 4
            elif row["date"] <= 201108 and row["date"] > 201106:
                val = 3
            elif row["date"] <= 201106 and row["date"] > 201104:
                val = 2
            else:
                val = 1
            return val

        Cust_date_UK["Recency_Flag"] = Cust_date_UK.apply(recency, axis=1)
        Cust_date_UK.head()

        tst = Cust_date_UK.groupby("Recency_Flag")
        tst.size()

        Cust_freq = df[["Country", "InvoiceNo", "CustomerID"]].drop_duplicates()
        Cust_freq.head()

        # Calculating the count of unique purchase for each customer and his buying freq in descending order
        Cust_freq_count = (
            Cust_freq.groupby(["Country", "CustomerID"])["InvoiceNo"]
            .aggregate("count")
            .reset_index()
            .sort_values("InvoiceNo", ascending=False)
        )

        Cust_freq_count_UK = Cust_freq_count[
            Cust_freq_count["Country"] == "United Kingdom"
        ]
        Cust_freq_count_UK.head()
        unique_invoice = Cust_freq_count_UK[["InvoiceNo"]].drop_duplicates()

        # Dividing in 5 equal parts
        unique_invoice["Freqency_Band"] = pd.qcut(unique_invoice["InvoiceNo"], 5)
        unique_invoice = unique_invoice[["Freqency_Band"]].drop_duplicates()
        unique_invoice

        def frequency(row):
            if row["InvoiceNo"] <= 13:
                val = 1
            elif row["InvoiceNo"] > 13 and row["InvoiceNo"] <= 25:
                val = 2
            elif row["InvoiceNo"] > 25 and row["InvoiceNo"] <= 38:
                val = 3
            elif row["InvoiceNo"] > 38 and row["InvoiceNo"] <= 55:
                val = 4
            else:
                val = 5
            return val

        Cust_freq_count_UK["Freq_Flag"] = Cust_freq_count_UK.apply(frequency, axis=1)

        # Let us check the distribution of Frequency flags:
        Cust_freq_count_UK.groupby(["Freq_Flag"]).size()

        
        # Calculating the Sum of total monetary purchase for each customer

        Cust_monetary = (
            df.groupby(["Country", "CustomerID"])["Total_Price"]
            .aggregate("sum")
            .reset_index()
            .sort_values("Total_Price", ascending=False)
        )
        Cust_monetary_UK = Cust_monetary[Cust_monetary["Country"] == "United Kingdom"]

        unique_price = Cust_monetary_UK[["Total_Price"]].drop_duplicates()
        unique_price = unique_price[unique_price["Total_Price"] > 0]
        unique_price["monetary_Band"] = pd.qcut(unique_price["Total_Price"], 5)
        unique_price = unique_price[["monetary_Band"]].drop_duplicates()
        unique_price

        def monetary(row):
            if row["Total_Price"] <= 243:
                val = 1
            elif row["Total_Price"] > 243 and row["Total_Price"] <= 463:
                val = 2
            elif row["Total_Price"] > 463 and row["Total_Price"] <= 892:
                val = 3
            elif row["Total_Price"] > 892 and row["Total_Price"] <= 1932:
                val = 4
            else:
                val = 5
            return val

        Cust_monetary_UK["Monetary_Flag"] = Cust_monetary_UK.apply(monetary, axis=1)

        # Let us check the distribution of Monetary flags:
        Cust_monetary_UK.groupby(["Monetary_Flag"]).size()


        # Combining all the three flags :
        Cust_UK_All = pd.merge(
            Cust_date_UK,
            Cust_freq_count_UK[["CustomerID", "Freq_Flag"]],
            on=["CustomerID"],
            how="left",
        )
        Cust_UK_All = pd.merge(
            Cust_UK_All,
            Cust_monetary_UK[["CustomerID", "Monetary_Flag"]],
            on=["CustomerID"],
            how="left",
        )

        # Cust_UK_All.head(10)
        sorted_data = Cust_UK_All.sort_values(
            by=["Recency_Flag", "Freq_Flag", "Monetary_Flag"],
            ascending=[False, False, False],
        )
        output_path = "RFM_resluts.csv"
        sorted_data.to_csv(output_path, index=False)
        results_df = pd.read_csv(output_path)
        df = pd.read_csv(output_path)

        # Step 2: Prepare the data
        X = df[['Recency_Flag', 'Freq_Flag', 'Monetary_Flag']]
        
        # Step 3: Split the data into train and test sets
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

        # Step 4: Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Step 5: Define a simple autoencoder model
        input_dim = X_train_scaled.shape[1]
        encoding_dim = 5  # Number of clusters
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)
        decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
        autoencoder = models.Model(inputs=input_layer, outputs=decoded)

        # Compile the model
        autoencoder.compile(optimizer='adam', loss='mse')

        # Train the autoencoder
        autoencoder.fit(X_train_scaled, X_train_scaled, epochs=20, batch_size=32, shuffle=True, validation_data=(X_test_scaled, X_test_scaled))

        # Extract the encoder part of the autoencoder to get the cluster assignments
        encoder = models.Model(inputs=input_layer, outputs=encoded)
        cluster_labels = encoder.predict(X_test_scaled).argmax(axis=1)

        # Step 6: Test the model and evaluate silhouette score
        silhouette_avg = silhouette_score(X_test_scaled, cluster_labels)
        #print(f"Silhouette Score: {silhouette_avg}")
        
        # Save silhouette score to a separate text file
        # with open("result_files\silhouette_score.txt", "w") as score_file:
        #     score_file.write(str(silhouette_avg))

        # Step 7: Visualize the clusters using PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test_scaled)
        
    

        # Step 8: Save the cluster results as a copy of the input file
        df_test = df.loc[X_test.index].copy()
        df_test['Cluster'] = cluster_labels
        df_test.to_csv('cluster_results.csv', index=False)

    # Step 9: Visualize the clusters using the original features
        st.subheader('Clusters Visualization')
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot Recency and Frequency with respective clusters
        scatter = ax.scatter(df_test['Recency_Flag'], df_test['Freq_Flag'], c=df_test['Cluster'], cmap='plasma')
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
        return ""
        

def new_clustering(df):
    # Connect to Cassandra
    cluster = Cluster(['localhost'])
    session = cluster.connect()

    # Keyspace and table information
    keyspace = "supplychain"
    table_name = "live_data_full"

    # Use the keyspace
    session.set_keyspace(keyspace)

    # Execute a CQL query to fetch data from the table
    query = f"SELECT * FROM {table_name};"
    result = session.execute(query)

    # Convert Cassandra rows to a list of dictionaries
    data = [{"store_nbr": row.store_nbr, "transactions": row.transactions, "city": row.city, "state": row.state} for row in result]

    # Create a Pandas DataFrame
    df_cluster = pd.DataFrame(data)

    # Consolidate transactions for each store
    total_transactions_df = df_cluster.groupby("store_nbr").agg({
        "transactions": "sum",
        "city": "first",  # Assuming city remains the same for each store
        "state": "first"  # Assuming state remains the same for each store
    }).reset_index()

    # Visualize the results
    st.subheader('K-Means Clustering of Total Transactions')
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(total_transactions_df["transactions"], total_transactions_df["cluster"])
    ax.set_xlabel("Total Transactions")
    ax.set_ylabel("City")
    ax.set_title("K-Means Clustering of Total Transactions")
    st.pyplot(fig)

    # Shutdown Cassandra cluster
    cluster.shutdown()
    return ""

def exploratory_items(df):
    # Group by 'family' and get counts, then select top 10
    top_families = df['family'].value_counts().head(10)

    # Bar plot for Top 10 Family distribution with counts
    st.subheader('Top 10 Families - Item Distribution')
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(x=top_families.index, y=top_families.values, ax=ax1)
    plt.title('Top 10 Families - Item Distribution')
    plt.xlabel('Family')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    st.pyplot(fig1)

    # Pie chart for Proportion of Perishable Items
    st.subheader('Proportion of Perishable Items')
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    perishable_counts = df['perishable'].value_counts()
    perishable_labels = ['Non Perishable', 'Perishable']
    perishable_counts.plot.pie(autopct='%1.1f%%', startangle=90, explode=(0, 0.1), colors=['lightblue', 'lightcoral'], ax=ax2)
    plt.title('Proportion of Perishable Items')
    plt.legend(labels=perishable_labels, loc="upper right")
    st.pyplot(fig2)

    st.write("")  # Add a space in the UI
    return ""


def exploratory_transactions(df):
    # Create a dropdown widget for selecting store numbers
    selected_store = st.sidebar.selectbox('Select Store Number:', ['All'] + list(df['store_nbr'].unique()), index=0)

    # Filter the DataFrame based on the selected store
    if selected_store == 'All':
        filtered_df = df.copy()
    else:
        filtered_df = df[df['store_nbr'] == selected_store]

    # Group by 'store_nbr' and sum transactions
    total_transactions = filtered_df.groupby('store_nbr')['transactions'].sum().reset_index()

    # Bar plot for Total Transactions per Store Number
    st.subheader('Total Transactions per Store Number')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(total_transactions['store_nbr'], total_transactions['transactions'], color='skyblue')
    ax.set_title(f'Total Transactions for Store {selected_store}')
    ax.set_xlabel('Store Number')
    ax.set_ylabel('Total Transactions')
    
    # Format y-axis to display exact numbers
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    st.pyplot(fig)

    st.write("")  # Add a space in the UI
    return ""

# def transactions_prediction():
#     #df = pd.read_csv("C://Kasyap//projects//bigdata//data//transactions.csv")
#     # Create a Spark session
#     spark = SparkSession.builder.appName("WeeklyForecast").getOrCreate()

#     # Read data from CSV file
#     csv_path = "C://Kasyap//projects//bigdata//data//transactions.csv"
#     df = spark.read.csv(csv_path, header=True, inferSchema=True)

#     # Convert the date column to a TimestampType
#     df = df.withColumn("date", F.to_timestamp("date", "dd-MM-yyyy"))

#     # Group transactions by store_nbr and week, summing the transactions
#     weekly_transactions = (
#         df.groupBy("store_nbr", F.window("date", "1 week"))
#         .agg(F.sum("transactions").alias("transactions"))
#         .select("store_nbr", "window.start", "transactions")
#         .withColumnRenamed("start", "date")
#         .toPandas()
#     )

#     # Perform ARIMA modeling for each store_nbr
#     for store_nbr in df.select("store_nbr").distinct().rdd.flatMap(lambda x: x).collect():
#         store_data = weekly_transactions[weekly_transactions["store_nbr"] == store_nbr].set_index("date")
        
#         # Fit ARIMA model
#         model = ARIMA(store_data["transactions"], order=(1, 1, 1))
#         fit_model = model.fit()

#         # Make predictions for the next week
#         forecast = fit_model.get_forecast(steps=1)
#         predicted_value = forecast.predicted_mean.values[0]

#         print(f"Forecast for Store {store_nbr}: {predicted_value}")

#     # Stop the Spark session
#     spark.stop()

def transactions_prediction_final(df):
    # Group transactions by store_nbr and date, summing the transactions
    store_transactions = (
        df.groupby(["store_nbr", pd.Grouper(key='date', freq='D')])
        .agg(transactions=('transactions', 'sum'))
        .reset_index()
    )

    # Initialize a dictionary to store last day's predictions for each store
    last_day_predictions = {}

    # Perform ARIMA modeling for each store_nbr
    for store_nbr in df['store_nbr'].unique():
        store_data = store_transactions[store_transactions["store_nbr"] == store_nbr].set_index("date")

        # Ensure a complete date index with a daily frequency
        complete_date_index = pd.date_range(start=store_data.index.min(), end=store_data.index.max(), freq='D')
        store_data = store_data.reindex(complete_date_index)

        # Fit ARIMA model
        model = ARIMA(store_data["transactions"], order=(1, 1, 1))
        fit_model = model.fit()

        # Make predictions for the next year
        forecast = fit_model.get_forecast(steps=365)
        predicted_values = forecast.predicted_mean

        # Store the last day's prediction for each store
        last_day_predictions[store_nbr] = predicted_values.iloc[-1]

        # Print or use the predicted_values as needed
        print(f"Predicted sales for Store {store_nbr} on the last day:\n{predicted_values.iloc[-1]}\n")

    # Calculate and print the average prediction for each store at the end
    average_predictions = pd.Series(last_day_predictions).mean()
    print(f"\nAverage Predicted sales for Each Store on the Last Day:\n{average_predictions}")

     # Add a new column for the number of units required
    predictions_df = pd.DataFrame(last_day_predictions.items(), columns=['Store Number', 'Predicted Transactions'])
    predictions_df['Units Required'] = (predictions_df['Predicted Transactions'] / 75).round()

    # Display predictions in Streamlit UI
    st.subheader('Demand for Each Store')
    st.dataframe(predictions_df)

    st.subheader('Average Predicted demand for Each Store on the Last Day')
    st.write(average_predictions)
    # Email credentials
    sender_email = "aravindkasyap317@gmail.com"
    sender_password = "uvoa gphx jxrj hczv"
    receiver_email = "aravindkasyap317@gmail.com"  # Replace with the recipient's email

    # Email content
    subject = "Predicted Demand details for each store"
    body = "Please maintain the inventory at each store based on the prediction results attached to this email."

    # Create the MIME object
    msg = MIMEMultipart()
    msg.attach(MIMEText(body, 'plain'))

    # Attachment (CSV file)
    attachment = MIMEText(predictions_df.to_csv(index=False), 'csv')
    attachment.add_header('Content-Disposition', 'attachment', filename='predicted_results.csv')
    msg.attach(attachment)

    # Establish a connection to the SMTP server
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
    # Button to send email
    
    st.success('Forecasts sent via email!')

# transactions_prediction_final()
# exploratory_transactions()
# exploratory_items()
# clustering()
# new_clustering()
# Function to create a downloadable link for a DataFrame
def download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # B64 encoding
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href

# Function for Exploratory Analysis
def exploratory_analysis():
     st.subheader("Exploratory Analysis")
     # Download template links
    #  st.subheader("please Download Templates for reference:")
     transactions_template = pd.DataFrame(columns=['date', 'store_nbr', 'transactions'])
     items_template = pd.DataFrame(columns=['item_nbr', 'family', 'class', 'perishable'])

     st.markdown(download_link(transactions_template, "transactions_template", "Download Transactions Template"), unsafe_allow_html=True)
     st.markdown(download_link(items_template, "items_template", "Download Items Template"), unsafe_allow_html=True)

    # Upload file for exploratory_transactions
     transactions_file = st.file_uploader("Upload CSV file for transactions analysis", type=["csv"])

        # Upload file for exploratory_items
     items_file = st.file_uploader("Upload CSV file for items analysis", type=["csv"])

     if transactions_file and items_file:
            # Save the uploaded files
            transactions_data = pd.read_csv(transactions_file, parse_dates=['date'], dayfirst=True)
            items_data = pd.read_csv(items_file)

            # Run exploratory functions
            exploratory_transactions(transactions_data)
            exploratory_items(items_data)

            

# Function for Clustering
def clustering_analysis():
    st.header("Clustering Analysis")
    
    # Upload file for clustering analysis
    uploaded_file = st.file_uploader("Upload your CSV file for clustering analysis", type=["csv"])

    if uploaded_file:
        # Save the uploaded file
        clustering_data = pd.read_csv(uploaded_file)

        # Run clustering function
        # new_clustering(clustering_data)
        clustering(clustering_data)

# Function for Transactions Prediction
def transactions_prediction_analysis():
    st.header("Transactions Prediction Analysis")
    
    # Upload file for transactions prediction analysis
    uploaded_file = st.file_uploader("Upload your CSV file for transactions prediction analysis", type=["csv"])

    if uploaded_file:
        # Save the uploaded file
        transactions_data = pd.read_csv(uploaded_file, parse_dates=['date'], dayfirst=True)

        # Run transactions prediction function
        transactions_prediction_final(transactions_data)

# Main Streamlit app
def main():
    st.markdown("<h1 style='text-align: center;'>Big Data Analytics project</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Supply chain Analytics fro Inventory and retail management</h3>", unsafe_allow_html=True)
    # st.title("Big Data Analytics project")
    # st.subheader(" ")

    # Sidebar with navigation
    menu = ["Exploratory Analysis", "Clustering Analysis", "Transactions Prediction Analysis"]
    choice = st.sidebar.radio("Select Analysis", menu)

    if choice == "Exploratory Analysis":
        exploratory_analysis()

    elif choice == "Clustering Analysis":
        clustering_analysis()

    elif choice == "Transactions Prediction Analysis":
        transactions_prediction_analysis()

if __name__ == "__main__":
    main()


