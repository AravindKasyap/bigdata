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


def clustering():
     # loading the dataset:
        df = pd.read_csv("C://Kasyap//projects//bigdata//data//E-commerce_Dataset_STP.csv")

        print(df.head(10))
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

def new_clustering():
    print("started")
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
    print(result)
    # Convert Cassandra rows to a list of dictionaries
    data = [{"store_nbr": row.store_nbr, "transactions": row.transactions, "city": row.city, "state": row.state} for row in result]

    # Create a Pandas DataFrame
    #df = pd.DataFrame(data)
    df_cluster = pd.read_csv("C://Kasyap//projects//bigdata//data//consolidated_data.csv")
    #print(df)
    # Consolidate transactions for each store
    total_transactions_df = df_cluster.groupby("store_nbr").agg({
        "transactions": "sum",
        "city": "first",  # Assuming city remains the same for each store
        "state": "first"  # Assuming state remains the same for each store
    }).reset_index()
    plt.scatter(total_transactions_df["state"], total_transactions_df["transactions"])
    plt.xlabel("Total Transactions")
    plt.ylabel("city")
    plt.title("K-Means Clustering of Total Transactions")
    plt.show()
    cluster.shutdown()
#print(total_transactions_df)

def exploratory_items():
    df = pd.read_csv("C://Kasyap//projects//bigdata//data//items.csv")
    # Group by 'family' and get counts, then select top 10
    top_families = df['family'].value_counts().head(10)

    # Bar plot for Top 10 Family distribution with counts
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_families.index, y=top_families.values)
    plt.title('Top 10 Families - Item Distribution')
    plt.xlabel('Family')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
    plt.show()
        
    #3
    plt.figure(figsize=(8, 8))
    perishable_counts = df['perishable'].value_counts()
    perishable_labels = ['Non Perishable', 'Perishable']
    perishable_counts.plot.pie(autopct='%1.1f%%', startangle=90, explode=(0, 0.1), colors=['lightblue', 'lightcoral'])
    plt.title('Proportion of Perishable Items')
    plt.legend(labels=perishable_labels, loc="upper right")
    plt.show()
    return ""

def exploratory_transactions():
    df = pd.read_csv("C://Kasyap//projects//bigdata//data//transactions.csv")
    # Group by 'store_nbr' and sum transactions
    total_transactions = df.groupby('store_nbr')['transactions'].sum().reset_index()

    # Bar plot for Total Transactions per Store Number
    plt.figure(figsize=(12, 6))
    plt.bar(total_transactions['store_nbr'], total_transactions['transactions'], color='skyblue')
    plt.title('Total Transactions per Store Number')
    plt.xlabel('Store Number')
    plt.ylabel('Total Transactions')
    plt.show()


def transactions_prediction():
    #df = pd.read_csv("C://Kasyap//projects//bigdata//data//transactions.csv")
    # Create a Spark session
    spark = SparkSession.builder.appName("WeeklyForecast").getOrCreate()

    # Read data from CSV file
    csv_path = "C://Kasyap//projects//bigdata//data//transactions.csv"
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    # Convert the date column to a TimestampType
    df = df.withColumn("date", F.to_timestamp("date", "dd-MM-yyyy"))

    # Group transactions by store_nbr and week, summing the transactions
    weekly_transactions = (
        df.groupBy("store_nbr", F.window("date", "1 week"))
        .agg(F.sum("transactions").alias("transactions"))
        .select("store_nbr", "window.start", "transactions")
        .withColumnRenamed("start", "date")
        .toPandas()
    )

    # Perform ARIMA modeling for each store_nbr
    for store_nbr in df.select("store_nbr").distinct().rdd.flatMap(lambda x: x).collect():
        store_data = weekly_transactions[weekly_transactions["store_nbr"] == store_nbr].set_index("date")
        
        # Fit ARIMA model
        model = ARIMA(store_data["transactions"], order=(1, 1, 1))
        fit_model = model.fit()

        # Make predictions for the next week
        forecast = fit_model.get_forecast(steps=1)
        predicted_value = forecast.predicted_mean.values[0]

        print(f"Forecast for Store {store_nbr}: {predicted_value}")

    # Stop the Spark session
    spark.stop()

def transactions_prediction_final():
    csv_path = "C://Kasyap//projects//bigdata//data//transactions.csv"
    df = pd.read_csv(csv_path, parse_dates=['date'], dayfirst=True)

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
        print(f"Predicted Transactions for Store {store_nbr} on the last day:\n{predicted_values.iloc[-1]}\n")

    # Calculate and print the average prediction for each store at the end
    average_predictions = pd.Series(last_day_predictions).mean()
    print(f"\nAverage Predicted Transactions for Each Store on the Last Day:\n{average_predictions}")

transactions_prediction_final()
exploratory_transactions()
exploratory_items()
clustering()
new_clustering()



