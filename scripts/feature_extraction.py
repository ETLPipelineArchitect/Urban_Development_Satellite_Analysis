from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.sql.functions import col
import pandas as pd

# Initialize Spark Session
spark = SparkSession.builder.appName("FeatureExtraction").getOrCreate()

# Load processed data
processed_data_path = "s3://your-bucket-name/satellite_images/processed/processed_images.parquet"
processed_df = spark.read.parquet(processed_data_path)

# Perform PCA
pca = PCA(k=10, inputCol='image_data', outputCol='pca_features')
pca_model = pca.fit(processed_df)

# Transform data to get PCA features
pca_result = pca_model.transform(processed_df)

# Save features
features_path = "s3://your-bucket-name/satellite_images/features/pca_features.parquet"
pca_result.write.parquet(features_path, mode='overwrite')

print("Feature extraction completed and features saved to S3.")
