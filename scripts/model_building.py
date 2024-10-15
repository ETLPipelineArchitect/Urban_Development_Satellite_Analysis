from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Initialize Spark Session
spark = SparkSession.builder.appName("ModelBuilding").getOrCreate()

# Load features
features_path = "s3://your-bucket-name/satellite_images/features/pca_features.parquet"
features_df = spark.read.parquet(features_path)

# Build KMeans model
kmeans = KMeans(featuresCol='pca_features', k=5)  # Example: cluster into 5 clusters
model = kmeans.fit(features_df)

# Evaluate model
predictions = model.transform(features_df)
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print(f'Silhouette Score: {silhouette}')

# Save model
model_path = "s3://your-bucket-name/satellite_images/models/kmeans_model"
model.save(model_path)

print("Model building completed and model saved to S3.")
