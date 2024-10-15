from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("DataAnalysis").getOrCreate()

# Load predictions
model_path = "s3://your-bucket-name/satellite_images/models/kmeans_model"
predictions_df = spark.read.parquet(model_path)

# Analysis: Count predictions by cluster
cluster_counts = predictions_df.groupBy("prediction").count()
cluster_counts.show()

# Save analysis results
output_path = "s3://your-bucket-name/satellite_images/output/cluster_analysis.parquet"
cluster_counts.write.parquet(output_path)

print("Data analysis completed and results saved to S3.")
