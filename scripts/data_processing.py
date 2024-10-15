import os
import boto3
from PIL import Image
import numpy as np
from pyspark.sql import SparkSession

# Initialize Spark Session
spark = SparkSession.builder.appName("SatelliteImageProcessing").getOrCreate()

# Initialize AWS S3 Client
s3 = boto3.client('s3')

# Define S3 bucket and prefix
bucket_name = "your-bucket-name"
prefix = "satellite_images/"

def process_image(image_key):
    """Process and save the image from S3."""
    # Download image
    image_obj = s3.get_object(Bucket=bucket_name, Key=image_key)
    with Image.open(image_obj['Body']) as img:
        img = img.resize((256, 256))  # Resize image
        img_np = np.array(img)  # Convert to numpy array
        return img_np

# List and process images
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
images_data = []
for obj in response.get('Contents', []):
    image_key = obj['Key']
    if image_key.endswith(('jpg', 'png')):
        img_array = process_image(image_key)
        images_data.append((image_key, img_array.tolist()))  # Convert numpy array to list for storage

# Create DataFrame
images_df = spark.createDataFrame(images_data, ["image_name", "image_data"])
processed_data_path = "s3://your-bucket-name/satellite_images/processed/processed_images.parquet"
images_df.write.parquet(processed_data_path, mode='overwrite')

print("Data processing completed and saved to S3.")
