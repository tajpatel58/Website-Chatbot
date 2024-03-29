from google.cloud import aiplatform
import os

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/tajsmac/Documents/Google-Cloud/GCP_Creds.json"

VERSION = 1
APP_NAME = 'chatbot'
CUSTOM_PREDICTOR_IMAGE_URI = 'gcr.io/website-chatbot-355122/chatbot:latest'
model_display_name = f"{APP_NAME}-v{VERSION}"
model_description = "Chatbot Model for Portfolio Website"

MODEL_NAME = APP_NAME
health_route = "/ping"
predict_route = f"/predictions/{MODEL_NAME}"
serving_container_ports = [8080]

model = aiplatform.Model.upload(
    display_name=model_display_name,
    description=model_description,
    location='us-central1',
    serving_container_image_uri=CUSTOM_PREDICTOR_IMAGE_URI,
    serving_container_predict_route=predict_route,
    serving_container_health_route=health_route,
    serving_container_ports=serving_container_ports,
)

model.wait()

print(model.display_name)
print(model.resource_name)

# Create Endpoint
endpoint_display_name = f"{APP_NAME}-endpoint"
endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)

# Deploy Model to Endpoint:
traffic_percentage = 100
machine_type = "n1-standard-4"
deployed_model_display_name = model_display_name
min_replica_count = 1
max_replica_count = 1
sync = True

model.deploy(
    endpoint=endpoint,
    deployed_model_display_name=deployed_model_display_name,
    machine_type=machine_type,
    traffic_percentage=traffic_percentage,
    sync=sync,
)

model.wait()