FROM pytorch/torchserve:latest-cpu

# Copy Requirements
COPY ./requirements.txt .

# Install Dependencies:
RUN pip install -r requirements.txt

# Make Directories:
RUN mkdir -p /home/model-server/deployment
RUN mkdir -p /home/model-server/Scripts
RUN mkdir -p /home/model-server/Models

#Copy Model Artifacts, Custom Handlers, etc. 
COPY ./deployment /home/model-server/deployment
COPY ./Models /home/model-server/Models
COPY ./Scripts/chatbot.py /home/model-server/Scripts
COPY ./Scripts/model.py /home/model-server/Scripts
COPY ./Scripts/text_cleaning.py /home/model-server/Scripts
COPY ./requirements.txt /home/model-server/

USER root

# Remove default directories. 
RUN rm -rf model-store
RUN rm -rf config.properties

# Expose Ports for Inference and Health Check: (Described in Config.Properties)
EXPOSE 8080
EXPOSE 8081
EXPOSE 8082

# Run Model Archiver. 
RUN torch-model-archiver \
    --model-name Chatbot_Model \
    --version 1.0 \
    --model-file ./Scripts/model.py \
    --serialized-file ./Models/chat_model.pth \
    --handler ./Scripts/chatbot.py \
    --requirements-file ./requirements.txt \
    --extra-files ./Scripts/text_cleaning.py,./Scripts/model.py \
    --export-path ./deployment/model-store \
    --force

# Once the container is running, start hosting the model. 
CMD ["torchserve", \
     "--start",  \
     "--ncs", \
     "--model-store deployment/model-store", \
     "--ts-config deployment/config.properties", \
     "--models chatbot=Chatbot_Model.mar"]