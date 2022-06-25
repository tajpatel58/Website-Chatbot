torchserve --start \
           --ncs \
           --model-store deployment/model-store \
           --ts-config deployment/config.properties \
           --models chatbot=Chatbot_Model.mar