torch-model-archiver \
    --model-name Chatbot_Model \
    --version 1.0 \
    --model-file ./Scripts/model.py \
    --serialized-file ./Models/chat_model.pth \
    --handler ./Scripts/chatbot.py \
    --requirements-file ./requirements.txt \
    --extra-files ./Scripts/text_cleaning.py,./Scripts/model.py


