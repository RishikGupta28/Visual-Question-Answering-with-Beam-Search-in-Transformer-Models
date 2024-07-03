# Visual-Question-Answering-with-Beam-Search-in-Transformer-Models

## GitHub Repository Description

### Project: Fine-tuning ViLT for Visual Question Answering (VQA)

This repository contains the implementation for fine-tuning the ViLT (Vision-and-Language Transformer) model for the task of Visual Question Answering (VQA). The approach is based on the paper [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334).

### Dataset Details

- **VQAv2 Dataset**: The dataset used for this project is VQAv2, which can be downloaded from the [official website](https://visualqa.org/download.html). It consists of images from the COCO dataset along with corresponding questions and answers.
- **Images**: Images are downloaded from the [COCO dataset](http://images.cocodataset.org/), specifically the validation set (val2014).

### Tech Stack

- **Programming Language**: Python
- **Libraries and Tools**:
  - `transformers`: For utilizing the ViLT model from Hugging Face.
  - `flask`: For creating a web application to demonstrate the VQA model.
  - `PIL` and `numpy`: For image processing tasks.
  - `tqdm`: For progress tracking.

### Algorithm and Approach

1. **Environment Setup**: Install the necessary libraries using pip.
    ```python
    !pip install -q transformers
    ```

2. **Data Loading**: Download and unzip the VQAv2 questions and COCO validation images.
    ```python
    !wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
    !unzip /content/v2_Questions_Val_mscoco.zip
    !wget http://images.cocodataset.org/zips/val2014.zip
    !unzip /content/val2014.zip
    ```

3. **Preprocessing**:
    - Load the JSON files containing the questions.
    - Extract question IDs and map them to the corresponding image filenames.

4. **Fine-tuning ViLT**:
    - Initialize the ViLT model and processor from the Hugging Face library.
    - Fine-tune the model on the VQAv2 dataset by preparing image-question pairs and training the model to predict answers.

5. **Web Application**:
    - A Flask web application is created to serve the VQA model.
    - Users can upload an image and ask a question about the image. The model then predicts the answer.
    - Various image processing techniques (Gaussian blur, Laplace filter, Sobel filter, Prewitt filter) are applied to observe their effects on the model's predictions.

### Usage

To run the project locally:
1. Clone the repository.
2. Install the required libraries.
3. Download the VQAv2 dataset and COCO images.
4. Run the Jupyter notebook for fine-tuning the model.
5. Start the Flask application to use the VQA model through a web interface.

### Example Code Snippets

#### Loading Data
```python
import json

# Opening JSON file
f = open('/content/v2_OpenEnded_mscoco_val2014_questions.json')
data_questions = json.load(f)
```

#### Fine-tuning the Model
```python
from transformers import ViltProcessor, ViltForQuestionAnswering
import torch

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
```

#### Flask Web Application
```python
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageFilter
import numpy as np
from transformers import ViltProcessor, ViltForQuestionAnswering

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('group_14_gui/templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle file upload and question input
    pass

if __name__ == '__main__':
    app.run()
```

This repository provides a comprehensive approach to fine-tuning ViLT for VQA and demonstrates the model's capabilities through a user-friendly web application.
