<p align="center">
  <h3 align="center">Physical Bullying Detection</h3>
</p>


<p align="center">
  Building 3D Convolutional Neural Network (CNN) to classify bullying action
</p>

<p align="center">
    <img alt="Python" title="Python" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
</p>

<p align="center">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white">
  <img alt="Numpy" title="Numpy" src="https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff"/>
    <img alt="Scikit Learn" title="Scikit Learn" src="https://img.shields.io/badge/Scikit%20Learn-F38020?logo=scikitlearn&logoColor=white"/>
  <img alt="FastApi" src="https://img.shields.io/badge/FastAPI-009485.svg?logo=fastapi&logoColor=white">
</p>

<p align="center">
    <img alt="Streamlit" title="Streamlit" src="https://img.shields.io/badge/Streamlit-%23DD0031.svg?logo=streamlit&logoColor=white"/>
  <img alt="Ngrok" title="Ngrok" src="https://img.shields.io/badge/Ngrok-512BD4?logo=ngrok&logoColor=fff"/>
</p>

<p align="center">
    <a href="https://drive.google.com/file/d/10qEs9j3TdOpLp2Xh7fKIko1vGpo8OhGg/view?usp=sharing">
      <img src="https://custom-icon-badges.demolab.com/badge/-Click%20Me%20to%20View%20Demo%20Video-000000?style=for-the-badge&logoColor=white" title="Demo Video" alt="Demo Video"/></a>
  </p>
  
<p align="center">
    <a href="https://drive.google.com/file/d/10qEs9j3TdOpLp2Xh7fKIko1vGpo8OhGg/view?usp=sharing">Web application demo video</a>
</p>


---



---

### Overview
- **Goals** -- Provide physical bullying classifier to identify physical bullying type.
- **Dataset** -- Using secondary data pulled from `https://www.brain-cog.network/dataset/Bullying10k/`. Dataset contain bullying action with `.npy` forrmat that represent action in frames.
- **User Interface** -- Build using streamlit module, contain four pages (Classifier, Model Breakdown, Bullying 101, Bullying Case Files).
- **Models** -- Trained 3 models. Best model will be deploy on streamlit that can be accessed through ngrok local tunnel.

---

### Models
1. **Convolutional 3D (C3D)** -- C3D captures spatial and temporal features in videos using 3D convolutions, making it well-suited for action recognition.
2. **Expanded 3D (X3D)** -- X3D expands on traditional 3D convolutions by optimizing kernel size and adding more depth to layers for improved feature extraction.
3. **Inflated 3D (I3D)** -- I3D inflates 2D convolutional filters into 3D, using pre-trained ImageNet models to initialize weights and improve performance on video-based tasks.

--- 

### Model Evaluation
- `BATCH_SIZE`: 4   
- `LEARNING_RATE`: 1e-5  
- `EPOCHS`: 30      
- `WEIGHT_DECAY`: 1e-4  

Model | Accuracy | 
----- | --- |
Convolutional 3D  (C3D) | 80% |
Inflated 3D (I3D) | 54% |
Expanded 3D (X3D) | 76% |
