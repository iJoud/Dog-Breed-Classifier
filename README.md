# Dog Breeds Classifier


## Table of Contents
1. [Overview](#Overview)
2. [Files](#Files)
3. [Prerequisites & Running](#Prerequisites_Running)
4. [Acknowledgements](#Acknowledgements)

<a id="Overview"></a>
## Overview
In this project, I completed the capstone project for my Data Scientist Nanodegree. I've trained multiple convolutional neural networks (CNN) to classify dog images to one of the 133 different dog breeds in the dataset. I chose the best performing model to deploy it in a web application to facilitate inference processes for the user. The model is capable of detecting whether the image contains a dog or human! The model will classify the human image into the breeds that it most closely resembles.

<img width="439" alt="Screenshot 2023-03-29 at 7 01 15 AM" src="https://user-images.githubusercontent.com/89661060/228685581-d4a5dcd9-8965-4000-8d56-7c84b9a593c1.png"> <img width="496" alt="Screenshot 2023-03-29 at 7 01 40 AM" src="https://user-images.githubusercontent.com/89661060/228685651-d16205d4-f2f5-4d43-bb2e-2913fab915e6.png"> <img width="439" alt="Screenshot 2023-03-29 at 7 01 25 AM" src="https://user-images.githubusercontent.com/89661060/228685660-2c194264-66ae-4785-87c4-d6e06ca38233.png">

<a id="Files"></a>
## Files

```bash
    
    - project                                   # Folder contains all the models training details and files
    ├── bottleneck_features                     # Folder to place downladed bottleneck features for chosen model
    ├── haarcascades                          
    │   └── haarcascade_frontalface_alt.xml     # Haar feature-based cascade classifiers to detect human faces in images model
    ├── dogs_and_humans_images                  # Folder conatins random images for humans, dogs, and objects.
    ├── images                                  # Folder images used in the notebook cells
    ├── requirements                            # Folder conatins requirements files to create environments to run the project on local machine
    │   ├── dog-linux-gpu.yml                   # Requirements to run the project on linux systems with GPU support
    │   ├── dog-linux.yml                       # Requirements to run the project on linux systems
    │   ├── dog-mac-gpu.yml                     # Requirements to run the project on macos systems with GPU supprot
    │   ├── dog-mac.yml                         # Requirements to run the project on macos systems 
    │   ├── dog-windows-gpu.yml                 # Requirements to run the project on windows systems with GPU supprot
    │   ├── dog-windows.yml                     # Requirements to run the project on windows
    │   ├── requirements-gpu.txt                # Requirements to run the project if you're using AWS with GPU supprot
    │   └── requirements.txt                    # Requirements to run the project if you're using AWS 
    ├── saved_models 
    │   └── weights.best.Inception.hdf5         # The fine-tuned model file 
    ├── dog_app.ipynb                           # Notebook contains all the training details
    ├── extract_bottleneck_features.py          # Python file contains functions to extract bottleneck features from multiple models.
    ├── README.md                               # Detailed readme file for all the details related to the website 
    └── LICENSE.txt

    - web-application                        # Folder contains the html template and python file to run the web app.
    ├── static                               # Folder to store uploaded images by the user
    ├── templates
    │   └── index.html                       # website home page 
    ├── requirements.txt                     # Requirements to run the project
    └── dog_names                            # JSON file contains the list of dog breeds names 
    
    - README.md
    - LICENSE.txt

```


<a id="Prerequisites_Running"></a>
## Prerequisites & Running
The best model weights are already stored and deployed on the web app. To run the web application, you need to create a virtual environment using the following commands in your terminal:

```
conda create --name tf --file requirements.txt 
conda activate tf
cd web-application
python run.py
```

After running **run.py**, open your browser on `http://localhost:8000`. You should see the following webpage, and you can start trying the deployed model.

<img width="1161" alt="Screenshot 2023-03-30 at 2 45 44 AM" src="https://user-images.githubusercontent.com/89661060/228692254-34d703b9-9d69-4e55-8257-23290c2313d1.png">

<a id="Acknowledgements"></a>
## Acknowledgements
The following resourses helped me with this project:
- [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
- [Building a simple Keras + deep learning REST API](https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)


