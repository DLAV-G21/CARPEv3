## Deep Learning for Autonomous Vehicle : Group 12

In this repository, you'll find all the code to run our project: 2D semantic keypoints detection. 

### Goal of the project

The idea of the project is to output the different keypoints of a car from an image input.
Starting from OpenPifPaf, we had 2 main ideas: 
1. Move to a full-transformer architecture where we input joint
2. Make use of Data Augmentation specifically design to hide part of cars to help the network to be robust to occlusion. 

Unfortunately, our journey was more complicated that it seems. 
Our first idea was to start from an architecture common to the other groups (using the HRFormer transformer), then our transformer heads should learn to predict the classes of each datapoint as well as their position. After coding everything from scratch (code available [here]()), we saw that the network wasn't learning anything at all. After having spent a lot of time trying to debug, we arrived to the conclusion that since a lot of different part of the code have been coded by ourselves, there is too much places where we could have been wrong and therefore it would be too hard to debug. The original idea was this one: 

e.g. The HRFormer backbone would extract all the interesant features from the image and the keypoint / links queries should enter the decoder architecture in order to gives a keypoint class and positions (or two positions in the case of links). The skeleton will be merged together using a decoder (available in the decoder final in the linked repository). Unfortunately, as said before, the network wasn't learning anything at all and due to all the different parts implemented by ourselves, we didn't really where to find our mistakes. 

After a meeting with the TA, we went to the conclusion that it will be too hard to debug and that we should start from a more complete codebase and modify the minimal amount of code. We started from the DETR models and specifically, we build a code that was similar to the one in the PE-Former model. Unfortunately, even though the results were better, i.e. the learning was actually learning something and the points were dispersed and not all at the same place, we saw that this technique is useful only if we want to find single instances of car/person in an image but not multiple. In PE-Former, they use 100 queries to find at most 17 keypoints. With images with more than 10 images the number of queries we should made will be too high. 

This leads to the third attempts in this project where we found the following paper which is written by EPFL collaborators. They did what we were interested in i.e. multi-instance keypoint detection end-to-end with transformer. The idea here is different from the other one. Instead of issuing a query for each kkeypoint and then merge the keypoints belonging to the same skeleton together, we do the following: 
- Each query corresponds to one instance in the image 
- Each query will generate a 2 (if the car is visible or not) + 2 (car center) +  24\*3 (for each keypoint, we predict the offset from the center point and also if the point is visible or not)
- The network is trained with 4 different losses (which you can find more detailed in the original paper) 
This approach was more intuitive and seems more robust.

### Repository structure
This is the file and folder structure of the github repository.

```
model	      					# Folder containing all our models
    ├── losses 					# Folder with the different losses used throughout the training
    ├── hrformer				# clone of the HRFormer repository containing the different parts of the HRFormer model
    ├── model_saves				# the different weights of the different trained_models
    ├── decoder.py				# Python decoder to match links and keypoints and create skeleton for the different car in the images
    ├── head.py					# The transformer deocder 
    ├── neck.py					# model between the transformer encoder and decoder
    ├── net.py					# model putting everything together and our final model.
utils  
    ├── coco_evaluator.py		# A file to convert our dataset to the COCO format. Mix between the one from OpenPifPaf and PE-Former repositories
    ├── eda.py					# helper method to perform the data exploration
    ├── openpifpaf_helper.py 	# constants copies directly from the openpifpaf project repository so that don't need to install the openpifpaf dependcy which is long to install on colab
    ├── processing.py			# helper file containing the mask segmentation as well as the train-val-test split of the dataset.
    └── visualizations.py 		# helper file to generate visualizations for both keypoint and exploratory data analysis.         
DLAV_Data_Exploration			# Jupyter notebook containing a small exploration of the dataset.
Dockerfile 						# File to create the docker image used in the projected
README.md
builder.py 						# convenience script to get optimizer and scheduler from config
dataset.py						# file containing the different dataset used throughout the projects
dlav_config.json				# Script containing all the config values for to run the project
inference.py					# Script to make predictions using our network.
requirements.txt				# All the dependecy library
run.sh							# Convenience sript to run training on Scitas.
setup.sh						# Setup file in the dockerfile
train.py						# script to train the network according to the config
trainer.py 						# python file containing the Trainer class used for training.
training.py			      		# Script to train our models
```

### Installation 

To get the docker image, you can do two different things: 
- Get it from DockerHub using the command:

```
docker pull alessioverardo/dlav_g21:latest
```
- Create the docker image locally using the following two commands: 

```
git clone git@github.com:DLAV-G21/ProjectRepository.git
docker build -t dlav_g21:latest .
```
You can also choose to run everything on your cluster and machine. You can install all the requirements using the command 
```
pip install -r requirements.txt
```

You can also submit jobs on the Scitas cluster using the command

```
ssh -X USERNAME@izar.epfl.ch
ssh-keygen -t rsa -b 4096
cat ~/.ssh/.id_pub
copy the code to  your github account
git clone git@github.com:DLAV-G21/ProjectRepository.git
scp path/images.zip USERNAME@izar.epfl.ch:~/ProjectRepository/dlav_data/
scp path/segm_npy.zip USERNAME@izar.epfl.ch:~/ProjectRepository/dlav_data/ 
unzip ProjectRepository/dlav_data/images.zip
unzip ProjectRepository/dlav_data/segm_npy.zip
module load gcc/8.4.0 python/3.7.7 
python -m venv --system-site-packages venvs/venv-g21
source venvs/venv-g21/bin/activate
pip install --no-cache-dir -r ProjectRepository/requirements.txt
$sbatch run.sh # submit the job
```
where
- `images.zip` is the compression of the images folder from the ApolloCar3D dataset `3d-car-understanding-train/train/images`
- `segm_npy.zip` is the output of the segmentation from `utils/processing.py` file. It is necessary only for training the networks with occlusion augmentation.  
### Dataset
This project relies on the ApolloCar3D dataset that is available [here](https://github.com/ApolloScapeAuto/dataset-api/blob/master/car_instance/README.md). It contains 5'277 high quality images from 6 videos of the road containing a certain amount of cars. You can find a preliminary data exploration of this dataset in the [exploratory data analysis notebook](DLAV_Data_Exploration.ipynb).
With the data, we then use the [openpifpaf](https://github.com/openpifpaf/openpifpaf) function to convert the semantic keypoints to a version that is similar to Coco. The exact command to generate the file is :

```
pip install openpifpaf
pip install opencv-python
python3 -m openpifpaf.plugins.apollocar3d.apollo_to_coco --dir_data PATH_TO_DS/3d-car-understanding-train/train --dir_out PATH_TO_DS/3d-car-understanding-train/annotations
```
This will generate keypoints in the Coco format for both training and validation annotations in 24 or 66 keypoints. This is the first conversion we use. We then use a the function [generate_segmentation](util/processing.py) to generate all segmentation (for Data Augmentation). You can find an example of the usage in [this notebook](DLAV_Data_Exploration.ipynb). After the segmentation has been generated, a sequence of functions defined in [DataSplit](DataSplit.ipynb) is used. The split is based on the size of the videos: 
- Training: videos [180116,180117,171206] for a total of 2571 images
- Validation:  videos [180114] for a total of 652 images
- Test: videos [180118,180310] for a total of 1060 images

The split has been made on the video name so that we don't cheat and get test video frames close to one in the training set. 

### Train


### Inference


