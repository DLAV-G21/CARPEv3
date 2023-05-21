## Deep Learning for Autonomous Vehicle : Group 12

In this repository, you'll find all the code to run our project: 2D semantic keypoints detection. 

### Goal of the project

The idea of the project is to output the different keypoints of a car from an image input. 
This field has been widely studied in the litterature and our idea is to use a reasoning similar to Openpifpaf where we want to output different links and keypoints and then merge them into a skeleton using a decoder. Compared to Openpifpaf, we want to use a transformer backbone instead of a CNN backbone. The decoder is also based on OpenPifPaf but we remove the Non-Maxima Suppression as our model outputs precise locations instead of heatmaps in the case of CNNs.

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
This project relies on the ApolloCar3D dataset that is available [here](https://github.com/ApolloScapeAuto/dataset-api/blob/master/car_instance/README.md). It contains 5'277 high quality images from the road containing a certain amount of cars. You can find a preliminary data exploration of this dataset in the [exploratory data analysis notebook](eda.ipynb).
With the data, we then use the [openpifpaf](https://github.com/openpifpaf/openpifpaf) function to convert the semantic keypoints to a version that is similar to Coco. The exact command to generate the file is :

```
pip install openpifpaf
pip install opencv-python
python3 -m openpifpaf.plugins.apollocar3d.apollo_to_coco --dir_data PATH_TO_DS/3d-car-understanding-train/train --dir_out PATH_TO_DS/3d-car-understanding-train/annotations
```
This will generate keypoints in the Coco format for both training and validation annotations in 24 or 66 keypoints. This is the first conversion we use. 

### Train


### Inference


