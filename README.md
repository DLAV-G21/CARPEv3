## Deep Learning for Autonomous Vehicle : Group 12

In this repository, you'll find all the code to run our project: 2D semantic keypoints detection. 

### Goal of the project

The idea of the project is to output the different keypoints of a car from an image input.
Starting from OpenPifPaf, we had 2 main ideas: 
1. Move to a full-transformer architecture to give an image and output the keypoints as scalar values and the class as a single integer.  
2. Make use of Data Augmentation specifically design to hide part of cars to help the network to be robust to occlusion. 

### Model design 


However, our journey proved more challenging than anticipated. Initially, we planned to use the HRFormer transformer as the common backbone model, assuming it was mandatory based on the project requirements. The transformer heads were intended to predict the classes and positions of each keypoint (24 classes + 1 no-keypoint class). We implemented the entire codebase from scratch, which can be found [here](https://github.com/DLAV-G21/CARPEvRIP). Unfortunately, we encountered a significant issue where the network failed to learn anything. Despite investing substantial time in debugging, we realized that since we had implemented multiple components ourselves (model, decoder, training loop, dataset, etc.), identifying the specific source of the problem became exceedingly difficult.

The original idea was as follows:

1. The HRFormer backbone would extract relevant features from the input image.
2. Queries for keypoints and links would be passed to the decoder. Each query would generate a 27-dimensional vector for keypoints or a 54-dimensional vector for links, including:
	- Keypoints:
		- (x, y) position
		- A distribution over 25 classes (24 keypoint classes and 1 no-keypoint class)
	- Links:
		- Two sets of (x, y) positions for the link endpoints
		- A distribution over 50 classes (49 keypoint classes and 1 no-keypoint class)
3. A greedy decoder would be employed to associate keypoints and links belonging together, resulting in the generation of final skeletons. This decoding process would resemble OpenPifPaf but without the need for Non-Maximum Suppression (NMS), as our approach relied on scalar values rather than heatmaps. 

Unfortunately, due to the aforementioned learning issue and the complexity of the self-implemented components, pinpointing the root cause of the problem proved challenging. veloping the entire codebase took a considerable amount of time, until May 12th. Subsequently, we dedicated one week to resolving the bug, but unfortunately, we were unable to obtain a working solution.

ollowing a meeting with our TA on May 19th, we concluded that debugging the existing codebase would be too challenging. We decided to start from a more comprehensive codebase and minimize modifications. Additionally, we were informed that using HRFormer was not mandatory. Thus, we switched to the [DETR](https://ai.facebook.com/research/publications/end-to-end-object-detection-with-transformers) ([Github](https://github.com/facebookresearch/detr)) models as our starting point and built upon the existing codebase for keypoint detection. Our modifications were based on the [PE-Former](https://github.com/padeler/PE-former) repository, customized to our specific problem. Instead of using specific models as presented in the [PE-Former paper](https://arxiv.org/pdf/2112.04981.pdf), we focused solely on the VisionTransformer. While the results were improved compared to our previous attempts, we observed that this technique was not really effective for predicting multiple instances in one image. In PE-Former, they utilized 100 queries to find a maximum of 17 keypoints. With more than 10 cars per image, each potentially having 24 keypoints and 49 links, the number of queries for keypoints would become impractical, not to mention the links. Our adapted codebase can be found [here](). 


 and we build upon that codebase to have a codebase for keypoint detection (related to  but adapted to our problem and with only the VisionTransformer and not specific model as they did in [their paper]. Even though the results were better, i.e. the learning was actually learning something and the points were dispersed and not all at the same place, we saw that this technique was really effective if we want to predict a single instance of car/person in an image and not multilple (as explained in the [PE-Former paper](https://arxiv.org/pdf/2112.04981.pdf)). In PE-Former, they use 100 queries to find at most 17 keypoints. With more than 10 cars per image which can have at most 24 keypoints and 49 links, the number of queries for keypoints will be too high and we don't talk about the links. Our adapated codebase can be found [here](https://github.com/DLAV-G21/CARPEv2).

At this point, we were thinking of the most effective way to translate our multi-instance pose estimation problem with transformers.We considered using input queries for each skeleton in an image to eliminate the need for a specific greedy decoder and reduce the required number of queries. At this point, we cross the road of the [End-to-End Trainable Multi-Instance Pose Estimation with Transformers](https://arxiv.org/pdf/2103.12115.pdf) paper authored by EPFL researchers, which aligned with our thinking. The proposed approach involves issuing a query for each skeleton instance in the image, resulting in the generation of a 76-dimensional vector per query. Specifically:

1. Each query corresponds to one instance car/person in the image 
2. Each query will generate a 76 dimensional vector.
- 2 values indicating visibility (0 if the car is visible, 1 if it is not)
- 2 values indicating the car's center position
- 24\*3 values for each keypoint, representing the offset from the center point and whether the point is visible or not
3. The network is trained with 4 different losses (for which you can find more details in [the original paper]((https://arxiv.org/pdf/2103.12115.pdf))) 

Our final repository, which you are currently viewing, is a forked and modified version of the [original repository](https://github.com/pranoyr/End-to-End-Trainable-Multi-Instance-Pose-Estimation-with-Transformers) We tailored it to work with cars and implemented additional enhancements for better visualizations. This approach proved more intuitive and yielded some positive results over the training set.

We would like to gracefully thanks Prof. Alexander Mathis for giving us access to the pretrained weights of their model. We will look at the performance if we fine-tuned a model made for human/animal and transfers its knowledge for car. This will hopefully increase our results even more. 

### Data augmentation
We also made an attept at creating a specific kind of data augmentation applied to our problem. Specifically, in Autonomous vehicles we have one big challenge which are occlusions. It is often the case that we have some fence, ads or other vehicle blocking the view of vehicle. As an attempt to help the network to learn to handle this case, we augment our data in two different ways:
- First, we discard random patches in the images of a specific size to simulate an occlusion. 
- We use RCNN to detect all cars in the image and generate a segmentation. At training, with some probability, we blur parts of cars or the background (so taht the network doesn't only learn that in the blurry region, a car is present.

### Metrics
To measure the performance of our network, we rely on the Object Keypoint Similarity defined in MS-COCO as 

<img src="docs/oks.png" alt="Benchmark" width="600" style="background-color:#2e3136">

### Experiments
To test our networks and impact of our data augmentation, we perform several training taking into account different augmentations. We report the performances below using the COCO eval files from the PE-Former repository with values of sigma set to 0.5 for each keypoint.

We performed the following experiments:
1. Baseline (only resize and crop)
2. Classical data augmentation using the albumentations library. 
3. Occlusion data augmentation as explained above. 
4. Take the pretrained model from here  and train our model from that.

|              |  AP  | AP.5 | AP.75 | AP medium | AP large | AR | AR.5 | AR.75 | AR medium | AR large | Checkpoint|
|--------------|------|------|-------|-----------|----------|----|------|-------|-----------|----------|-----------|
| Baseline     | 0.297 | 0.399 | 0.299 | 0.222 | 0.481 | 0.346 | 0.444 | 0.347 | 0.221 | 0.542 | [Download]() |
| Classical DA | 0.274 | 0.378 | 0.277 | 0.225 | 0.439 | 0.354 | 0.459 | 0.355 | 0.222 | 0.555 | [Download]() |
| Occlusion DA |      |      |       |           |          |    |      |       |           |          | [Download]() |
| Fine-tuning  |      |      |       |           |          |    |      |       |           |          | [Download]() |

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
git clone git@github.com:DLAV-G21/CARPEv3.git
cd CARPEv3
mkdir dlav_data
scp path/images.zip USERNAME@izar.epfl.ch:~/CARPEv3/dlav_data/
unzip ProjectRepository/dlav_data/images.zip
module load gcc/8.4.0 python/3.7.7 
python -m venv --system-site-packages venvs/venv-g21
source venvs/venv-g21/bin/activate
pip install --no-cache-dir -r CARPEv3/requirements.txt
$sbatch run.sh # submit the job
```
where
- `images.zip` is the compression of the images folder from the ApolloCar3D dataset `3d-car-understanding-train/train/images`
- `segm_npy.zip` is the output of the segmentation from `utils/processing.py` file. It is necessary only for training the networks with occlusion augmentation.  
### Dataset
This project relies on the ApolloCar3D dataset that is available [here](https://github.com/ApolloScapeAuto/dataset-api/blob/master/car_instance/README.md). It contains 5'277 high quality images from 6 videos of the road containing a certain amount of cars. You can find a preliminary data exploration of this dataset in the [exploratory data analysis notebook](data_exploration.ipynb).
With the data, we then use the [openpifpaf](https://github.com/openpifpaf/openpifpaf) function to convert the semantic keypoints to a version that is similar to Coco. The exact command to generate the file is :

```
pip install openpifpaf
pip install opencv-python
python3 -m openpifpaf.plugins.apollocar3d.apollo_to_coco --dir_data PATH_TO_DS/3d-car-understanding-train/train --dir_out PATH_TO_DS/3d-car-understanding-train/annotations
```
This will generate keypoints in the Coco format for both training and validation annotations in 24 or 66 keypoints. A sequence of functions defined in [DataSplit](DataSplit.ipynb) is used. The split is based on the size of the videos: 
- Training: videos [180116,180117,171206] for a total of 2571 images.
- Validation:  videos [180114] for a total of 652 images.
- Test: videos [180118,180310] for a total of 1060 images.

The split has been made on the video name so that we don't cheat and get test video frames close to one in the training set. 

The different datasets object created from the annotation, images and segmentation are [coco](datasets/coco.py) (mostly taken from the []() with minor adjustments to our needs) and the [inference dataset](datasets/inference_dataset.py) which is used to make inference using our model. 

For convenience, we provide a download link for the data and the generated annotations [here](https://drive.google.com/file/d/1Mk1vCvPa_ed-vl4JZjoCav-IDuG4heJ9/view?usp=sharing). Note that the data doesn't belong to us and are the property of the [ApolloCar3D dataset](https://arxiv.org/abs/1811.12222).

### Train
You can easily trained your model by giving the following command 
```
python train.py training_name --coco_path path/to/coco/annotations --batch_size 16
```
where `path/to/coco/annotations` is expected to have the following structures
```
annotations
    ├── keypoints_test_24.json	
    ├── keypoints_train_24.json	
    └── keypoints_val_24.json
test
test_segm_npz
train
train_segm_npz
val
val_segm_npz
```
Training name is used to save the best model files in the snapshot folder. 

### Inference
To run the inference script, you will need to provide at least the following arguments: 
- `image_folder`: the folder in which we should look for the images
- `pretrained_weight_path`: the pretrained models weights needed to make inference. 
You can run the following command to use the inference script

```
python inference.py path/to/images path/to/model -j path/to/file.json -v path/to/folder --coco_file_path path/to/coco/annotations/file.json
```
If the path to coco annotations is given (the annotation files corresponding to the imagesin `path/to/images`), the performances will be computed and the Average precision at different level will be displayed. `--viz` allows to make the visualizations and save them in `inference_out_folder`. The option `-j` allows to save the output as a json file in the output folder.

### References 
- <a id="1" href="https://arxiv.org/abs/2005.12872">[1]</a> Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-End Object Detection with Transformers.[Github](https://github.com/facebookresearch/detr)
- <a id="2" href="https://arxiv.org/abs/2103.12115">[2]</a> Stoffl, L., Vidal, M., & Mathis, A. (2021). End-to-End Trainable Multi-Instance Pose Estimation with Transformers.[Github](https://github.com/amathislab/poet)
- <a id="3" href="https://arxiv.org/abs/2112.04981">[3]</a> Panteleris, P., & Argyros, A. (2021). PE-former: Pose Estimation Transformer. [Github](https://github.com/padeler/PE-former)
- <a id="4" href="https://arxiv.org/abs/2103.02440">[4]</a> Kreiss, S., Bertoni, L., & Alahi, A. (2021). OpenPifPaf: Composite Fields for Semantic Keypoint Detection and Spatio-Temporal Association. [Github](https://github.com/openpifpaf/openpifpaf)
- <a id="5" href="https://arxiv.org/abs/1809.06839">[5]</a> Buslaev, A., Parinov, A., Khvedchenya, E., Iglovikov, V. I.,
  & Kalinin, A. A. (2020). Albumentations: fast and flexible image augmentations. Information, 11(2), 125. doi:
  10.3390/info11020125

