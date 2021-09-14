## Towards Multi-Label Image Retrieval for Remote Sensing

This repository contains the implementation of the methods described in: 
#### Towards Multi-Label Image Retrieval for Remote Sensing 
R. Imbriaco, C. Sebastian, E. Bondarev, P.H.N de With [IEEE](https://ieeexplore.ieee.org/document/9491804)

## Setup

The code provided in this repository assumes that the datasets have been already 
acquired from their corresponding sources. Furthermore, additional steps are 
necessary for utilizing BigEarthNet. BigEarthNet should be first pre-processed 
using the scripts in  `src/common/preprocessing/prepare_splits`. These will 
generate the corresponding LMDB objects used during training.  

Afterward, the script `/src/common/preprocessing/retrieval.py` should be executed. 
This generates the necessary ground truth files for testing BigEarthNet.

Packages employed in developing this project are listed on `requirements.txt`.

## Train

Training of a network is done by preparing a YAML configuration file like the 
ones provided in `configs`. This contains all the necessary information for 
training/testing a model including backbones, losses, learning rate, 
augmentations, etc. Some of the functionalities included in this codebase
 are not covered in the original paper. Some of these are:

* Addition of attention modules.
* Convolutional backbones other than ResNet.
* Different part-based models.
* Embedding dimensionality compression. 
* Usage of original BigEarthNet labels.
 
Be aware that functionalities/modules not covered in the original paper
have not been tested and as such no guarantee of them working as intended is made. 


## Test

Testing is done similarly to training in that the same YAML configuration is used.
However, the 'mode' attribute should be changed to 'test'. This will run the 
extraction and retrieval processes. Be aware that due to BigEarthNet's size 
retrieval can take a long time and use large amounts of RAM memory. 

Checkpoints used in our research are available below:

[Checkpoints](https://mega.nz/file/gJQHhQLJ#zux6Db4NCK__A5lGtydRitsUTq84WmlVD-RveZzmCDk)
