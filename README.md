
## Traffic Lights Detection and Classification


This is a fork of original Bosch code, modified by Kung Fu Panda team to
use in Udacity Self-Driving Car Engineer Nanodegree Capstone project.

## Setup

The Bosch Small Traffic Lights Dataset
can be downloaded [here](https://hci.iwr.uni-heidelberg.de/node/6132).

* Please only download `rgb` named archives.
* Put the files in `data` folder.
* Concatenate multi-part zip archives like `cat x.zip.001 x.zip.002 > z.zip`
* Extract the files and re-arrange so the folder structure is as follows:
```
data
├── rgb
│   ├── additional
│   │   ├── 2015-10-05-10-52-01_bag
│   │   │   ├── 24594.png
│   │   │   ├── 24664.png
│   │   │   └── 24734.png
│   │   ├── 2015-10-05-10-55-33_bag
│   │   │   ├── 56988.png
│   │   │   ├── 57058.png
...
│           ├── 238804.png
│           └── 238920.png
├── rgb
│   ├── train
...
├── rgb
│   ├── test
...
├── additional_train.yaml
├── test.yaml
└── train.yaml
```

You can verify/view the data using:
```bash
python dataset_stats.py data/train.yaml
python show_label_images.py data/train.yaml
```

## Extracting Traffic Light Images from Annotated Pictures

In order to train a classifier we created a script to
extract and save actual traffic lights images into
separate folders. You can do the extraction as follows:

```bash
python save_tl_images.py data/train.yaml data/tl-extract-train
```

You may see the following warnings:

```
libpng warning: Image width is zero in IHDR
libpng error: Invalid IHDR data
```

Please ignore them. The saved pictures are valid and usable.

The resulting images (of variable sizes) are saved with the following
naming convention, i.e. sequential number padded to 6 digits
then lower-case name of the class.

```
-rw-r--r--      1 alexeysimonov  staff     177 27 Aug 15:55 000001_yellow.png
-rw-r--r--      1 alexeysimonov  staff     188 27 Aug 15:55 000002_yellow.png
-rw-r--r--      1 alexeysimonov  staff     245 27 Aug 15:55 000003_yellow.png
-rw-r--r--      1 alexeysimonov  staff     159 27 Aug 15:55 000004_redleft.png
-rw-r--r--      1 alexeysimonov  staff     200 27 Aug 15:55 000005_red.png
-rw-r--r--      1 alexeysimonov  staff     257 27 Aug 15:55 000006_red.png
-rw-r--r--      1 alexeysimonov  staff     228 27 Aug 15:55 000007_redleft.png
-rw-r--r--      1 alexeysimonov  staff     265 27 Aug 15:55 000008_red.png
-rw-r--r--      1 alexeysimonov  staff     220 27 Aug 15:55 000009_red.png
-rw-r--r--      1 alexeysimonov  staff     329 27 Aug 15:55 000010_red.png
-rw-r--r--      1 alexeysimonov  staff     199 27 Aug 15:55 000011_redleft.png
-rw-r--r--      1 alexeysimonov  staff     352 27 Aug 15:55 000012_red.png
-rw-r--r--      1 alexeysimonov  staff     195 27 Aug 15:55 000013_redleft.png
-rw-r--r--      1 alexeysimonov  staff     307 27 Aug 15:55 000014_red.png
-rw-r--r--      1 alexeysimonov  staff     244 27 Aug 15:55 000015_red.png
```

## Train a Classifier

We have defined `TLClassifierCNN` in `tl_classfier_cnn.py`
loosely based on CIFAR-10 network architecture.
To train it update training parameters in `train.py` and run
as follows:
```bash
python train.py
```

It saves checkpoints and summaries as it goes along.
At the end it saves the model.

## Predictions

`TLClassifierCNN` can load a pre-trained model to run predictions.
The following script demonstrates the model loading and prediction:
```bash
python predict.py
```
