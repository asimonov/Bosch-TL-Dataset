
## Traffic Lights Detection and Classification Dataset

This is a fork of original Bosch code, modified by Kung Fu Panda team to
use in Udacity Self-Driving Car Engineer Nanodegree Capstone project.

## Bosch Small Traffic Lights Dataset

The dataset can be downloaded [here](https://hci.iwr.uni-heidelberg.de/node/6132).

Getting and extracting the data:
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
```python
python dataset_stats.py data/train.yaml
python show_label_images.py data/train.yaml
```


