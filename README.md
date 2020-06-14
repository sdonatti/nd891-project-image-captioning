# Udacity Computer Vision Nanodegree

## Image Captioning

Combine convolutional and recurrent neural networks to build an automatic image captioning application.

![teaser](./images/social_profile_cvnd_sample.png)

### Requirements

1. Download and install [Anaconda Python](http://www.anaconda.com)
2. Create and activate a [Conda environment](http://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

### Set-up

Clone the project repository
```
git clone http://github.com/sdonatti/nd891-project-image-captioning
```

Install required Python packages
```
cd nd891-project-image-captioning
conda install --file requirements.txt -c pytorch
```

Install [COCO API](http://github.com/cocodataset/cocoapi)
```
git clone http://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py install
```

Download [COCO Dataset](http://cocodataset.org/#download)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json at cocoapi/annotations/)
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json at location cocoapi/annotations/)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract train2014 folder at cocoapi/images/)
  * **2014 Val images [41K/6GB]** (extract val2014 folder at location cocoapi/images/)
  * **2014 Test images [41K/6GB]** (extract test2014 folder at location cocoapi/images/)

Launch the project Jupyter Notebooks
```
cd ../../
jupyter notebook
```

### License

This project is licensed under the [MIT License](LICENSE)