# Team 9 repository - MCV M6. Video analysis

Welcome! You have just found our github repository for
[Module 6 "Video Analysis"](http://pagines.uab.cat/mcv/content/m6-video-analysis)
of the [Master in Computer Vision](http://pagines.uab.cat/mcv/).

This project aims to develop a video surveillance system for road
traffic monitoring. It allows the authors to explore and apply in a
real problem some of the video processing techniques shown in the
_Module 6_ lectures, such as background substraction, foreground
segmentation and filtering, optical flow estimation, video
stabilization or region tracking.

You can also check our [workshop slides](https://docs.google.com/presentation/d/1XTjxj2qV_XuitkfQL0ZdJQsu-eehlLoTKeotdI-agRs/edit?usp=sharing) or download them as "M6-Team_9-Workshop.pdf" from this repository.

## Environment and dependencies

We have developed and test our code with:

- Python 3.6
- numpy 1.14.1
- matplotlib 2.1.2
- scikit-learn 0.19.1
- opencv-contrib-python 3.4.0.12
- pillow 5.0.0
- imageio 2.2.0
- scikit_image 0.13.1
- imutils 0.4.6
- pandas 0.22.0
- scipy 1.0.0

A `requirements.txt` with the above python dependencies can be found
in the root folder of the repository. You may want to install them by
running:

```bash
pip install -r requirements.txt
```

## Datasets

During the development process we will use the following datasets to
evaluate the performance of our system:

- [ChangeDetection.NET (CDNET)](http://changedetection.net/)
- [KITTI dataset for Optical Flow 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow)
- [C-17 highway sequences for the 5th Workshop on Road Traffic Monitoring](https://drive.google.com/open?id=1ByWklwjsNoko40NptaSBvt5L5gOrFwt4)

You may need to copy the datasets in the `datasets/` folder to run some
of the weekly tasks. When needed, it will be reported in the weekly
code sections below.


## Material

### Final report

In the following link you will find our final report summarizing the
stages and methods applied:

- Mas, I., Prol, H., Puyoles, J., Grau, M., *[Video Surveillance for Road Traffic Monitoring](https://www.overleaf.com/read/jnbhxkndbrmt)*, 5th Workshop on Road Traffic Monitoring, 2018.

### Presentations

In addition to the source code we will keep updated this section with
the links to weekly presentations, which summarize the work done and
results:

- Week 1: [Introduction to the datasets, evaluation metrics and optical
  flow analysis](https://docs.google.com/presentation/d/1VQUlbHy3PaaCxYBiEG8HufPkYlS-PI2vXVjm6JIfX0Q/edit?usp=sharing).
- Week 2: [Background estimation and Stauffer & Grimson's approach](https://docs.google.com/presentation/d/1aI1owlfyb7za4ij8lUc4j1mNEmD4aXV679oRF2lDiBk/edit#slide=id.p)
- Week 3: [Foreground segmentation, filtering and shadow removal](https://docs.google.com/presentation/d/1bLqRug-OUk6e5cf1uCqNKUt3DlqElFyYameDbwt_n-A/edit?usp=sharing)
- Week 4: [Optical flow and video stabilization](https://docs.google.com/presentation/d/1MSC-2cTM0PF6hQIKMk92vDQJizMgdOAmFFFq6BjObpU/edit#slide=id.p).
- Week 5: Region tracking and Kalman filter, included in our
  [Final presentation for the 5th Workshop on Road Traffic Monitoring](https://docs.google.com/presentation/d/1XTjxj2qV_XuitkfQL0ZdJQsu-eehlLoTKeotdI-agRs/edit?usp=sharing)

### Week 1 Code

Dataset prerequisites:

* Download the [results_testAB_changedetection.zip](https://e-aules.uab.cat/2017-18/pluginfile.php/509054/mod_page/content/33/results_testAB_changedetection.zip)
  file and unzip into the `datasets/` folder. The containing
  `test_*_.png` images should be in the
  `datasets/results_testAB_changedetection/results/highway` directory.

* Download the [CDNET 2014 Highway Dataset](http://jacarini.dinf.usherbrooke.ca/static/dataset/baseline/highway.zip)
  file and unzip. Rename the `highway` folder to `gt` and move it
  inside `datasets/results_testAB_changedetection`.

To sequentially run the tasks of this week, `cd` to the root folder of
the repository and execute:

```bash
python run.py week1
```

### Week 2 Code

Dataset prerequisites:

* Download the [CDNET 2014 Highway Dataset](http://jacarini.dinf.usherbrooke.ca/static/dataset/baseline/highway.zip), [CDNET 2014 Fall Dataset](http://jacarini.dinf.usherbrooke.ca/static/dataset/dynamicBackground/fall.zip) and [CDNET 2014 Traffic Dataset](http://jacarini.dinf.usherbrooke.ca/static/dataset/cameraJitter/traffic.zip) datasets and unzip on the datasets folder

* Update the environment by running again:

```bash
pip install -r requirements.txt
```

To run each task of this week, `cd` to the root folder of
the repository and execute:

```bash
python run.py week2_tN
```

where N is the number of the task (1 to 4).

### Week 3 Code

Prerequisites:

* We use the same datasets as Week 2 (see instructions above).
* We have added new python dependencies for the project, namely
  `scikit-image`, for morphology operations. Update the environment by
  running again:

```bash
pip install -r requirements.txt
```

To run each task of this week, `cd` to the root folder of
the repository and execute:

```bash
python run.py week3 -t N -d {highway, fall, traffic}
```

where N is the number of the task (1 to 5) and the `-d` argument
defines the dataset to use. Eg: `python run.py week3 -t 1 -d fall` will
execute Task 1 for the Fall dataset.

### Week 4 Code

Prerequisites:

* Download the [KITTI dataset for Optical Flow  2012](http://kitti.is.tue.mpg.de/kitti/data_stereo_flow.zip) and unzip into the `datasets/` folder. You should have then two folders:
  `datasets/data_stereo_flow/{training, testing}`.
* We have added new python dependencies for the project (imutils and
  pandas). Update the environment by running `pip install -r requirements.txt`.

To run the tasks of this week, `cd` to the root folder of the
repository and execute: `python run.py week4 -t N`, where `N` can be:

* 1 to execute tasks 1.1 and 1.2, related to optical flow estimation.
* 2 to execute tasks 2.1 and 2.2, related to video compensation.

### Week 5 Code

Prerequisites:

* Download the [C-17 highway sequences for the 5th Workshop on Road Traffic Monitoring](https://drive.google.com/open?id=1ByWklwjsNoko40NptaSBvt5L5gOrFwt4) and unzip into
  the `datasets/` folder. It contains two video sequences
  inside a `workshop` folder.

To run the tasks of this week, `cd` to the root folder of the
repository and execute:

* `python run.py week5 -t 1 -d dataset`: run Kalman and Meanshift
  tracking over `dataset` (either 'highway' or 'traffic'). The script
  generates the corresponding video sequences in the project root
  directory.
* `python run.py week5 -t 2 -d dataset`: run the speed estimator on
  `dataset` (either 'highway' or 'traffic') and generates the video
  sequences.
* `python run.py week5 -t 3 -d dataset`: run our application demo for
  the _5th Workshop on Road Traffic Monitoring_. Posible values for
  `dataset` are 'sequence2' and 'sequence3'. It generates a video with
  tracking, speed estimation and vehicle statistics in the [C-17
  highway](https://www.google.es/maps/place/41%C2%B032'39.9%22N+2%C2%B013'26.6%22E/@41.544427,2.2218703,17z/data=!3m1!4b1!4m5!3m4!1s0x0:0x0!8m2!3d41.544427!4d2.224059).

## Authors

- [Ignasi Mas](mailto:ignasi.masm@e-campus.uab.cat)
- [Hugo Prol](mailto:hugo.prol@e-campus.uab.cat)
- [Jordi Puyoles](mailto:jordi.puyoles@e-campus.uab.cat)
- [Marc Grau](mailto:marc.grau@e-campus.uab.cat)
