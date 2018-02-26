# Team 9 repository - MCV M6. Video analysis

Welcome! You have just found our github repository for
[Module 6 "Video Analysis"](http://pagines.uab.cat/mcv/content/m6-video-analysis)
of the [Master in Computer Vision](http://pagines.uab.cat/mcv/).

This project aims to develop a video surveillance system for road
traffic monitoring. It allows the authors to explore and apply in a
real problem some of the video processing techniques shown in the
lectures, such as background substraction, foreground segmentation and
filtering, optical flow estimation, video stabilization or region
tracking.

## Environment and dependencies

We have developed and test our code with:

- Python 3.6
- numpy 1.14.1
- matplotlib 2.1.2
- scikit-learn 0.19.1
- opencv-python 3.4.0.12
- pillow 5.0.0

A `requirements.txt` with the above python dependencies can be found
in the root folder of the repository. You may want to install them by
running:

```bash
pip install -r requirements.txt
```

## Material

### Presentations

In addition to the source code we will keep updated this section with
the links to weekly presentations, which summarize the work done and
results:

- Week 1: Introduction to the datasets, evaluation metrics and optical
  flow analysis.
- Week 2: Background estimation and Stauffer & Grimson's approach.
- Week 3: Foreground segmentation, filtering and shadow removal.
- Week 4: Optical flow and video stabilization.
- Week 5: Region tracking and Kalman filter.
- Week 6: Pipeline assembly and performance evaluation.

### Week 1 Code

To sequentially run the tasks of this week, `cd` to the root folder of
the repository and execute:

```bash
python run.py week1
```

## Datasets

During the development process we will use the following datasets to
evaluate the performance of our system:

- [ChangeDetection.NET (CDNET)](http://changedetection.net/)
- [KITTI dataset for Optical Flow 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow)

You may need to copy the datasets in the `datasets` folder to run some
of the weekly tasks. When needed, it will be reported in the weekly
code sections above.


## Authors

- [Ignasi Mas](mailto:ignasi.masm@e-campus.uab.cat)
- [Hugo Prol](mailto:hugo.prol@e-campus.uab.cat)
- [Jordi Puyoles](mailto:jordi.puyoles@e-campus.uab.cat)
