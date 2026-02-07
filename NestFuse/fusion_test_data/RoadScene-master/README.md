# RoadScene：a new dataset of aligned infrared and visible images

This datset has 221 aligned Vis and IR image pairs containing rich scenes such as roads, vehicles, pedestrians and so on. These images are highly representative scenes from the [FLIR video](https://www.flir.com/oem/adas/adas-dataset-form/). We preprocess the background thermal noise in the original IR images, accurately align the Vis and IR image pairs, and cut out the exact registration regions to form this dataset.<br>

Some typical examples are shown below(the left column are Vis images and the right column are IR images):<br>


<img src="https://github.com/hanna-xu/road-scene-infrared-visible-images/blob/master/crop_HR_visible/FLIR_05164.jpg" width="400" height="220"/>    <img src="https://github.com/hanna-xu/road-scene-infrared-visible-images/blob/master/cropinfrared/FLIR_05164.jpg" width="400" height="220"/>

<img src="https://github.com/hanna-xu/road-scene-infrared-visible-images/blob/master/crop_HR_visible/FLIR_06832.jpg" width="400" height="260"/>    <img src="https://github.com/hanna-xu/road-scene-infrared-visible-images/blob/master/cropinfrared/FLIR_06832.jpg" width="400" height="260"/>

<img src="https://github.com/hanna-xu/road-scene-infrared-visible-images/blob/master/crop_HR_visible/FLIR_07202.jpg" width="400" height="240"/>    <img src="https://github.com/hanna-xu/road-scene-infrared-visible-images/blob/master/cropinfrared/FLIR_07202.jpg" width="400" height="240"/>

<img src="https://github.com/hanna-xu/road-scene-infrared-visible-images/blob/master/crop_HR_visible/FLIR_07206.jpg" width="400" height="230"/>    <img src="https://github.com/hanna-xu/road-scene-infrared-visible-images/blob/master/cropinfrared/FLIR_07206.jpg" width="400" height="230"/>

If you use the dataset in your research, please cite the following paper:
```
@inproceedings{xu2020aaai,
title={FusionDN: A Unified Densely Connected Network for Image Fusion},
author={Xu, Han and Ma, Jiayi and Le, Zhuliang and Jiang, Junjun and Guo, Xiaojie},
booktitle={proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence},
year={2020}
}
```
