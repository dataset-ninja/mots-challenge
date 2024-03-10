**MOTSChallenge Dataset** is a dataset for instance segmentation, semantic segmentation, and object detection tasks. It is used in the automotive industry. 

The dataset consists of 2862 images with 55616 labeled objects belonging to 3 different classes including *pedestrian*, *ignore region*, and *car*.

Images in the MOTSChallenge dataset have pixel-level instance segmentation annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation (only one mask for every class) or object detection (bounding boxes for every object) tasks. All images are labeled (i.e. with annotations). There is 1 split in the dataset: *train* (2862 images). Additionally, every image marked with its ***sequence*** tag. Every label contains information about its ***class id***. Explore it in Supervisely labelling tool. The dataset was released in 2019 by the <span style="font-weight: 600; color: grey; border-bottom: 1px dashed #d3d3d3;">RWTH Aachen University, Germany</span> and <span style="font-weight: 600; color: grey; border-bottom: 1px dashed #d3d3d3;">MPI for Intelligent Systems and University of Tubingen, Germany</span>.

<img src="https://github.com/dataset-ninja/mots-challenge/raw/main/visualizations/poster.png">
