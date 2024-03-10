The authors create dense pixel-level annotations for **MOTSChallenge Dataset** using a semi-automatic annotation procedure. They further annotated 4 of 7 sequences of the [MOTChallenge 2017](https://motchallenge.net/data/MOT17/) training dataset. MOTSChallenge focuses on *pedestrian* in crowded scenes and is very challenging due to many occlusion cases, for which a pixel-wise description is especially beneficial.

Note, similar **MOTSChallenge Dataset** dataset is also available on the [DatasetNinja.com](https://datasetninja.com/):

- [KITTI MOTS: Multi Object Tracking and Segmentation Dataset](https://datasetninja.com/kitti-mots)

## Motivation

In recent years, significant strides have been made in the computer vision field, particularly with the advent of deep learning techniques. These advancements have led to remarkable performance improvements in tasks such as object detection and image segmentation. However, tracking multiple objects remains a challenging endeavor, especially as bounding box-level tracking performance appears to have reached a plateau in recent evaluations. To achieve further enhancements, transitioning to pixel-level tracking is deemed necessary. To address this challenge, the authors advocate for a holistic approach that considers detection, segmentation, and tracking as interconnected problems. Currently, datasets suitable for training and evaluating instance segmentation models typically lack annotations on video data or object identities across different frames. Conversely, common datasets for multiobject tracking primarily offer bounding box annotations, which may prove inadequate in scenarios involving partial occlusion, where bounding boxes may contain more information from surrounding objects than the target itself.

<img src="https://github.com/dataset-ninja/kitti-mots/assets/120389559/d8f9aac1-6a7d-4da9-a13a-59e3fc23c3f7" alt="image" width="600">

<span style="font-size: smaller; font-style: italic;">Segmentations vs. Bounding Boxes. When objects pass each other, large parts of an object’s bounding box may belong to another instance, while per-pixel segmentation masks locate objects precisely. The shown annotations are crops from the KITTI MOTS dataset.</span>

In such scenarios, employing pixel-wise segmentation for object delineation offers a more faithful depiction of the scene, potentially enhancing subsequent processing stages. Unlike bounding boxes, segmentation masks provide a clearly defined ground truth, whereas bounding boxes may vary widely in their fit to an object. Moreover, tracks with overlapping bounding boxes introduce ambiguities during evaluation, often necessitating heuristic matching methods for resolution. In contrast, tracking results derived from segmentation-based approaches are inherently non-overlapping, facilitating a direct comparison with ground truth annotations. The authors have put forth a proposition to expand the conventional multi-object tracking task to include instance segmentation tracking. As far as current knowledge goes, there are no existing datasets tailored specifically for this task. While numerous methods for bounding box tracking are documented in the literature, achieving success in MOTS necessitates the integration of temporal and mask cues.

## Dataset description

Creating pixel masks for every object in a video frame is an immensely time-consuming task, leading to limited availability of such data. Currently, there are no existing datasets tailored specifically for the MOTS (Multi-Object Tracking and Segmentation) task. However, some datasets do provide annotations at the bounding box level for MOT, lacking segmentation masks necessary for MOTS. To address this gap, the authors enhanced two MOT datasets by adding segmentation masks to the bounding boxes. In total, they annotated 65,213 segmentation masks, rendering the datasets suitable for training and evaluating modern learning-based techniques. To manage the annotation workload effectively, the authors devised a semi-automatic method to extend bounding box annotations with segmentation masks. This method involves an iterative process of generating and refining masks until achieving pixel-level accuracy for all annotation masks.

To convert bounding boxes into segmentation masks, the authors employed a fully convolutional refinement network utilizing DeepLabv3+. This network operates by taking a crop of the input image specified by the bounding box, alongside a small context region, and an additional input channel encoding the bounding box as a mask. Leveraging these inputs, the refinement network generates a segmentation mask corresponding to the provided box. Initially, the refinement network undergoes pre-training on [COCO](https://cocodataset.org/#home) and [Mapillary](https://www.mapillary.com/datasets) datasets, followed by further training on manually crafted segmentation masks specific to the target dataset. Initially, the authors annotate two segmentation masks per object in the dataset, delineated as polygons. The refinement network undergoes initial training using all manually created masks, followed by individual fine-tuning for each object. These fine-tuned iterations of the network are then applied to generate segmentation masks for all bounding boxes associated with the corresponding object in the dataset. This approach allows the network to adapt to the specific appearance and context of each object. Although the use of two manually annotated segmentation masks per object for fine-tuning generally yields satisfactory masks for the object's appearances in other frames, minor errors may persist. Therefore, the authors manually correct some of the imperfectly generated masks and iterate the training procedure accordingly. Additionally, their annotators rectify imprecise or erroneous bounding box annotations present in the original MOT datasets.

<img src="https://github.com/dataset-ninja/kitti-mots/assets/120389559/0f7ae9d6-fe53-4e01-b6db-79e3e9c209cb" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Sample Images of the authors Annotations. KITTI MOTS (top) and MOTSChallenge (bottom).</span>

The authors further annotated 4 of 7 sequences of the [MOTChallenge 2017](https://motchallenge.net/data/MOT17/) training dataset and obtained the MOTSChallenge dataset. MOTSChallenge focuses on pedestrians in crowded scenes and is very challenging due to many occlusion cases, for which a pixel-wise
description is especially beneficial.

|                | KITTI   | MOTS  | MOTSChallenge |
|----------------|---------|-------|---------------|
|                | train   | val   | train         |
| Sequences      | 12      | 9     | 4             |
| Frames         | 5,027   | 2,981 | 2,862         |
| Tracks Pedestrian | 99   | 68    | 228           |
| Masks Pedestrian Total | 8,073 | 3,347 | 26,894  |
| Manually annotated    | 1,312 | 647   | 3,930     |
| Tracks Car     | 431     | 151   | -             |
| Masks Car Total | 18,831 | 8,068 | -             |
| Manually annotated    | 1,509 | 593   | -          |

<span style="font-size: smaller; font-style: italic;">Statistics of the Introduced KITTI MOTS and MOTSChallenge Datasets.</span>

