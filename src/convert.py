import os
import shutil
from collections import defaultdict

import numpy as np
import pycocotools.mask as mask_util
import supervisely as sly
from dataset_tools.convert import unpack_if_archive
from supervisely.io.fs import (
    file_exists,
    get_file_ext,
    get_file_name,
    get_file_name_with_ext,
    get_file_size,
)
from tqdm import tqdm

import src.settings as s


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    # Possible structure for bbox case. Feel free to modify as you needs.

    train_images_path = "/home/alex/DATASETS/TODO/MOTSChallenge/MOTSChallenge/train/images"
    ins_path = "/home/alex/DATASETS/TODO/MOTSChallenge/MOTSChallenge/train/instances_txt"
    batch_size = 30

    ds_name_to_images = {"train": train_images_path}

    def convert_rle_mask_to_polygon(rle_mask_data):
        if type(rle_mask_data["counts"]) is str:
            rle_mask_data["counts"] = bytes(rle_mask_data["counts"], encoding="utf-8")
            mask = mask_util.decode(rle_mask_data)
        else:
            rle_obj = mask_util.frPyObjects(
                rle_mask_data,
                rle_mask_data["size"][0],
                rle_mask_data["size"][1],
            )
            mask = mask_util.decode(rle_obj)
        mask = np.array(mask, dtype=bool)
        return sly.Bitmap(mask).to_contours()

    def create_ann(image_path):
        labels = []

        image_np = sly.imaging.image.read(image_path)[:, :, 0]
        img_height = image_np.shape[0]
        img_wight = image_np.shape[1]

        ann_data = im_name_to_data[get_file_name_with_ext(image_path)]
        for curr_ann_data in ann_data:
            l_tags = []
            obj_class = idx_to_class.get(curr_ann_data[1])
            if obj_class is not None:
                identity_value = int(curr_ann_data[0])
                if identity_value != 10000:
                    tag = sly.Tag(identity_meta, value=identity_value)
                    l_tags.append(tag)
                    instance_value = int(curr_ann_data[0][1:])
                    instance = sly.Tag(instance_meta, value=instance_value)
                    l_tags.append(instance)

                rle_mask_data = {
                    "size": [image_np.shape[0], image_np.shape[1]],
                    "counts": curr_ann_data[4],
                }
                polygons = convert_rle_mask_to_polygon(rle_mask_data)
                for polygon in polygons:
                    if polygon.area > 35:
                        label = sly.Label(polygon, obj_class, tags=l_tags)
                        labels.append(label)
                        label_r = sly.Label(polygon.to_bbox(), obj_class, tags=l_tags)
                        labels.append(label_r)

        return sly.Annotation(img_size=(img_height, img_wight), labels=labels, img_tags=[seq])

    pedestrian = sly.ObjClass("pedestrian", sly.AnyGeometry)
    ignore = sly.ObjClass("ignore region", sly.AnyGeometry)

    idx_to_class = {"2": pedestrian, "10": ignore}

    identity_meta = sly.TagMeta("object id", sly.TagValueType.ANY_NUMBER)
    instance_meta = sly.TagMeta("instance id", sly.TagValueType.ANY_NUMBER)
    seq_meta = sly.TagMeta("sequence", sly.TagValueType.ANY_STRING)

    project = api.project.create(workspace_id, project_name, change_name_if_conflict=True)
    meta = sly.ProjectMeta(
        obj_classes=[pedestrian, ignore],
        tag_metas=[identity_meta, seq_meta, instance_meta],
    )
    api.project.update_meta(project.id, meta.to_json())

    for ds_name, images_pathes in ds_name_to_images.items():

        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)

        for subfolder in os.listdir(images_pathes):

            seq = sly.Tag(seq_meta, value=subfolder)

            images_path = os.path.join(images_pathes, subfolder)
            ann_path = os.path.join(ins_path, subfolder + ".txt")

            im_name_to_data = defaultdict(list)
            with open(ann_path) as f:
                content = f.read().split("\n")
                for row in content:
                    if len(row) > 0:
                        row_data = row.split(" ")
                        im_name_to_data[row_data[0].zfill(6) + ".jpg"].append(row_data[1:])

            images_names = [
                im_name for im_name in os.listdir(images_path) if get_file_ext(im_name) == ".jpg"
            ]

            progress = sly.Progress("Create dataset {}".format(ds_name), len(images_names))

            for images_names_batch in sly.batched(images_names, batch_size=batch_size):
                im_names_batch = []
                img_pathes_batch = []
                for image_name in images_names_batch:
                    img_pathes_batch.append(os.path.join(images_path, image_name))
                    im_names_batch.append(subfolder + "_" + image_name)

                img_infos = api.image.upload_paths(dataset.id, im_names_batch, img_pathes_batch)
                img_ids = [im_info.id for im_info in img_infos]

                anns = [create_ann(image_path) for image_path in img_pathes_batch]
                api.annotation.upload_anns(img_ids, anns)

                progress.iters_done_report(len(images_names_batch))

    return project
