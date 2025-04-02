import numpy as np
import albumentations as A
from datasets import load_dataset, DatasetDict
from functools import partial
from transformers import AutoImageProcessor


def validate_bbox(bbox):
    x_min, y_min, x_max, y_max = bbox
    if x_min >= x_max or y_min >= y_max:
        return False
    return True


def build_dataset() -> DatasetDict:
    raw_datasets = load_dataset("cppe-5")

    if "validation" not in raw_datasets:
        split = raw_datasets["train"].train_test_split(0.15, seed=1337)
        raw_datasets["train"] = split["train"]
        raw_datasets["validation"] = split["test"]

    raw_datasets["test"] = load_dataset("cppe-5", split="test")

    return raw_datasets


train_augment_and_transform = A.Compose(
    [
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
)

validation_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
)


def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def augment_and_transform_batch(examples, transform, image_processor, return_pixel_mask=False):
    images = []
    annotations = []

    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))

        valid_bboxes = []
        valid_categories = []
        valid_areas = []

        for bbox, category, area in zip(objects["bbox"], objects["category"], objects["area"]):
            if validate_bbox(bbox):
                valid_bboxes.append(bbox)
                valid_categories.append(category)
                valid_areas.append(area)
            else:
                print(f"Invalid bbox for image_id {image_id}: {bbox}, skipping.")

        if not valid_bboxes:  # Skip images with no valid bounding boxes
            print(f"No valid bounding boxes for image_id {image_id}. Skipping.")
            continue

        output = transform(image=image, bboxes=valid_bboxes, category=valid_categories)
        images.append(output["image"])

        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], valid_areas, output["bboxes"]
        )
        annotations.append(formatted_annotations)

    result = image_processor(images=images, annotations=annotations, return_tensors="pt")

    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


def add_preprocessing(dataset, processor) -> DatasetDict:
    train_transform_batch = partial(
        augment_and_transform_batch,
        transform=train_augment_and_transform,
        image_processor=processor,
    )

    validation_transform_batch = partial(
        augment_and_transform_batch,
        transform=validation_transform,
        image_processor=processor,
    )

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)
    dataset["test"] = dataset["test"].with_transform(validation_transform_batch)

    return dataset


raw_datasets = build_dataset()
processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
processed_datasets = add_preprocessing(raw_datasets, processor)
