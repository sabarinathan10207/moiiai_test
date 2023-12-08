import json
import os

import pandas as pd
from cv2 import imread, imwrite, rectangle

def label(img_folder_path, json_path, output_folder_path):
    images_files = os.listdir(img_folder_path)


    with open(json_path,  "r" ) as f:
        data = json.load(f)

    image_data = data.get("images")
    annotation_data = data.get("annotations")
    image_df = pd.DataFrame(image_data, columns=["id", "file_name"])
    annotation_df = pd.DataFrame(annotation_data, columns=["image_id", "bbox"])
    merged_df = annotation_df.merge(
        image_df, left_on="image_id", right_on="id", how="left"
    )
    merged_df["modified_filename"] = merged_df["file_name"].apply(
        lambda x: x.split("/")[-1]
    )
    grouped_df = merged_df.groupby("modified_filename")

    for img in images_files:
        grouped_df2 = grouped_df.get_group(img)
        grouped_df2 = grouped_df2.reset_index(drop=True)
        image_path = os.path.join(img_folder_path, img)
        images = imread(image_path)
        for i in range(len(grouped_df2)):
            xmin, ymin, width, height = grouped_df2["bbox"][i]
            rectangle(
                images,
                (xmin, ymin),
                (xmin + width, ymin + height),
                (255, 1, 1),
                2,
            )
        annotated_img_path = os.path.join(output_folder_path, img)
        imwrite(annotated_img_path, images)

folder_path = "images"
json_path = "result.json" 
output_folder_path = "BBOX"
label(folder_path, json_path, output_folder_path)