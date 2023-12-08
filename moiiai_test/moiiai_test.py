import json
import os

import pandas as pd
from cv2 import imread, imwrite, rectangle


def label(img_folder_path, json_path, output_folder_path):
    # Read the images
    images_files = os.listdir(os.path.join(".", img_folder_path))

    # Read the JSON file
    with open(os.path.join(".", json_path), "r") as f:
        data = json.load(f)

    # Extract image and annotation data from JSON
    image_data = data.get("images")
    annotation_data = data.get("annotations")

    # DataFrames from image and annotation data
    image_df = pd.DataFrame(image_data, columns=["id", "file_name"])
    annotation_df = pd.DataFrame(annotation_data, columns=["image_id", "bbox"])

    # Joining the two DataFrames on Image ID
    merged_df = annotation_df.merge(
        image_df, left_on="image_id", right_on="id", how="left"
    )

    # Creating a feature named 'modified_filename' derived from the column file_name
    merged_df["modified_filename"] = merged_df["file_name"].apply(
        lambda x: x.split("/")[-1]
    )

    # Plotting the Coordinates from DataFrame to the image and storing it in a directory
    grouped_df = merged_df.groupby("modified_filename")

    for img in images_files:
        grouped_df2 = grouped_df.get_group(img)
        grouped_df2 = grouped_df2.reset_index(drop=True)
        image_path = os.path.join(img_folder_path, img)
        images = imread(image_path)

        for i in range(len(grouped_df2)):
            xmin, ymin, width, height = grouped_df2["bbox"][i]
            rectangle(
                images, (xmin, ymin), (xmin + width, ymin + height), (255, 1, 1), 2
            )

        # Save the labeled image
        annotated_img_path = os.path.join(output_folder_path, img)
        imwrite(annotated_img_path, images)


# Set the folder path, JSON file path, and output JSON file path
folder_path = "images"
json_path = "./result.json"
output_folder_path = "BBOX"

label(folder_path, json_path, output_folder_path)
