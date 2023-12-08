import os
import json
import cv2
import numpy as np
import pandas as pd
import pytest
from moiiai_test import label


@pytest.fixture
def sample_data(tmpdir):
    temp_dir = tmpdir.mkdir("temp_test_data")
    img_path = os.path.join(temp_dir, "sample_image.jpg")
    cv2.imwrite(img_path, np.zeros((100, 100, 3), dtype=np.uint8))
    json_path = os.path.join(temp_dir, "sample_data.json")
    sample_json_data = {
        "images": [{"id": 1, "file_name": "sample_image.jpg"}],
        "annotations": [{"image_id": 1, "bbox": [10, 10, 50, 50]}],
    }
    with open(json_path, "w") as json_file:
        json.dump(sample_json_data, json_file)
    return img_path, json_path, temp_dir


def test_labeling(sample_data):
    img_path, json_path, output_folder_path = sample_data
    label(img_path, json_path, output_folder_path)
    annotated_img_path = os.path.join(output_folder_path, "sample_image.jpg")
    assert os.path.exists(annotated_img_path)
    annotated_img = cv2.imread(annotated_img_path)
    assert annotated_img is not None
    assert np.array_equal(annotated_img[10:60, 10:60], [255, 1, 1])


if __name__ == "__main__":
    pytest.main([__file__])
