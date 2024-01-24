import json
import os
import cv2

def count_objects_in_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise cv2.error(f"Error reading image: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_objects = len(contours)
        return num_objects
    except cv2.error as e:
        print(f"Error processing image {image_path}: {e}")
        return 0 
    
image_directory = r"C:\Users\sabar\Documents\L_Project\output"
results = []
for filename in os.listdir(image_directory):
    if filename.endswith((".jpg", ".png")):
        image_path = os.path.join(image_directory, filename)
        frame_number_str = filename.split("_")[-1].split(".")[0]
        try:
            frame_number = int(frame_number_str)
        except ValueError:
            print(f"Error extracting frame number from image name: {filename}")
            continue  
        num_objects = count_objects_in_image(image_path)
        results.append(
            {
                "frame_number": frame_number,
                "image_name": filename,
                "num_objects": num_objects,
            }
        )
output_file = "output.json"
with open(output_file, "w") as json_file:
    json.dump(results, json_file, indent=2)

print(f"Results saved to {output_file}")
