import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import argparse

def main(args):

    # Load face detector model
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    detector = cv2.dnn.readNetFromCaffe(configFile, modelFile)


    # imgs = ["rup6bb","rydie3"]
    # for img_id in imgs:
    # Load data to blur
    with open(args.filepath, 'r') as f:
        data = json.load(f)
    for thread in data['threads']:
        img_id = thread['submission_id']
        subreddit = thread['subreddit']
    
        img_f = f"datasets/redcaps/batch0_pilot_300/images/{img_id}.jpg"

        image = cv2.imread(img_f)
        base_img = image.copy()

        original_size = image.shape
        target_size = (300, 300)
        # print("original image size: ", original_size)
        image = cv2.resize(image, target_size)
        aspect_ratio_x = (original_size[1] / target_size[1])
        aspect_ratio_y = (original_size[0] / target_size[0])
        imageBlob = cv2.dnn.blobFromImage(image = image)

        detector.setInput(imageBlob)
        detections = detector.forward()
        detections[0][0].shape

        detections_df = pd.DataFrame(detections[0][0], columns = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"])
        detections_df = detections_df[detections_df['is_face'] == 1] #0: background, 1: face
        detections_df = detections_df[detections_df['confidence'] >= args.confidence]
        # print(detections_df.head())
        
        for i, instance in detections_df.iterrows():
            
            print(f"Found face in img id {img_id}. Blurring and saving...")
            # print(instance)
            
            confidence_score = str(round(100*instance["confidence"], 2))+" %"
            
            left = int(instance["left"] * 300)
            bottom = int(instance["bottom"] * 300)
            right = int(instance["right"] * 300)
            top = int(instance["top"] * 300)
            
            detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)]
            
            # Add ellipse blurring mask
            ellipse_center = (int((int(left*aspect_ratio_x)+int(right*aspect_ratio_x))/2), int((int(bottom*aspect_ratio_y)+int(top*aspect_ratio_y))/2))
            ellipse_axesLength = (int(abs((int(right*aspect_ratio_x)-int(left*aspect_ratio_x))/2)), int(abs((int(top*aspect_ratio_y)-int(bottom*aspect_ratio_y))/2)))

            mask_img = np.zeros(base_img.shape, dtype='uint8')
            cv2.ellipse(mask_img,center=ellipse_center, axes=ellipse_axesLength,angle=0, startAngle=0, endAngle=360, color=(255,155,255), thickness=-1)
            base_image_blurred = cv2.medianBlur(base_img, 99)
            base_img = np.where(mask_img > 0, base_image_blurred, base_img)
     
            new_img_f = f"datasets/redcaps/batch0_pilot_300/images/{img_id}_blurred.jpg"
            cv2.imwrite(new_img_f, base_img)
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filepath', required=True, type=str)
    parser.add_argument('-c', '--confidence', type=float, default=0.5)
    args = parser.parse_args()
    main(args)