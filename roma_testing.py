from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import numpy as np
from roma.utils.utils import tensor_to_pil
import supervision as sv
from roma import roma_outdoor, tiny_roma_v1_outdoor
import cv2 
import os
from scipy.ndimage import zoom
from tqdm import tqdm
from parse_annotations import load_annotations, parse_annotations, filter_images_with_classes
from signshape import determine_shape

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_middle_points(image):
    height, width = image.shape[:2]
    
    middle_top = (0, width // 2)
    middle_bottom = (height - 1, width // 2)
    middle_left = (height // 2, 0)
    middle_right = (height // 2, width - 1)
    
    return np.array([middle_top, middle_bottom, middle_left, middle_right])

def transform_points(points, H, scale_x, scale_y):

    points_homogeneous = np.hstack((points[:, ::-1], np.ones((points.shape[0], 1))))  # (N, 3) with (x, y, 1)

    transformed_points = H @ points_homogeneous.T


    transformed_points_cartesian = np.vstack((transformed_points[1, :], transformed_points[0, :])).T.astype(int)

    transformed_points_scaled = np.vstack((transformed_points_cartesian[:, 0] * scale_y, 
                                           transformed_points_cartesian[:, 1] * scale_x)).T.astype(int)
    return transformed_points_scaled


def euclidean_distance(point1, point2):
    return np.abs(point1 - point2)

def calculate_bounding_box(polygon):
    """Calculate the bounding box of a polygon."""
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)

def create_segmentation_mask(img_size, polygon):
    """Create a segmentation mask for a given polygon."""
    mask = Image.new('L', img_size, 0)
    draw = ImageDraw.Draw(mask)
    polygon_pixels = [(x * img_size[0], y * img_size[1]) for x, y in polygon]
    draw.polygon(polygon_pixels, outline=1, fill=1)
    return np.array(mask)

def detect_pole(image, bounding_boxes):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    poles = []
    for box in bounding_boxes:
        x, y, w, h = box
        roi = edges[y + h:y + h + 10, x:x + w]  # Adjust the height range as needed
        
        # Morphological operations to enhance pole structure
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the region of interest
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:  # Filter small contours
                x_p, y_p, w_p, h_p = cv2.boundingRect(cnt)
                pole_rect = (x + x_p, y + h + y_p, w_p, h_p)
                poles.append(pole_rect)
                
    return poles

def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham's Line Algorithm to generate points between (x0, y0) and (x1, y1)
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

if __name__ == "__main__":

    
    
    ds = sv.DetectionDataset.from_yolo(
    images_directory_path="../data/outputs/images",
    annotations_directory_path="../data/outputs/labels",
    data_yaml_path="../data/outputs/data.yaml",
)
    
    index = list(ds.images.keys()).index('../data/outputs/images/iStock-1226736457-flood-uk-679x419.jpg')
    IMAGE_NAME = list(ds.images.keys())[index]

    image = ds.images[IMAGE_NAME]
    annotations = ds.annotations[IMAGE_NAME]

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    labels = [f"{ds.classes[class_id]}" for class_id in annotations.class_id]

    annotated_image = mask_annotator.annotate(image.copy(), detections=annotations)
    annotated_image = box_annotator.annotate(
        annotated_image, detections=annotations, labels=labels
    )

    sv.plot_image(image=annotated_image, size=(8, 8))
    
    
    

    WATER_CLASS_ID = 0  
    TRAFFIC_SIGN_CLASS_ID = 1  
    SIGN_HEIGHT = 600

    image_dir = '../data/outputs/images'
    annotation_dir = '../data/outputs/labels'

    #template_dir = "../data/templates/circlesigns"
    #im2_path = "../data/outputs/images/673131085073620992.jpg"
    #annotation_file = "../data/outputs/labels/673131085073620992.txt"

    selected_images = filter_images_with_classes(image_dir, annotation_dir, WATER_CLASS_ID, TRAFFIC_SIGN_CLASS_ID)
    roma_full = roma_outdoor(device=device, upsample_res=(864, 1152), amp_dtype=torch.float32)
    roma_tiny = tiny_roma_v1_outdoor(device=device)
    for image in tqdm(selected_images):

        image_base_name = os.path.splitext(os.path.basename(image))[0]
        annotation_file = os.path.join(annotation_dir, f"{image_base_name}.txt")
        im2 = Image.open(image)
        im2_np = np.array(im2)[:,:,0]
        im2_2_np = np.array(im2)
        img_width, img_height = im2.size
        #im2_np = np.asarray(crop)[:,:,0]
        #print(im2_np.shape)
        annotations = load_annotations(annotation_file)
        parsed_annotations = parse_annotations(annotations)
        num = 0
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        for class_id, polygon in parsed_annotations:
            if class_id == WATER_CLASS_ID:
                water_mask = create_segmentation_mask((img_width,img_height), polygon)
                mask = mask + water_mask
                #print(np.unique(mask))
        
        for class_id,polygon in parsed_annotations:
            if class_id == TRAFFIC_SIGN_CLASS_ID:
                num += 1 
                error = {}
                min_x, min_y, max_x, max_y = calculate_bounding_box(polygon)
                left, top = int(min_x * img_width), int(min_y * img_height)
                right, bottom = int(max_x * img_width), int(max_y * img_height)


                
                crop = im2.crop((left, top, right, bottom))
                crop_np = np.array(crop)[:,:,0]
                mask_sign = create_segmentation_mask((img_width, img_height), polygon)
                cropped_mask = mask_sign[top:bottom, left:right]
                sign_shape = determine_shape(cropped_mask)
                print(f"Detected shape: {sign_shape}")

                if sign_shape in ["Circle", "Triangle"]:
                    template_dir = {"Circle": "../data/templates/circlesigns",
                    "Triangle": "../data/templates/trianglesigns"}.get(sign_shape)
                else:
                    print(f"No matching template found for {image_base_name}")
                    continue

                dst_points = find_middle_points(crop_np)
                #print(crop_np.shape)
                #print(crop_np.shape[0] * crop_np.shape[1])
                if crop_np.shape[0] * crop_np.shape[1] <=450:
                     continue
            
                poles = detect_pole(im2_2_np, [(left,top,right-left,bottom-top)])
                for (x, y, w, h) in poles:
                    cv2.rectangle(im2_2_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
                 
                cv2.imshow('Poles', im2_2_np)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                """           
                im2 = Image.open(im2_path)
                img_width, img_height = im2.size
                annotations = load_annotations(annotation_file)
                parsed_annotations = parse_annotations(annotations)
                print(len(parsed_annotations))
                class_id, polygon = parsed_annotations[1]
                if class_id == 1:
                    min_x, min_y, max_x, max_y = calculate_bounding_box(polygon)

                    left, top = int(min_x * img_width), int(min_y * img_height)
                    right, bottom = int(max_x * img_width), int(max_y * img_height)

                    crop = im2.crop((left, top, right, bottom))
                    crop.show()
                    print(crop)
                """

    #im2_np = np.asarray(crop)[:,:,0]
    #print(im2_np.shape)
    #roma_model = tiny_roma_v1_outdoor(device=device)
    #error = {}
                for root, _, files in os.walk(template_dir):
                    for template_file in files:
                                    im1_path = os.path.join(root, template_file)
                                    #im1_path = "./data/templates/trianglesigns/602.jpg"
                                                        
                #save_path = "roma_warp.jpg"
                

                                    im1 = Image.open(im1_path)
                                    im1_np = np.asarray(im1)
                                    zoom_factors = (im1_np.shape[0] / crop_np.shape[0],  im1_np.shape[1] / crop_np.shape[1])
                                    resized_array = zoom(crop_np, zoom_factors)
                                    resized_array_normalized = ((resized_array - resized_array.min()) / (resized_array.max() - resized_array.min()) * 255).astype(np.uint8)
                                    resized_array_3channel = np.stack([resized_array_normalized]*3, axis=-1)
                                    crop_scale = Image.fromarray(resized_array_3channel)
            # Convert the single-channel array to a 3-channel array
                #resized_array_3channel = np.stack([resized_array_normalized]*3, axis=-1)
                #im2 = Image.fromarray(resized_array_3channel)
                #print(im2.size)

                

                                    #roma_model = roma_outdoor(device=device, amp_dtype=torch.float32)
                                    W_A, H_A = im1.size
                                    W_B, H_B = crop_scale.size

                                    # Match
                                    warp, certainty = roma_tiny.match(im1, crop_scale)
                                    # Sample matches for estimation
                                    #roma_model.visualize_warp(warp,certainty, im1, cropped_pil)
                                    matches, certainty = roma_tiny.sample(warp, certainty)
                                    kpts1, kpts2 = roma_tiny.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)  

                                    try:  
                                        F, inlier_mask = cv2.findHomography(
                                            kpts1.cpu().numpy(), kpts2.cpu().numpy(), ransacReprojThreshold=30, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
                                        )
                                    except Exception:
                                         continue

                                    #print(mask[mask>0].shape)
                                    #print(F)

                                    corners_image1 = np.array([[0, 0],
                                    [im1_np.shape[0] - 1, 0],
                                    [im1_np.shape[0] - 1, im1_np.shape[1]- 1],
                                    [0, im1_np.shape[1] - 1]], dtype=np.float32)
                                    
                                    corners_image2 = np.array([[0, 0],
                                    [resized_array_3channel.shape[0] - 1, 0],
                                    [resized_array_3channel.shape[0] - 1, resized_array_3channel.shape[1]- 1],
                                    [0, resized_array_3channel.shape[1] - 1]], dtype=np.float32)
                                    
                                    #corners_image1_homogeneous = np.hstack([corners_image1, np.ones((4, 1))])
                                    #transformed_corners_image1 = np.dot(F, corners_image1_homogeneous.T).T
                                    #transformed_corners_image1 = transformed_corners_image1[:, :2] / transformed_corners_image1[:, 2, np.newaxis]
                                    #distances = np.linalg.norm(transformed_corners_image1 - corners_image2, axis=1)
                                    #rms_distance = np.sqrt(np.mean(distances**2))
                                    src_points = find_middle_points(im1_np)

                                    height_src, width_src = im1_np.shape[:2]
                                    height_dst, width_dst = crop_np.shape[:2]
                                    scale_x = width_dst / width_src
                                    scale_y = height_dst / height_src
                                    src_points_t = transform_points(src_points,F,scale_x, scale_y)

                                    #print(src_points_t)
                                    #print(dst_points)

                                    distances = np.linalg.norm(src_points_t-dst_points)

                                    det = np.linalg.det(F)
                                    num_inliers = np.sum(inlier_mask)
                                    if num_inliers > 0:
                                        reprojected_pts = cv2.perspectiveTransform(kpts1.cpu().numpy().reshape(-1, 1, 2), F)
                                        reprojection_error = np.sqrt(np.sum((kpts2.cpu().numpy() - reprojected_pts.reshape(-1, 2)) ** 2, axis=1)).mean()
                                    else:
                                        reprojection_error = np.inf

                                    if F is not None:
                                        if (-0.1 < det < 0.1):
                                            score = np.inf

                                        else:
                                            score = distances
                                    #print(template_file, det, num_inliers, reprojection_error, distances,score)

                                    #warped_img = cv2.warpPerspective(im1_np, F, (resized_array_3channel.shape[1],resized_array_3channel.shape[0]))

                                    #cv2.imshow('Warped', warped_img)
                                    #cv2.waitKey(0)
                                    #cv2.destroyAllWindows()

                                    #output_img = cv2.addWeighted(resized_array_3channel, 0.5, warped_img, 0.5, 0)
                                    #cv2.imshow('Result', output_img)
                                    #cv2.waitKey(0)
                                    #cv2.destroyAllWindows()
                                    #score = num_inliers
                                    error[im1_path] = score
                                    min_key = min(error, key=error.get)
                                    #print(template_file, num_inliers, reprojection_error,distances)
                                    #print(min_key)
                        
                print(image, min_key)
                with open("signs.txt", "a") as file:
                     #Write a single string to the file
                    file.write(f"\n{image}, {num}, {min_key}\n")
                
                im1 = Image.open(str(min_key))
                #im1 = Image.open('../data/templates/circlesigns/616.jpg')
                im1_np = np.asarray(im1)[:,:,0]

                zoom_factors = (im1_np.shape[0] / crop_np.shape[0],  im1_np.shape[1] / crop_np.shape[1])
                resized_array = zoom(crop_np, zoom_factors)
                resized_array_normalized = ((resized_array - resized_array.min()) / (resized_array.max() - resized_array.min()) * 255).astype(np.uint8)
                resized_array_3channel = np.stack([resized_array_normalized]*3, axis=-1)
                crop_scale = Image.fromarray(resized_array_3channel)

                W_A, H_A = Image.open(im1_path).size
                W_B, H_B = crop.size

                warp, certainty = roma_full.match(im1, crop)
                matches, certainty = roma_full.sample(warp, certainty)
                kpts1, kpts2 = roma_full.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
                #print(kpts2.cpu().numpy())
                adjusted_keypoints = [(kp[0] + left, kp[1] + top) for kp in kpts2]
                adjusted_keypoints = np.array(adjusted_keypoints)
                #print(adjusted_keypoints)

                F, _ = cv2.findHomography(
                kpts1.cpu().numpy(), adjusted_keypoints, ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=100000
                )

                
                crop_np = np.asarray(crop)

                template_height, template_width = im1_np.shape[:2]
                bottom_middle_point = (template_width // 2, template_height)
                pixel_height = SIGN_HEIGHT / template_height
                line_length = 2200 / pixel_height
                line_endpoint = (bottom_middle_point[0], bottom_middle_point[1] + line_length)

                transformed_bottom_middle_point = cv2.perspectiveTransform(np.array([bottom_middle_point], dtype=np.float32).reshape(-1, 1, 2), F)
                transformed_line_endpoint = cv2.perspectiveTransform(np.array([line_endpoint], dtype=np.float32).reshape(-1, 1, 2), F)

                transformed_bottom_middle_point = tuple(map(int, transformed_bottom_middle_point[0][0]))
                transformed_line_endpoint = tuple(map(int, transformed_line_endpoint[0][0]))

                query_img_with_line = cv2.line(im2_np.astype(np.uint8), transformed_bottom_middle_point, transformed_line_endpoint, (0, 255, 0), 2)
                #print(transformed_bottom_middle_point,transformed_line_endpoint)

                warped_img = cv2.warpPerspective(im1_np, F, (im2_np.shape[1],im2_np.shape[0]))
                output_img = cv2.addWeighted(im2_np, 0.5, warped_img, 0.5, 0)
                cv2.imshow('Warped', warped_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                output_img = cv2.addWeighted(im2_np, 0.5, warped_img, 0.5, 0)
                cv2.imshow('Result', output_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                cv2.imshow('Query Image with Line', query_img_with_line)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                


                line_points = bresenham_line(transformed_bottom_middle_point[0], transformed_bottom_middle_point[1], transformed_line_endpoint[0], transformed_line_endpoint[1])
                line_array = np.zeros((im2_np.shape[0], im2_np.shape[1]), dtype=int)                
                for x, y in line_points:
                    if 0 <= x < line_array.shape[1] and 0 <= y < line_array.shape[0]: 
                        line_array[y, x] = 1

                # Find the lowest y-value of the line that corresponds with a non-zero value in mask
                lowest_y = None
                for x, y in line_points:
                    #print(x,y) 
                    #print(mask[y,x])
                    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]: 
                        if mask[y, x] != 0:
                            if lowest_y is None or y < lowest_y:
                                lowest_y = y
                    
                    else:
                        lowest_y = None

                #print("Lowest y-value where the line intersects a non-zero value in the other array:", lowest_y)
                above_count = 0
                below_count = 0
                if lowest_y != None:
                    for x, y in line_points:
                        if y < lowest_y:
                            above_count += 1
                        elif y > lowest_y:
                            below_count += 1
                    #print(above_count)
                    #print(below_count)
                    
                    if below_count > 0: 
                        ratio = below_count / (above_count + below_count)
                    else:
                        ratio = 0  
                else:
                    ratio = 0

                height = 2100 * ratio
                print(image, num, height)
                with open("depth2.txt", "a") as file:
                    #Write a single string to the file
                    file.write(f"\n{image}, {num}, {height}\n")

        
    
    