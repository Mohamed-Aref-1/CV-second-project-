import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def is_valid_image(file_path):
    """
    Check if the file is a valid image file
    Args:
        file_path: Path to the image file
    Returns:
        bool: True if valid image file, False otherwise
    """
    # List of valid image extensions
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return False
        
    # Check file extension
    if not file_path.lower().endswith(valid_extensions):
        print(f"Error: File {file_path} is not a supported image format")
        print(f"Supported formats: {', '.join(valid_extensions)}")
        return False
        
    return True

def detect_and_compute_features(image, detector='SIFT'):
    """
    Detect and compute features from an image using the specified detector
    Args:
        image: Input image
        detector: Feature detector to use ('SIFT' or 'ORB')
    Returns:
        keypoints and descriptors
    """
    if detector == 'SIFT':
        feature_detector = cv2.SIFT_create()
    elif detector == 'ORB':
        feature_detector = cv2.ORB_create()
    else:
        raise ValueError(f"Unsupported detector: {detector}")
    
    keypoints, descriptors = feature_detector.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(des1, des2, detector='SIFT'):
    """
    Match features between two sets of descriptors
    Args:
        des1, des2: Feature descriptors to match
        detector: Feature detector used ('SIFT' or 'ORB')
    Returns:
        List of good matches
    """
    if detector == 'SIFT':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    else:  # ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good_matches = bf.match(des1, des2)
        good_matches = sorted(good_matches, key=lambda x: x.distance)
    
    return good_matches

def process_image(query_path, target_path, detector='SIFT'):
    """
    Process a single image pair to find matches
    """
    # Validate image files
    if not is_valid_image(query_path) or not is_valid_image(target_path):
        return
    
    # Read images with error handling
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    if query_img is None:
        print(f"Error: Could not read query image: {query_path}")
        print("Please ensure the file is not corrupted and has read permissions")
        return

    target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    if target_img is None:
        print(f"Error: Could not read target image: {target_path}")
        print("Please ensure the file is not corrupted and has read permissions")
        return

    print(f"Successfully loaded images:")
    print(f"Query image size: {query_img.shape}")
    print(f"Target image size: {target_img.shape}")

    # Detect and compute features
    try:
        kp1, des1 = detect_and_compute_features(query_img, detector)
        kp2, des2 = detect_and_compute_features(target_img, detector)
        
        if des1 is None or des2 is None:
            print("No features detected in one or both images")
            return
            
        print(f"Features detected:")
        print(f"Query image: {len(kp1)} keypoints")
        print(f"Target image: {len(kp2)} keypoints")
        
        # Match features
        good_matches = match_features(des1, des2, detector)
        print(f"Number of good matches found: {len(good_matches)}")
        
        # Draw matches
        result_img = cv2.drawMatches(query_img, kp1, target_img, kp2, good_matches[:30], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Display results
        plt.figure(figsize=(12,6))
        plt.imshow(result_img)
        plt.title(f'{detector} Matches')
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

def process_video(query_path, video_path, detector='SIFT'):
    """
    Process video to track object (Bonus part)
    """
    # Read query image
    query_img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    if query_img is None:
        raise ValueError("Could not read query image")
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get query image features
    kp1, des1 = detect_and_compute_features(query_img, detector)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect and match features
        kp2, des2 = detect_and_compute_features(frame_gray, detector)
        good_matches = match_features(des1, des2, detector)
        
        if len(good_matches) >= 4:
            # Get matched keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is not None:
                # Get query image dimensions
                h, w = query_img.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                
                # Transform corners
                dst = cv2.perspectiveTransform(pts, H)
                
                # Draw bounding box
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3)
        
        # Display result
        cv2.imshow('Object Tracking', frame)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Object detection using feature matching')
    parser.add_argument('--mode', choices=['image', 'video'], required=True,
                       help='Processing mode: image or video')
    parser.add_argument('--query', required=True, help='Path to query image')
    parser.add_argument('--target', required=True,
                       help='Path to target image or video file')
    parser.add_argument('--detector', choices=['SIFT', 'ORB'], default='SIFT',
                       help='Feature detector to use')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'image':
            process_image(args.query, args.target, args.detector)
        else:  # video mode
            process_video(args.query, args.target, args.detector)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
