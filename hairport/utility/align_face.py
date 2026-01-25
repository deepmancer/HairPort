"""
FFHQ-style face alignment using MediaPipe.

This module implements the FFHQ face alignment algorithm using MediaPipe for
landmark detection instead of dlib. The alignment is fully consistent with
the original FFHQ dataset preprocessing method.

Reference:
    - FFHQ Dataset: https://github.com/NVlabs/ffhq-dataset
    - Original alignment code by lzhbrian: https://lzhbrian.me
"""

import sys
from pathlib import Path
from typing import Union, Optional, List, Tuple

import numpy as np
import scipy.ndimage
from PIL import Image

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.facial_landmark_detector import FacialLandmarkDetector


def ffhq_align_single_face(
    image: Image.Image,
    landmarks: np.ndarray,
    output_size: int = 1024,
    transform_size: int = 4096,
    enable_padding: bool = True
) -> Image.Image:
    """Apply FFHQ-style alignment to a single face.
    
    This implements the exact alignment algorithm used for the FFHQ dataset.
    
    Args:
        image: Input PIL Image.
        landmarks: Dlib 68-point landmarks, shape (68, 2).
        output_size: Output image size (default 1024x1024).
        transform_size: Intermediate transform size for high quality.
        enable_padding: Whether to enable boundary padding with reflection.
    
    Returns:
        Aligned face as PIL Image.
    """
    lm = landmarks
    
    # Extract landmark groups (Dlib 68-point format)
    lm_chin = lm[0:17]          # Face contour, left-right
    lm_eyebrow_left = lm[17:22]  # Left eyebrow, left-right
    lm_eyebrow_right = lm[22:27] # Right eyebrow, left-right
    lm_nose = lm[27:31]          # Nose bridge, top-down
    lm_nostrils = lm[31:36]      # Nostrils, top-down
    lm_eye_left = lm[36:42]      # Left eye, left-clockwise
    lm_eye_right = lm[42:48]     # Right eye, left-clockwise
    lm_mouth_outer = lm[48:60]   # Outer mouth, left-clockwise
    lm_mouth_inner = lm[60:68]   # Inner mouth, left-clockwise
    
    # Calculate auxiliary vectors
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    
    # Choose oriented crop rectangle
    # The alignment is based on the angle between eyes and mouth
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    
    # Define quadrilateral corners for crop
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2
    
    img = image.copy()
    
    # Shrink if needed
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(img.size[0]) / shrink)),
            int(np.rint(float(img.size[1]) / shrink))
        )
        img = img.resize(rsize, Image.LANCZOS)
        quad /= shrink
        qsize /= shrink
    
    # Crop
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1])))
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1])
    )
    
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
    
    # Pad
    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1])))
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0)
    )
    
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(
            np.float32(img),
            ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
            'reflect'
        )
        h, w, _ = img.shape
        y_grid, x_grid, _ = np.ogrid[:h, :w, :1]
        
        # Create smooth transition mask for padding boundary
        mask = np.maximum(
            1.0 - np.minimum(
                np.float32(x_grid) / pad[0] if pad[0] > 0 else np.inf,
                np.float32(w - 1 - x_grid) / pad[2] if pad[2] > 0 else np.inf
            ),
            1.0 - np.minimum(
                np.float32(y_grid) / pad[1] if pad[1] > 0 else np.inf,
                np.float32(h - 1 - y_grid) / pad[3] if pad[3] > 0 else np.inf
            )
        )
        
        # Apply Gaussian blur and median blend at boundaries
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    
    # Apply perspective transform
    img = img.transform(
        (transform_size, transform_size),
        Image.QUAD,
        (quad + 0.5).flatten(),
        Image.BILINEAR
    )
    
    # Resize to final output size
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.LANCZOS)
    
    return img


def align_face_from_image(
    image: Union[str, Path, np.ndarray, Image.Image],
    detector: Optional[FacialLandmarkDetector] = None,
    output_size: int = 1024,
) -> Optional[Image.Image]:
    """Align face in an image using FFHQ-style alignment.
    
    Args:
        image: Input image (file path, numpy array, or PIL Image).
        detector: FacialLandmarkDetector instance (will create one if None).
        output_size: Output size for aligned faces.
    
    Returns:
        Aligned face as PIL Image, or None if no face is detected.
    """
    # Create detector if not provided
    if detector is None:
        detector = FacialLandmarkDetector(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    
    # Load image
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert('RGB')
    else:
        img = image.convert('RGB')
    
    # Detect landmarks using FacialLandmarkDetector
    result = detector.get_lmk_full(img)
    
    if result is None:
        return None
    
    # Get Dlib 68-point landmarks
    landmarks = result['ldm68']
    
    return ffhq_align_single_face(img, landmarks, output_size=output_size)


def align_face(
    data_dir: Union[str, Path],
    output_size: int = 1024,
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp', '.bmp'),
    skip_existing: bool = True,
    verbose: bool = True
) -> dict:
    """Perform FFHQ-style face alignment on images in a directory.
    
    This function processes all images in {data_dir}/image/, performs face
    alignment using MediaPipe for landmark detection, and saves the aligned
    results to {data_dir}/aligned_image/{image_name}.jpg.
    
    The alignment algorithm is fully consistent with the original FFHQ dataset
    preprocessing method.
    
    Args:
        data_dir: Base directory containing an 'image' subdirectory.
        output_size: Output size for aligned faces (default 1024x1024).
        image_extensions: Tuple of valid image file extensions.
        skip_existing: If True, skip images that have already been aligned.
        verbose: If True, print progress information.
    
    Returns:
        Dictionary with statistics:
            - 'processed': Number of images successfully aligned
            - 'skipped': Number of images skipped (already aligned)
            - 'failed': Number of images where face detection failed
            - 'failed_files': List of filenames where processing failed
    
    Example:
        >>> stats = align_face('/path/to/data')
        >>> print(f"Processed: {stats['processed']}, Failed: {stats['failed']}")
    """
    data_dir = Path(data_dir)
    input_dir = data_dir / 'image'
    output_dir = data_dir / 'aligned_image'
    
    # Validate input directory
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    image_files = sorted(set(image_files))
    
    if verbose:
        print(f"Found {len(image_files)} images in {input_dir}")
    
    # Initialize detector
    detector = FacialLandmarkDetector(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    
    # Statistics
    stats = {
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'failed_files': []
    }
    
    # Process each image
    for img_path in image_files:
        # Output filename (always .jpg)
        output_filename = img_path.stem + '.jpg'
        output_path = output_dir / output_filename
        
        # Skip if already processed
        if skip_existing and output_path.exists():
            if verbose:
                print(f"Skipping (already exists): {img_path.name}")
            stats['skipped'] += 1
            continue
        
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Detect landmarks using FacialLandmarkDetector
            result = detector.get_lmk_full(img)
            
            if result is None:
                if verbose:
                    print(f"Warning: No face detected in {img_path.name}")
                stats['failed'] += 1
                stats['failed_files'].append(str(img_path.name))
                continue
            
            # Get Dlib 68-point landmarks
            landmarks = result['ldm68']
            
            # Align face
            aligned_img = ffhq_align_single_face(
                img, landmarks,
                output_size=output_size
            )
            
            # Save aligned image
            aligned_img.save(output_path, 'JPEG', quality=95)
            
            if verbose:
                print(f"Aligned: {img_path.name} -> {output_filename}")
            stats['processed'] += 1
            
        except Exception as e:
            if verbose:
                print(f"Error processing {img_path.name}: {e}")
            stats['failed'] += 1
            stats['failed_files'].append(str(img_path.name))
    
    if verbose:
        print(f"\nAlignment complete:")
        print(f"  Processed: {stats['processed']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"  Failed: {stats['failed']}")
    
    return stats


def align_face_batch(
    data_dirs: List[Union[str, Path]],
    output_size: int = 1024,
    skip_existing: bool = True,
    verbose: bool = True
) -> dict:
    """Perform FFHQ-style face alignment on multiple directories.
    
    Args:
        data_dirs: List of data directories to process.
        output_size: Output size for aligned faces.
        skip_existing: If True, skip images that have already been aligned.
        verbose: If True, print progress information.
    
    Returns:
        Aggregated statistics dictionary.
    """
    total_stats = {
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'failed_files': []
    }
    
    for data_dir in data_dirs:
        if verbose:
            print(f"\nProcessing: {data_dir}")
        
        try:
            stats = align_face(
                data_dir,
                output_size=output_size,
                skip_existing=skip_existing,
                verbose=verbose
            )
            
            total_stats['processed'] += stats['processed']
            total_stats['skipped'] += stats['skipped']
            total_stats['failed'] += stats['failed']
            total_stats['failed_files'].extend(
                [f"{data_dir}/{f}" for f in stats['failed_files']]
            )
        except Exception as e:
            if verbose:
                print(f"Error processing directory {data_dir}: {e}")
    
    return total_stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='FFHQ-style face alignment using MediaPipe'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/workspace/outputs',
        help='Base directory containing image/ subdirectory'
    )
    parser.add_argument(
        '--output-size',
        type=int,
        default=1024,
        help='Output image size (default: 1024)'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Do not skip already aligned images'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    stats = align_face(
        args.data_dir,
        output_size=args.output_size,
        skip_existing=not args.no_skip,
        verbose=not args.quiet
    )
