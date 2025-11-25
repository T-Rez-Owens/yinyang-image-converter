"""
Yin-Yang Image Converter - Configuration-Driven Version
No terminal input needed - edit config.ini to adjust all settings!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import matplotlib.patches as patches
from PIL import Image, ImageEnhance, ImageFilter
import os
from urllib.request import urlretrieve
import requests
from scipy import ndimage
import glob
import sys
from pathlib import Path
from config_loader import YinYangConfig

def get_next_output_filename(output_dir, prefix, method_suffix=""):
    """Get the next available output filename with optional method suffix"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    suffix = f"_{method_suffix}" if method_suffix else ""
    pattern = f'{prefix}*{suffix}.png'
    existing_files = list(output_path.glob(pattern))
    
    if not existing_files:
        return output_path / f'{prefix}1{suffix}.png'
    
    # Extract numbers and find the highest
    numbers = []
    for f in existing_files:
        try:
            # Extract number between prefix and the suffix
            base = f.stem.replace(prefix, '', 1)
            if suffix:
                num_part = base.replace(suffix, '')
            else:
                num_part = base
            num = int(num_part)
            numbers.append(num)
        except (ValueError, IndexError):
            continue
    
    next_num = max(numbers) + 1 if numbers else 1
    return output_path / f'{prefix}{next_num}{suffix}.png'

def find_available_images(directory):
    """Find available image files in the specified directory"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(directory.glob(f'*{ext}'))
        image_files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted([str(f) for f in image_files])

def create_sample_images(output_dir):
    """Create sample images if no images are found"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a simple pattern image 1 (checkerboard)
    img1 = Image.new('RGB', (400, 400), 'white')
    pixels1 = img1.load()
    for i in range(400):
        for j in range(400):
            if (i // 30 + j // 30) % 2:
                pixels1[i, j] = (30, 60, 150)  # Deep blue
            else:
                pixels1[i, j] = (220, 220, 255)  # Light blue
    
    sample1_path = output_path / 'sample1.png'
    img1.save(sample1_path)
    print(f"Created sample image: {sample1_path}")
    
    # Create a simple pattern image 2 (gradient)
    img2 = Image.new('RGB', (400, 400), 'white')
    pixels2 = img2.load()
    for i in range(400):
        for j in range(400):
            # Create a radial gradient
            center_x, center_y = 200, 200
            distance = ((i - center_x)**2 + (j - center_y)**2)**0.5
            color_val = int(255 * (distance / 280)) % 255
            pixels2[i, j] = (255 - color_val, color_val // 2, color_val)
    
    sample2_path = output_path / 'sample2.png'
    img2.save(sample2_path)
    print(f"Created sample image: {sample2_path}")
    
    return [str(sample1_path), str(sample2_path)]

def analyze_image_characteristics(img):
    """Analyze image characteristics like brightness, contrast, and dominant colors"""
    # Convert to grayscale for luminance analysis
    gray = img.convert('L')
    gray_array = np.array(gray)
    
    # Calculate characteristics
    brightness = np.mean(gray_array)
    contrast = np.std(gray_array)
    
    # Get color statistics
    rgb_array = np.array(img)
    avg_color = np.mean(rgb_array, axis=(0, 1))
    
    return {
        'brightness': brightness,
        'contrast': contrast, 
        'avg_color': avg_color,
        'is_dark': brightness < 128
    }

def unify_images(img1, img2, method='histogram_match'):
    """
    Unify two images using techniques from ASCII art programs
    Methods: 'histogram_match', 'color_balance', 'brightness_match', 'edge_enhance'
    """
    img1_chars = analyze_image_characteristics(img1)
    img2_chars = analyze_image_characteristics(img2)
    
    if method == 'histogram_match':
        return histogram_match_images(img1, img2)
    elif method == 'color_balance':
        return color_balance_images(img1, img2, img1_chars, img2_chars)
    elif method == 'brightness_match':
        return brightness_match_images(img1, img2, img1_chars, img2_chars)
    elif method == 'edge_enhance':
        return edge_enhance_images(img1, img2)
    else:
        return img1, img2

def histogram_match_images(img1, img2):
    """Match histogram of img2 to img1 for tonal consistency"""
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    result_img2 = np.zeros_like(img2_array)
    
    for channel in range(3):
        hist1, bins1 = np.histogram(img1_array[:,:,channel].flatten(), 256, [0,256])
        hist2, bins2 = np.histogram(img2_array[:,:,channel].flatten(), 256, [0,256])
        
        cdf1 = hist1.cumsum()
        cdf2 = hist2.cumsum()
        
        cdf1 = cdf1 / cdf1[-1] * 255
        cdf2 = cdf2 / cdf2[-1] * 255
        
        lut = np.interp(cdf2, cdf1, range(256))
        result_img2[:,:,channel] = np.interp(img2_array[:,:,channel].flatten(), range(256), lut).reshape(img2_array[:,:,channel].shape)
    
    return img1, Image.fromarray(result_img2.astype(np.uint8))

def color_balance_images(img1, img2, chars1, chars2):
    """Balance colors between images for harmony"""
    img1_hsv = img1.convert('HSV')
    img2_hsv = img2.convert('HSV')
    
    img1_array = np.array(img1_hsv)
    img2_array = np.array(img2_hsv)
    
    target_sat = (np.mean(img1_array[:,:,1]) + np.mean(img2_array[:,:,1])) / 2
    img2_array[:,:,1] = (img2_array[:,:,1] * 0.7 + target_sat * 0.3).astype(np.uint8)
    
    result_img2 = Image.fromarray(img2_array, 'HSV').convert('RGB')
    return img1, result_img2

def brightness_match_images(img1, img2, chars1, chars2):
    """Match brightness and contrast for cohesion"""
    target_brightness = (chars1['brightness'] + chars2['brightness']) / 2
    
    enhancer1 = ImageEnhance.Brightness(img1)
    enhancer2 = ImageEnhance.Brightness(img2)
    
    brightness_factor1 = target_brightness / chars1['brightness']
    brightness_factor2 = target_brightness / chars2['brightness']
    
    brightness_factor1 = np.clip(brightness_factor1, 0.5, 2.0)
    brightness_factor2 = np.clip(brightness_factor2, 0.5, 2.0)
    
    result_img1 = enhancer1.enhance(brightness_factor1)
    result_img2 = enhancer2.enhance(brightness_factor2)
    
    contrast_enhancer1 = ImageEnhance.Contrast(result_img1)
    contrast_enhancer2 = ImageEnhance.Contrast(result_img2)
    
    target_contrast = (chars1['contrast'] + chars2['contrast']) / 2
    contrast_factor1 = target_contrast / chars1['contrast'] if chars1['contrast'] > 0 else 1.0
    contrast_factor2 = target_contrast / chars2['contrast'] if chars2['contrast'] > 0 else 1.0
    
    contrast_factor1 = np.clip(contrast_factor1, 0.5, 2.0)
    contrast_factor2 = np.clip(contrast_factor2, 0.5, 2.0)
    
    result_img1 = contrast_enhancer1.enhance(contrast_factor1)
    result_img2 = contrast_enhancer2.enhance(contrast_factor2)
    
    return result_img1, result_img2

def edge_enhance_images(img1, img2):
    """Enhance edges to create ASCII-art-like effect"""
    edge_filter = ImageFilter.EDGE_ENHANCE_MORE
    
    result_img1 = img1.filter(edge_filter)
    result_img2 = img2.filter(edge_filter)
    
    sat_enhancer1 = ImageEnhance.Color(result_img1)
    sat_enhancer2 = ImageEnhance.Color(result_img2)
    
    result_img1 = sat_enhancer1.enhance(0.8)
    result_img2 = sat_enhancer2.enhance(0.8)
    
    return result_img1, result_img2

def yin_yang_with_images(config, image1_path, image2_path, unify_method='brightness_match'):
    """Create a yin-yang symbol using two different images as fill patterns."""
    
    # Get settings from config
    transforms = config.get_transformations()
    processing = config.get_processing_settings()
    output_settings = config.get_output_settings()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Load and prepare images
    try:
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')
        print(f"Loaded images: {Path(image1_path).name} and {Path(image2_path).name}")
        
        # Analyze images before unification
        chars1 = analyze_image_characteristics(img1)
        chars2 = analyze_image_characteristics(img2)
        print(f"Image 1 - Brightness: {chars1['brightness']:.1f}, Contrast: {chars1['contrast']:.1f}")
        print(f"Image 2 - Brightness: {chars2['brightness']:.1f}, Contrast: {chars2['contrast']:.1f}")
        
        # Unify images
        print(f"Applying unification method: {unify_method}")
        img1, img2 = unify_images(img1, img2, unify_method)
        
    except FileNotFoundError as e:
        print(f"Error loading images: {e}")
        return None
    
    # Create coordinate grids
    R = processing['radius']
    resolution = processing['resolution']
    x = np.linspace(-R*1.2, R*1.2, resolution)
    y = np.linspace(-R*1.2, R*1.2, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create masks for different regions
    outside_circle = X**2 + Y**2 > R**2
    left_half = X <= 0
    upper_circle = X**2 + (Y - R/2)**2 <= (R/2)**2
    lower_circle = X**2 + (Y + R/2)**2 <= (R/2)**2
    upper_eye = X**2 + (Y - R/2)**2 <= (R/8)**2
    lower_eye = X**2 + (Y + R/2)**2 <= (R/8)**2
    
    # Create the final image array
    result_img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    
    # Resize input images to match our grid
    img1_resized = img1.resize((resolution, resolution))
    img2_resized = img2.resize((resolution, resolution))
    
    # Apply transformations
    if transforms['img1_flip_horizontal']:
        img1_resized = img1_resized.transpose(Image.FLIP_LEFT_RIGHT)
    if transforms['img1_flip_vertical']:
        img1_resized = img1_resized.transpose(Image.FLIP_TOP_BOTTOM)
    if transforms['img1_rotation'] != 0:
        img1_resized = img1_resized.rotate(-transforms['img1_rotation'], expand=True)
    
    if transforms['img2_flip_horizontal']:
        img2_resized = img2_resized.transpose(Image.FLIP_LEFT_RIGHT)
    if transforms['img2_flip_vertical']:
        img2_resized = img2_resized.transpose(Image.FLIP_TOP_BOTTOM)
    if transforms['img2_rotation'] != 0:
        img2_resized = img2_resized.rotate(-transforms['img2_rotation'], expand=True)
    
    # Resize back to target resolution after rotation
    img1_resized = img1_resized.resize((resolution, resolution))
    img2_resized = img2_resized.resize((resolution, resolution))
    
    img1_array = np.array(img1_resized)
    img2_array = np.array(img2_resized)
    
    # Fill regions with appropriate images
    for i in range(resolution):
        for j in range(resolution):
            if outside_circle[i, j]:
                result_img[i, j] = [255, 255, 255]
            elif upper_eye[i, j]:
                result_img[i, j] = img2_array[i, j]
            elif lower_eye[i, j]:
                result_img[i, j] = img1_array[i, j]
            elif left_half[i, j] and not upper_circle[i, j]:
                result_img[i, j] = img1_array[i, j]
            elif not left_half[i, j] and not lower_circle[i, j]:
                result_img[i, j] = img2_array[i, j]
            elif upper_circle[i, j]:
                result_img[i, j] = img2_array[i, j]
            elif lower_circle[i, j]:
                result_img[i, j] = img1_array[i, j]
            else:
                result_img[i, j] = [255, 255, 255]
    
    # Display the result
    ax.imshow(result_img, extent=[-R*1.2, R*1.2, -R*1.2, R*1.2], origin='lower')
    
    # Add border
    outer_border = Circle((0, 0), R, facecolor='none', edgecolor='black', linewidth=4)
    ax.add_patch(outer_border)
    
    # Formatting
    ax.set_xlim(-R*1.2, R*1.2)
    ax.set_ylim(-R*1.2, R*1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title(f'Yin-Yang with Images ({unify_method})', fontsize=16, pad=20)
    
    # Save to output directory
    output_filename = get_next_output_filename(
        output_settings['output_directory'], 
        output_settings['filename_prefix'], 
        unify_method
    )
    plt.savefig(output_filename, dpi=output_settings['dpi'], bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_filename}")
    
    # Show the image if configured
    if output_settings['show_images']:
        plt.show()
    else:
        plt.close()
    
    return output_filename

def main():
    """Main function - reads config and generates yin-yangs"""
    print("Yin-Yang Image Converter")
    print("=======================")
    print("Configuration-driven - edit config.ini to customize settings!")
    print()
    
    # Load configuration
    config = YinYangConfig()
    
    # Get image paths
    img1_path, img2_path = config.get_image_paths()
    
    if not (img1_path and img2_path):
        # Auto-detect images from bottomimage and topimage directories
        image_dirs = config.get_image_directories()
        bottom_dir = image_dirs['bottomimage']
        top_dir = image_dirs['topimage']
        
        bottom_images = find_available_images(bottom_dir)
        top_images = find_available_images(top_dir)
        
        if bottom_images and top_images:
            img1_path = bottom_images[0]  # First image from bottomimage folder (yin)
            img2_path = top_images[0]     # First image from topimage folder (yang)
            print(f"Bottom image: {Path(img1_path).name} from {bottom_dir}")
            print(f"Top image: {Path(img2_path).name} from {top_dir}")
        else:
            missing_dirs = []
            if not bottom_images:
                missing_dirs.append(f"bottomimage ({bottom_dir})")
            if not top_images:
                missing_dirs.append(f"topimage ({top_dir})")
            
            print(f"No images found in: {', '.join(missing_dirs)}")
            print("Creating sample images...")
            sample_images = create_sample_images(config.get_output_settings()['output_directory'])
            img1_path, img2_path = sample_images[0], sample_images[1]
    
    # Get methods to run
    methods = config.get_methods()
    
    print(f"\\nGenerating {len(methods)} yin-yang variation(s)...")
    
    # Generate yin-yangs with each method
    results = []
    for method in methods:
        print(f"\\nProcessing with {method}...")
        result = yin_yang_with_images(config, img1_path, img2_path, method)
        if result:
            results.append(result)
    
    print(f"\\n✅ Generated {len(results)} yin-yang images!")
    print("\\n💡 To customize settings, edit config.ini")
    
    return results

if __name__ == "__main__":
    main()