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
    low_img = Image.new('RGB', (400, 400), 'white')
    pixels1 = low_img.load()
    for i in range(400):
        for j in range(400):
            if (i // 30 + j // 30) % 2:
                pixels1[i, j] = (30, 60, 150)  # Deep blue
            else:
                pixels1[i, j] = (220, 220, 255)  # Light blue
    
    sample1_path = output_path / 'sample1.png'
    low_img.save(sample1_path)
    print(f"Created sample image: {sample1_path}")
    
    # Create a simple pattern top image (gradient)
    top_img = Image.new('RGB', (400, 400), 'white')
    top_pixels = top_img.load()
    for i in range(400):
        for j in range(400):
            # Create a radial gradient
            center_x, center_y = 200, 200
            distance = ((i - center_x)**2 + (j - center_y)**2)**0.5
            color_val = int(255 * (distance / 280)) % 255
            top_pixels[i, j] = (255 - color_val, color_val // 2, color_val)
    
    sample2_path = output_path / 'sample2.png'
    top_img.save(sample2_path)
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

def unify_images(low_img, top_img, method='histogram_match'):
    """
    Unify two images using techniques from ASCII art programs
    Methods: 'histogram_match', 'color_balance', 'brightness_match', 'edge_enhance'
    """
    low_img_chars = analyze_image_characteristics(low_img)
    top_img_chars = analyze_image_characteristics(top_img)
    
    if method == 'histogram_match':
        return histogram_match_images(low_img, top_img)
    elif method == 'color_balance':
        return color_balance_images(low_img, top_img, low_img_chars, top_img_chars)
    elif method == 'brightness_match':
        return brightness_match_images(low_img, top_img, low_img_chars, top_img_chars)
    elif method == 'edge_enhance':
        return edge_enhance_images(low_img, top_img)
    else:
        return low_img, top_img

def histogram_match_images(low_img, top_img):
    """Match histogram of top_img to low_img for tonal consistency"""
    low_img_array = np.array(low_img)
    top_img_array = np.array(top_img)
    result_top_img = np.zeros_like(top_img_array)
    
    for channel in range(3):
        hist1, bins1 = np.histogram(low_img_array[:,:,channel].flatten(), 256, [0,256])
        hist2, bins2 = np.histogram(top_img_array[:,:,channel].flatten(), 256, [0,256])
        
        cdf1 = hist1.cumsum()
        cdf2 = hist2.cumsum()
        
        cdf1 = cdf1 / cdf1[-1] * 255
        cdf2 = cdf2 / cdf2[-1] * 255
        
        lut = np.interp(cdf2, cdf1, range(256))
        result_top_img[:,:,channel] = np.interp(top_img_array[:,:,channel].flatten(), range(256), lut).reshape(top_img_array[:,:,channel].shape)
    
    return low_img, Image.fromarray(result_top_img.astype(np.uint8))

def color_balance_images(low_img, top_img, chars1, top_img_chars):
    """Balance colors between images for harmony"""
    low_img_hsv = low_img.convert('HSV')
    top_img_hsv = top_img.convert('HSV')
    
    low_img_array = np.array(low_img_hsv)
    top_img_array = np.array(top_img_hsv)
    
    target_sat = (np.mean(low_img_array[:,:,1]) + np.mean(top_img_array[:,:,1])) / 2
    top_img_array[:,:,1] = (top_img_array[:,:,1] * 0.7 + target_sat * 0.3).astype(np.uint8)
    
    result_top_img = Image.fromarray(top_img_array, 'HSV').convert('RGB')
    return low_img, result_top_img

def brightness_match_images(low_img, top_img, chars1, top_img_chars):
    """Match brightness and contrast for cohesion"""
    target_brightness = (chars1['brightness'] + chars2['brightness']) / 2
    
    enhancer1 = ImageEnhance.Brightness(low_img)
    enhancer_top = ImageEnhance.Brightness(top_img)
    
    brightness_factor1 = target_brightness / chars1['brightness']
    brightness_factor2 = target_brightness / chars2['brightness']
    
    brightness_factor1 = np.clip(brightness_factor1, 0.5, 2.0)
    brightness_factor_top = np.clip(brightness_factor_top, 0.5, 2.0)
    
    result_low_img = enhancer1.enhance(brightness_factor1)
    result_top_img = enhancer_top.enhance(brightness_factor_top)
    
    contrast_enhancer1 = ImageEnhance.Contrast(result_low_img)
    contrast_enhancer_top = ImageEnhance.Contrast(result_top_img)
    
    target_contrast = (chars1['contrast'] + top_img_chars['contrast']) / 2
    contrast_factor1 = target_contrast / chars1['contrast'] if chars1['contrast'] > 0 else 1.0
    contrast_factor_top = target_contrast / top_img_chars['contrast'] if top_img_chars['contrast'] > 0 else 1.0
    
    contrast_factor1 = np.clip(contrast_factor1, 0.5, 2.0)
    contrast_factor_top = np.clip(contrast_factor_top, 0.5, 2.0)
    
    result_low_img = contrast_enhancer1.enhance(contrast_factor1)
    result_top_img = contrast_enhancer_top.enhance(contrast_factor_top)
    
    return result_low_img, result_top_img

def edge_enhance_images(low_img, top_img):
    """Enhance edges to create ASCII-art-like effect"""
    edge_filter = ImageFilter.EDGE_ENHANCE_MORE
    
    result_low_img = low_img.filter(edge_filter)
    result_top_img = top_img.filter(edge_filter)
    
    sat_enhancer1 = ImageEnhance.Color(result_low_img)
    sat_enhancer_top = ImageEnhance.Color(result_top_img)
    
    result_low_img = sat_enhancer1.enhance(0.8)
    result_top_img = sat_enhancer_top.enhance(0.8)
    
    return result_low_img, result_top_img

def yin_yang_with_images(config, lower_image_path, top_image_path, unify_method='brightness_match'):
    """Create a yin-yang symbol using two different images as fill patterns."""
    
    # Get settings from config
    transforms = config.get_transformations()
    processing = config.get_processing_settings()
    output_settings = config.get_output_settings()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Load and prepare images
    try:
        low_img = Image.open(lower_image_path).convert('RGB')
        top_img = Image.open(top_image_path).convert('RGB')
        print(f"Loaded images: {Path(lower_image_path).name} and {Path(top_image_path).name}")
        
        # Analyze images before unification
        chars1 = analyze_image_characteristics(low_img)
        top_img_chars = analyze_image_characteristics(top_img)
        print(f"Image 1 - Brightness: {chars1['brightness']:.1f}, Contrast: {chars1['contrast']:.1f}")
        print(f"Top Image - Brightness: {top_img_chars['brightness']:.1f}, Contrast: {top_img_chars['contrast']:.1f}")
        
        # Unify images
        print(f"Applying unification method: {unify_method}")
        low_img, top_img = unify_images(low_img, top_img, unify_method)
        
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
    low_img_resized = low_img.resize((resolution, resolution))
    top_img_resized = top_img.resize((resolution, resolution))
    
    # Apply transformations
    if transforms['low_img_flip_horizontal']:
        low_img_resized = low_img_resized.transpose(Image.FLIP_LEFT_RIGHT)
    if transforms['low_img_flip_vertical']:
        low_img_resized = low_img_resized.transpose(Image.FLIP_TOP_lower)
    if transforms['low_img_rotation'] != 0:
        low_img_resized = low_img_resized.rotate(-transforms['low_img_rotation'], expand=True)
    
    if transforms['top_img_flip_horizontal']:
        top_img_resized = top_img_resized.transpose(Image.FLIP_LEFT_RIGHT)
    if transforms['top_img_flip_vertical']:
        top_img_resized = top_img_resized.transpose(Image.FLIP_TOP_lower)
    if transforms['top_img_rotation'] != 0:
        top_img_resized = top_img_resized.rotate(-transforms['top_img_rotation'], expand=True)
    
    # Resize back to target resolution after rotation
    low_img_resized = low_img_resized.resize((resolution, resolution))
    top_img_resized = top_img_resized.resize((resolution, resolution))
    
    low_img_array = np.array(low_img_resized)
    top_img_array = np.array(top_img_resized)
    
    # Fill regions with appropriate images
    for i in range(resolution):
        for j in range(resolution):
            if outside_circle[i, j]:
                result_img[i, j] = [255, 255, 255]
            elif upper_eye[i, j]:
                result_img[i, j] = top_img_array[i, j]
            elif lower_eye[i, j]:
                result_img[i, j] = low_img_array[i, j]
            elif left_half[i, j] and not upper_circle[i, j]:
                result_img[i, j] = low_img_array[i, j]
            elif not left_half[i, j] and not lower_circle[i, j]:
                result_img[i, j] = top_img_array[i, j]
            elif upper_circle[i, j]:
                result_img[i, j] = top_img_array[i, j]
            elif lower_circle[i, j]:
                result_img[i, j] = low_img_array[i, j]
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
    low_img_path, top_image_path = config.get_image_paths()
    
    if not (low_img_path and top_image_path):
        # Auto-detect images from lower_image and top_image directories
        image_dirs = config.get_image_directories()
        lower_dir = image_dirs['lower_image']
        top_dir = image_dirs['top_image']
        
        lower_images = find_available_images(lower_dir)
        top_images = find_available_images(top_dir)
        
        if lower_images and top_images:
            low_img_path = lower_images[0]  # First image from lower_image folder (yin)
            top_image_path = top_images[0]     # First image from top_image folder (yang)
            print(f"lower image: {Path(low_img_path).name} from {lower_dir}")
            print(f"Top image: {Path(top_image_path).name} from {top_dir}\")
        else:
            missing_dirs = []
            if not lower_images:
                missing_dirs.append(f"lower_image ({lower_dir})")
            if not top_images:
                missing_dirs.append(f"top_image ({top_dir})")
            
            print(f"No images found in: {', '.join(missing_dirs)}")
            print("Creating sample images...")
            sample_images = create_sample_images(config.get_output_settings()['output_directory'])
            low_img_path, top_image_path = sample_images[0], sample_images[1]
    
    # Get methods to run
    methods = config.get_methods()
    
    print(f"\\nGenerating {len(methods)} yin-yang variation(s)...")
    
    # Generate yin-yangs with each method
    results = []
    for method in methods:
        print(f"\\nProcessing with {method}...")
        result = yin_yang_with_images(config, low_img_path, top_image_path, method)
        if result:
            results.append(result)
    
    print(f"\\n✅ Generated {len(results)} yin-yang images!")
    print("\\n💡 To customize settings, edit config.ini")
    
    return results

if __name__ == "__main__":
    main()