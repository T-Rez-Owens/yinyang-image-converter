import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import matplotlib.patches as patches
from PIL import Image, ImageEnhance, ImageFilter
import os
from urllib.request import urlretrieve
import requests
from scipy import ndimage
from dataclasses import dataclass
from typing import Union, Tuple, Optional

@dataclass
class YinYangField:
    """Yin-yang topology data structure"""
    region_id: np.ndarray      # (H, W) with region identifiers
    uv_A: np.ndarray          # (H, W, 2) UV coords for lobe A  
    uv_B: np.ndarray          # (H, W, 2) UV coords for lobe B
    dist_boundary: np.ndarray  # (H, W) signed distance to boundaries

def generate_yinyang_field(resolution: int = 400, R: float = 1.0) -> YinYangField:
    """Generate yin-yang topology field with UV coordinates for each lobe"""
    # Create coordinate grids
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
    
    # Initialize region IDs: 0=outside, 1=lobe_A, 2=lobe_B, 3=dot_A, 4=dot_B
    region_id = np.zeros((resolution, resolution), dtype=np.int32)
    
    # Assign regions
    for i in range(resolution):
        for j in range(resolution):
            if outside_circle[i, j]:
                region_id[i, j] = 0  # outside
            elif upper_eye[i, j]:
                region_id[i, j] = 4  # dot_B (opposite)
            elif lower_eye[i, j]:
                region_id[i, j] = 3  # dot_A (opposite)
            elif left_half[i, j] and not upper_circle[i, j]:
                region_id[i, j] = 1  # lobe_A
            elif not left_half[i, j] and not lower_circle[i, j]:
                region_id[i, j] = 2  # lobe_B
            elif upper_circle[i, j]:
                region_id[i, j] = 2  # lobe_B
            elif lower_circle[i, j]:
                region_id[i, j] = 1  # lobe_A
    
    # Generate UV coordinates for each lobe
    uv_A = np.zeros((resolution, resolution, 2), dtype=np.float32)
    uv_B = np.zeros((resolution, resolution, 2), dtype=np.float32)
    
    # Simple UV mapping: normalize X,Y to [0,1] within each lobe's bounding box
    lobe_A_mask = (region_id == 1) | (region_id == 3)
    lobe_B_mask = (region_id == 2) | (region_id == 4)
    
    if np.any(lobe_A_mask):
        x_min_A, x_max_A = X[lobe_A_mask].min(), X[lobe_A_mask].max()
        y_min_A, y_max_A = Y[lobe_A_mask].min(), Y[lobe_A_mask].max()
        uv_A[lobe_A_mask, 0] = (X[lobe_A_mask] - x_min_A) / (x_max_A - x_min_A)
        uv_A[lobe_A_mask, 1] = (Y[lobe_A_mask] - y_min_A) / (y_max_A - y_min_A)
    
    if np.any(lobe_B_mask):
        x_min_B, x_max_B = X[lobe_B_mask].min(), X[lobe_B_mask].max()
        y_min_B, y_max_B = Y[lobe_B_mask].min(), Y[lobe_B_mask].max()
        uv_B[lobe_B_mask, 0] = (X[lobe_B_mask] - x_min_B) / (x_max_B - x_min_B)
        uv_B[lobe_B_mask, 1] = (Y[lobe_B_mask] - y_min_B) / (y_max_B - y_min_B)
    
    # Distance to boundary (simplified)
    dist_boundary = np.minimum(np.sqrt(X**2 + Y**2) - R, R - np.sqrt(X**2 + Y**2))
    
    return YinYangField(region_id, uv_A, uv_B, dist_boundary)

def load_content_as_brightness(source: Union[str, np.ndarray, callable]) -> np.ndarray:
    """Load arbitrary content and convert to normalized brightness array [0,1]"""
    if isinstance(source, str):
        # File path - load as image
        if os.path.exists(source):
            img = Image.open(source).convert('L')  # Convert to grayscale
            return np.array(img, dtype=np.float32) / 255.0
        else:
            raise FileNotFoundError(f"Content source not found: {source}")
    elif isinstance(source, np.ndarray):
        # Direct brightness array
        return np.clip(source.astype(np.float32), 0, 1)
    elif callable(source):
        # Procedural generator
        return np.clip(source().astype(np.float32), 0, 1)
    else:
        raise ValueError(f"Unsupported content source type: {type(source)}")

def warp_content_to_lobe(content: np.ndarray, field: YinYangField, lobe: str, orientation: str = "upright") -> np.ndarray:
    """Warp content brightness into specified lobe using UV mapping"""
    resolution = field.region_id.shape[0]
    warped = np.zeros((resolution, resolution), dtype=np.float32)
    
    # Select appropriate lobe mask and UV coordinates
    if lobe == "A":
        lobe_mask = (field.region_id == 1) | (field.region_id == 3)
        uv_coords = field.uv_A
    elif lobe == "B":
        lobe_mask = (field.region_id == 2) | (field.region_id == 4)
        uv_coords = field.uv_B
    else:
        raise ValueError(f"Invalid lobe: {lobe}. Must be 'A' or 'B'")
    
    if not np.any(lobe_mask):
        return warped
    
    # Get content dimensions
    content_h, content_w = content.shape[:2]
    
    # Apply orientation transformation to UV coordinates
    uv = uv_coords[lobe_mask]
    if orientation == "inverted":
        uv = 1.0 - uv  # Flip both U and V
    
    # Map UV [0,1] to content pixel coordinates
    content_x = np.clip((uv[:, 0] * (content_w - 1)).astype(int), 0, content_w - 1)
    content_y = np.clip((uv[:, 1] * (content_h - 1)).astype(int), 0, content_h - 1)
    
    # Sample content at mapped coordinates
    warped[lobe_mask] = content[content_y, content_x]
    
    return warped

def determine_brightness_roles(content_A: np.ndarray, content_B: np.ndarray, strategy: str = "auto") -> dict:
    """Determine which content should be dark vs light lobe"""
    if strategy == "auto":
        mean_A = np.mean(content_A[content_A > 0])  # Ignore background
        mean_B = np.mean(content_B[content_B > 0])
        
        # Darker content becomes dark lobe
        if mean_A < mean_B:
            return {"dark_lobe": "A", "light_lobe": "B", "swap": False}
        else:
            return {"dark_lobe": "B", "light_lobe": "A", "swap": True}
    else:
        # Default: no swap
        return {"dark_lobe": "A", "light_lobe": "B", "swap": False}

def compose_dual_brightness(field: YinYangField, warped_A: np.ndarray, warped_B: np.ndarray, roles: dict) -> np.ndarray:
    """Compose final brightness field with proper yin-yang contrast"""
    resolution = field.region_id.shape[0]
    result = np.ones((resolution, resolution), dtype=np.float32)  # White background
    
    # Apply role-based contrast
    if roles["swap"]:
        warped_A, warped_B = warped_B, warped_A
    
    # Fill lobe A regions (dark lobe)
    lobe_A_mask = (field.region_id == 1) | (field.region_id == 3)
    result[lobe_A_mask] = warped_A[lobe_A_mask]
    
    # Fill lobe B regions (light lobe) - invert for contrast
    lobe_B_mask = (field.region_id == 2) | (field.region_id == 4)
    result[lobe_B_mask] = 1.0 - warped_B[lobe_B_mask]
    
    # Set outside region to background
    outside_mask = (field.region_id == 0)
    result[outside_mask] = 1.0
    
    return result

def yin_yang_with_content(content_A_source, content_B_source, resolution: int = 400, roles_strategy: str = "auto") -> np.ndarray:
    """Main entry point for content-agnostic yin-yang generation"""
    # Generate topology
    field = generate_yinyang_field(resolution)
    
    # Load content
    content_A = load_content_as_brightness(content_A_source)
    content_B = load_content_as_brightness(content_B_source)
    
    # Warp content to lobes
    warped_A = warp_content_to_lobe(content_A, field, "A", "upright")
    warped_B = warp_content_to_lobe(content_B, field, "B", "inverted")
    
    # Determine roles
    roles = determine_brightness_roles(warped_A, warped_B, roles_strategy)
    
    # Compose final brightness
    brightness = compose_dual_brightness(field, warped_A, warped_B, roles)
    
    return brightness

def find_available_images():
    """Find available image files in the current directory"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
    image_files = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    return image_files

def create_sample_images():
    """Create sample images if no images are found"""
    
    # Create a simple pattern image 1 (checkerboard)
    img1 = Image.new('RGB', (400, 400), 'white')
    pixels1 = img1.load()
    for i in range(400):
        for j in range(400):
            if (i // 30 + j // 30) % 2:
                pixels1[i, j] = (30, 60, 150)  # Deep blue
            else:
                pixels1[i, j] = (220, 220, 255)  # Light blue
    img1.save('default1.png')
    print("Created default1.png (blue pattern)")
    
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
    img2.save('default2.png')
    print("Created default2.png (gradient pattern)")
    
    return ['default1.png', 'default2.png']

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
        # Match histograms for similar tonal distribution
        return histogram_match_images(img1, img2)
    
    elif method == 'color_balance':
        # Balance colors to create harmony
        return color_balance_images(img1, img2, img1_chars, img2_chars)
    
    elif method == 'brightness_match':
        # Match brightness and contrast for cohesion
        return brightness_match_images(img1, img2, img1_chars, img2_chars)
    
    elif method == 'edge_enhance':
        # Enhance edges and create ASCII-like effect
        return edge_enhance_images(img1, img2)
    
    else:
        return img1, img2

def histogram_match_images(img1, img2):
    """Match histogram of img2 to img1 for tonal consistency"""
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    result_img2 = np.zeros_like(img2_array)
    
    for channel in range(3):
        # Calculate histograms
        hist1, bins1 = np.histogram(img1_array[:,:,channel].flatten(), 256, [0,256])
        hist2, bins2 = np.histogram(img2_array[:,:,channel].flatten(), 256, [0,256])
        
        # Calculate CDFs
        cdf1 = hist1.cumsum()
        cdf2 = hist2.cumsum()
        
        # Normalize
        cdf1 = cdf1 / cdf1[-1] * 255
        cdf2 = cdf2 / cdf2[-1] * 255
        
        # Create lookup table
        lut = np.interp(cdf2, cdf1, range(256))
        result_img2[:,:,channel] = np.interp(img2_array[:,:,channel].flatten(), range(256), lut).reshape(img2_array[:,:,channel].shape)
    
    return img1, Image.fromarray(result_img2.astype(np.uint8))

def color_balance_images(img1, img2, chars1, chars2):
    """Balance colors between images for harmony"""
    # Convert to HSV for better color manipulation
    img1_hsv = img1.convert('HSV')
    img2_hsv = img2.convert('HSV')
    
    img1_array = np.array(img1_hsv)
    img2_array = np.array(img2_hsv)
    
    # Adjust saturation to create harmony
    target_sat = (np.mean(img1_array[:,:,1]) + np.mean(img2_array[:,:,1])) / 2
    
    # Adjust img2's saturation toward the average
    img2_array[:,:,1] = (img2_array[:,:,1] * 0.7 + target_sat * 0.3).astype(np.uint8)
    
    # Convert back to RGB
    result_img2 = Image.fromarray(img2_array, 'HSV').convert('RGB')
    
    return img1, result_img2

def brightness_match_images(img1, img2, chars1, chars2):
    """Match brightness and contrast for cohesion"""
    # Target brightness - average of both images
    target_brightness = (chars1['brightness'] + chars2['brightness']) / 2
    
    # Adjust brightness
    enhancer1 = ImageEnhance.Brightness(img1)
    enhancer2 = ImageEnhance.Brightness(img2)
    
    brightness_factor1 = target_brightness / chars1['brightness']
    brightness_factor2 = target_brightness / chars2['brightness']
    
    # Limit adjustment to prevent over-correction
    brightness_factor1 = np.clip(brightness_factor1, 0.5, 2.0)
    brightness_factor2 = np.clip(brightness_factor2, 0.5, 2.0)
    
    result_img1 = enhancer1.enhance(brightness_factor1)
    result_img2 = enhancer2.enhance(brightness_factor2)
    
    # Also match contrast
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
    # Apply edge enhancement filter
    edge_filter = ImageFilter.EDGE_ENHANCE_MORE
    
    result_img1 = img1.filter(edge_filter)
    result_img2 = img2.filter(edge_filter)
    
    # Slightly reduce saturation for ASCII-like feel
    sat_enhancer1 = ImageEnhance.Color(result_img1)
    sat_enhancer2 = ImageEnhance.Color(result_img2)
    
    result_img1 = sat_enhancer1.enhance(0.8)
    result_img2 = sat_enhancer2.enhance(0.8)
    
    return result_img1, result_img2

def yin_yang_with_images(R=1.0, image1_path=None, image2_path=None, unify_method='brightness_match'):
    """
    Create a yin-yang symbol using two different images as fill patterns.
    
    Parameters:
    R: Radius of the yin-yang
    image1_path: Path to first image (for "black" half)
    image2_path: Path to second image (for "white" half)
    unify_method: Method to unify images ('histogram_match', 'color_balance', 'brightness_match', 'edge_enhance')
    """
    
    # Auto-find images if not specified
    if image1_path is None or image2_path is None:
        available_images = find_available_images()
        if len(available_images) >= 2:
            image1_path = available_images[0]
            image2_path = available_images[1]
            print(f"Using found images: {image1_path} and {image2_path}")
        else:
            print("Not enough images found, creating sample images...")
            sample_images = create_sample_images()
            image1_path = sample_images[0]
            image2_path = sample_images[1]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Load and prepare images
    try:
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')
        print(f"Loaded images: {image1_path} and {image2_path}")
        
        # Analyze images before unification
        chars1 = analyze_image_characteristics(img1)
        chars2 = analyze_image_characteristics(img2)
        print(f"Image 1 - Brightness: {chars1['brightness']:.1f}, Contrast: {chars1['contrast']:.1f}")
        print(f"Image 2 - Brightness: {chars2['brightness']:.1f}, Contrast: {chars2['contrast']:.1f}")
        
        # Unify images using ASCII art techniques
        print(f"Applying unification method: {unify_method}")
        img1, img2 = unify_images(img1, img2, unify_method)
        
    except FileNotFoundError as e:
        print(f"Error loading images: {e}")
        print("Using solid colors instead")
        return yin_yang_solid(R)
    
    # Create coordinate grids
    x = np.linspace(-R*1.2, R*1.2, 400)
    y = np.linspace(-R*1.2, R*1.2, 400)
    X, Y = np.meshgrid(x, y)
    
    # Create masks for different regions
    # Outside main circle
    outside_circle = X**2 + Y**2 > R**2
    
    # Left half (original "black" region)
    left_half = X <= 0
    
    # Upper small circle (centered at (0, R/2))
    upper_circle = X**2 + (Y - R/2)**2 <= (R/2)**2
    
    # Lower small circle (centered at (0, -R/2))
    lower_circle = X**2 + (Y + R/2)**2 <= (R/2)**2
    
    # Upper eye (centered at (0, R/2))
    upper_eye = X**2 + (Y - R/2)**2 <= (R/8)**2
    
    # Lower eye (centered at (0, -R/2))
    lower_eye = X**2 + (Y + R/2)**2 <= (R/8)**2
    
    # Create the final image array
    result_img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Resize input images to match our grid
    img1_resized = img1.resize((400, 400))
    img2_resized = img2.resize((400, 400))
    img1_array = np.array(img1_resized)
    img2_array = np.array(img2_resized)
    
    # Fill regions with appropriate images
    for i in range(400):
        for j in range(400):
            if outside_circle[i, j]:
                # Outside the main circle - white background
                result_img[i, j] = [255, 255, 255]
            elif upper_eye[i, j]:
                # Upper eye - use image2 (opposite of the region it's in)
                result_img[i, j] = img2_array[i, j]
            elif lower_eye[i, j]:
                # Lower eye - use image1 (opposite of the region it's in)
                result_img[i, j] = img1_array[i, j]
            elif left_half[i, j] and not upper_circle[i, j]:
                # Left half minus the upper circle - use image1
                result_img[i, j] = img1_array[i, j]
            elif not left_half[i, j] and not lower_circle[i, j]:
                # Right half minus the lower circle - use image2
                result_img[i, j] = img2_array[i, j]
            elif upper_circle[i, j]:
                # Upper circle - use image2
                result_img[i, j] = img2_array[i, j]
            elif lower_circle[i, j]:
                # Lower circle - use image1
                result_img[i, j] = img1_array[i, j]
            else:
                # Fallback - white
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
    plt.show()

def yin_yang_solid(R=1.0):
    """
    Fallback function for solid color yin-yang (original version)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create the outer white circle
    outer_circle = Circle((0, 0), R, facecolor='white', edgecolor='black', linewidth=3)
    ax.add_patch(outer_circle)
    
    # Create the black half using a semicircle wedge (left side)
    black_half = Wedge((0, 0), R, 90, 270, facecolor='black', edgecolor='none')
    ax.add_patch(black_half)
    
    # Add the upper small white semicircle (in the black region)
    upper_white = Circle((0, R/2), R/2, facecolor='white', edgecolor='none')
    ax.add_patch(upper_white)
    
    # Add the lower small black semicircle (in the white region)
    lower_black = Circle((0, -R/2), R/2, facecolor='black', edgecolor='none')
    ax.add_patch(lower_black)
    
    # Add the small eyes
    # Black eye in the white upper circle
    black_eye = Circle((0, R/2), R/8, facecolor='black', edgecolor='none')
    ax.add_patch(black_eye)
    
    # White eye in the black lower circle
    white_eye = Circle((0, -R/2), R/8, facecolor='white', edgecolor='none')
    ax.add_patch(white_eye)
    
    # Add outer border
    outer_border = Circle((0, 0), R, facecolor='none', edgecolor='black', linewidth=3)
    ax.add_patch(outer_border)
    
    # Formatting
    ax.set_xlim(-R*1.2, R*1.2)
    ax.set_ylim(-R*1.2, R*1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title('Classic Yin-Yang Symbol', fontsize=16, pad=20)
    plt.show()

def download_sample_images():
    """Download some sample images for demonstration"""
    try:
        # Download a nature image (this is just an example - you can use any images)
        if not os.path.exists('nature.jpg'):
            print("You can add your own images to the same directory as this script.")
            print("Supported formats: .png, .jpg, .jpeg, .bmp, .gif")
            print("Name them 'image1.png' and 'image2.png' or specify custom paths.")
    except Exception as e:
        print(f"Note: {e}")

# Run the yin-yang visualization
if __name__ == "__main__":
    print("Content-Agnostic Yin-Yang Generator")
    print("===================================")
    print("This script creates a yin-yang using any two content sources!")
    print()
    
    # Show available images
    available_images = find_available_images()
    if available_images:
        print(f"Found {len(available_images)} images in directory:")
        for i, img in enumerate(available_images):
            print(f"  {i+1}. {img}")
        print()
    
    # Check mode selection
    mode = input("Enter mode:\n1. Use available images (default)\n2. Legacy image mode\n3. Custom content paths\n> ").strip()
    
    if mode == '3':
        # Custom content paths
        content1 = input("Enter path to first content source: ").strip()
        content2 = input("Enter path to second content source: ").strip()
        
        try:
            print(f"Generating yin-yang with content: {content1} and {content2}")
            brightness = yin_yang_with_content(content1, content2)
            
            # Display the result
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(brightness, cmap='gray', extent=[-1.2, 1.2, -1.2, 1.2], origin='lower')
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')
            ax.axis('off')
            plt.title('Content-Agnostic Yin-Yang', fontsize=16, pad=20)
            plt.show()
            
        except Exception as e:
            print(f"Error: {e}")
            print("Falling back to legacy mode...")
            yin_yang_with_images(R=1.0)
            
    elif mode == '2':
        # Legacy image mode
        methods = ['brightness_match', 'histogram_match', 'color_balance', 'edge_enhance']
        method_choice = input("Choose unification method (1-4) or press Enter for default:\n1. Brightness Match\n2. Histogram Match\n3. Color Balance\n4. Edge Enhance\n> ").strip()
        
        if method_choice in ['1', '2', '3', '4']:
            method = methods[int(method_choice) - 1]
        else:
            method = 'brightness_match'
        
        yin_yang_with_images(R=1.0, unify_method=method)
        
    else:
        # Use available images with new system
        if len(available_images) >= 2:
            print(f"Using new content-agnostic system with: {available_images[0]} and {available_images[1]}")
            try:
                brightness = yin_yang_with_content(available_images[0], available_images[1])
                
                # Display the result
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(brightness, cmap='gray', extent=[-1.2, 1.2, -1.2, 1.2], origin='lower')
                
                # Add border
                outer_border = Circle((0, 0), 1.0, facecolor='none', edgecolor='black', linewidth=4)
                ax.add_patch(outer_border)
                
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)
                ax.set_aspect('equal')
                ax.axis('off')
                plt.title('Content-Agnostic Yin-Yang', fontsize=16, pad=20)
                plt.show()
                
            except Exception as e:
                print(f"Error with new system: {e}")
                print("Falling back to legacy mode...")
                yin_yang_with_images(R=1.0)
        else:
            print("Not enough images found, creating sample images...")
            sample_images = create_sample_images()
            brightness = yin_yang_with_content(sample_images[0], sample_images[1])
            
            # Display the result
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(brightness, cmap='gray', extent=[-1.2, 1.2, -1.2, 1.2], origin='lower')
            
            # Add border
            outer_border = Circle((0, 0), 1.0, facecolor='none', edgecolor='black', linewidth=4)
            ax.add_patch(outer_border)
            
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')
            ax.axis('off')
            plt.title('Content-Agnostic Yin-Yang (Sample)', fontsize=16, pad=20)
            plt.show()
    
    print("\nTip: The new content-agnostic system can use any image file as content!")
    print("Place images in this directory and run mode 1, or specify custom paths with mode 3.")
