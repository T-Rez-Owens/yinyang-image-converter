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

def get_next_output_filename(method_suffix=""):
    """Get the next available yinyang output filename with optional method suffix"""
    if not os.path.exists('output'):
        os.makedirs('output')
    
    suffix = f"_{method_suffix}" if method_suffix else ""
    pattern = f'output/yinyang*{suffix}.png'
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return f'output/yinyang1{suffix}.png'
    
    # Extract numbers and find the highest
    numbers = []
    for f in existing_files:
        try:
            # Extract number between 'yinyang' and the suffix
            base = f.split('yinyang')[1]
            if suffix:
                num_part = base.split(suffix)[0]
            else:
                num_part = base.split('.')[0]
            num = int(num_part)
            numbers.append(num)
        except (ValueError, IndexError):
            continue
    
    next_num = max(numbers) + 1 if numbers else 1
    return f'output/yinyang{next_num}{suffix}.png'

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
    low_img = Image.new('RGB', (400, 400), 'white')
    pixels1 = low_img.load()
    for i in range(400):
        for j in range(400):
            if (i // 30 + j // 30) % 2:
                pixels1[i, j] = (30, 60, 150)  # Deep blue
            else:
                pixels1[i, j] = (220, 220, 255)  # Light blue
    low_img.save('default1.png')
    print("Created default1.png (blue pattern)")
    
    # Create a simple pattern image 2 (gradient)
    top_img = Image.new('RGB', (400, 400), 'white')
    pixels2 = top_img.load()
    for i in range(400):
        for j in range(400):
            # Create a radial gradient
            center_x, center_y = 200, 200
            distance = ((i - center_x)**2 + (j - center_y)**2)**0.5
            color_val = int(255 * (distance / 280)) % 255
            pixels2[i, j] = (255 - color_val, color_val // 2, color_val)
    top_img.save('default2.png')
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

def unify_images(low_img, top_img, method='histogram_match'):
    """
    Unify two images using techniques from ASCII art programs
    Methods: 'histogram_match', 'color_balance', 'brightness_match', 'edge_enhance'
    """
    low_img_chars = analyze_image_characteristics(low_img)
    top_img_chars = analyze_image_characteristics(top_img)
    
    if method == 'histogram_match':
        # Match histograms for similar tonal distribution
        return histogram_match_images(low_img, top_img)
    
    elif method == 'color_balance':
        # Balance colors to create harmony
        return color_balance_images(low_img, top_img, low_img_chars, top_img_chars)
    
    elif method == 'brightness_match':
        # Match brightness and contrast for cohesion
        return brightness_match_images(low_img, top_img, low_img_chars, top_img_chars)
    
    elif method == 'edge_enhance':
        # Enhance edges and create ASCII-like effect
        return edge_enhance_images(low_img, top_img)
    
    else:
        return low_img, top_img

def histogram_match_images(low_img, top_img):
    """Match histogram of top_img to low_img for tonal consistency"""
    low_img_array = np.array(low_img)
    top_img_array = np.array(top_img)
    result_top_img = np.zeros_like(top_img_array)
    
    for channel in range(3):
        # Calculate histograms
        hist1, bins1 = np.histogram(low_img_array[:,:,channel].flatten(), 256, [0,256])
        hist2, bins2 = np.histogram(top_img_array[:,:,channel].flatten(), 256, [0,256])
        
        # Calculate CDFs
        cdf1 = hist1.cumsum()
        cdf2 = hist2.cumsum()
        
        # Normalize
        cdf1 = cdf1 / cdf1[-1] * 255
        cdf2 = cdf2 / cdf2[-1] * 255
        
        # Create lookup table
        lut = np.interp(cdf2, cdf1, range(256))
        result_top_img[:,:,channel] = np.interp(top_img_array[:,:,channel].flatten(), range(256), lut).reshape(top_img_array[:,:,channel].shape)
    
    return low_img, Image.fromarray(result_top_img.astype(np.uint8))

def color_balance_images(low_img, top_img, chars1, chars2):
    """Balance colors between images for harmony"""
    # Convert to HSV for better color manipulation
    low_img_hsv = low_img.convert('HSV')
    top_img_hsv = top_img.convert('HSV')
    
    low_img_array = np.array(low_img_hsv)
    top_img_array = np.array(top_img_hsv)
    
    # Adjust saturation to create harmony
    target_sat = (np.mean(low_img_array[:,:,1]) + np.mean(top_img_array[:,:,1])) / 2
    
    # Adjust top_img's saturation toward the average
    top_img_array[:,:,1] = (top_img_array[:,:,1] * 0.7 + target_sat * 0.3).astype(np.uint8)
    
    # Convert back to RGB
    result_top_img = Image.fromarray(top_img_array, 'HSV').convert('RGB')
    
    return low_img, result_top_img

def brightness_match_images(low_img, top_img, chars1, chars2):
    """Match brightness and contrast for cohesion"""
    # Target brightness - average of both images
    target_brightness = (chars1['brightness'] + chars2['brightness']) / 2
    
    # Adjust brightness
    enhancer1 = ImageEnhance.Brightness(low_img)
    enhancer2 = ImageEnhance.Brightness(top_img)
    
    brightness_factor1 = target_brightness / chars1['brightness']
    brightness_factor2 = target_brightness / chars2['brightness']
    
    # Limit adjustment to prevent over-correction
    brightness_factor1 = np.clip(brightness_factor1, 0.5, 2.0)
    brightness_factor2 = np.clip(brightness_factor2, 0.5, 2.0)
    
    result_low_img = enhancer1.enhance(brightness_factor1)
    result_top_img = enhancer2.enhance(brightness_factor2)
    
    # Also match contrast
    contrast_enhancer1 = ImageEnhance.Contrast(result_low_img)
    contrast_enhancer2 = ImageEnhance.Contrast(result_top_img)
    
    target_contrast = (chars1['contrast'] + chars2['contrast']) / 2
    contrast_factor1 = target_contrast / chars1['contrast'] if chars1['contrast'] > 0 else 1.0
    contrast_factor2 = target_contrast / chars2['contrast'] if chars2['contrast'] > 0 else 1.0
    
    contrast_factor1 = np.clip(contrast_factor1, 0.5, 2.0)
    contrast_factor2 = np.clip(contrast_factor2, 0.5, 2.0)
    
    result_low_img = contrast_enhancer1.enhance(contrast_factor1)
    result_top_img = contrast_enhancer2.enhance(contrast_factor2)
    
    return result_low_img, result_top_img

def edge_enhance_images(low_img, top_img):
    """Enhance edges to create ASCII-art-like effect"""
    # Apply edge enhancement filter
    edge_filter = ImageFilter.EDGE_ENHANCE_MORE
    
    result_low_img = low_img.filter(edge_filter)
    result_top_img = top_img.filter(edge_filter)
    
    # Slightly reduce saturation for ASCII-like feel
    sat_enhancer1 = ImageEnhance.Color(result_low_img)
    sat_enhancer2 = ImageEnhance.Color(result_top_img)
    
    result_low_img = sat_enhancer1.enhance(0.8)
    result_top_img = sat_enhancer2.enhance(0.8)
    
    return result_low_img, result_top_img

def yin_yang_with_images(R=1.0, lower_image_path=None, top_image_path=None, unify_method='brightness_match', 
                        low_img_rotation=50, top_img_rotation=230, 
                        low_img_flip_horizontal=False, low_img_flip_vertical=False,
                        top_img_flip_horizontal=True, top_img_flip_vertical=False):
    """
    Create a yin-yang symbol using two different images as fill patterns.
    
    Parameters:
    R: Radius of the yin-yang
    lower_image_path: Path to first image (for yin half - left side)
    top_image_path: Path to second image (for yang half - right side)
    unify_method: Method to unify images ('histogram_match', 'color_balance', 'brightness_match', 'edge_enhance')
    low_img_rotation: Degrees to rotate lower_image (default: 110 - base 20° + 90° adjustment)
    top_img_rotation: Degrees to rotate top image (default: 245 - base 200° + 45° adjustment + horizontal flip) 
    low_img_flip_horizontal: Flip lower_image horizontally (default: False)
    low_img_flip_vertical: Flip lower_image vertically (default: False)
    top_img_flip_horizontal: Flip top image horizontally (default: True - creates mirror effect for yin-yang)
    top_img_flip_vertical: Flip top image vertically (default: False)
    """
    
    # Auto-find images if not specified
    if lower_image_path is None or top_image_path is None:
        available_images = find_available_images()
        if len(available_images) >= 2:
            lower_image_path = available_images[0]
            top_image_path = available_images[1]
            print(f"Using found images: {lower_image_path} and {top_image_path}")
        else:
            print("Not enough images found, creating sample images...")
            sample_images = create_sample_images()
            lower_image_path = sample_images[0]
            top_image_path = sample_images[1]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Load and prepare images
    try:
        low_img = Image.open(lower_image_path).convert('RGB')
        top_img = Image.open(top_image_path).convert('RGB')
        print(f"Loaded images: {lower_image_path} and {top_image_path}")
        
        # Analyze images before unification
        chars1 = analyze_image_characteristics(low_img)
        chars2 = analyze_image_characteristics(top_img)
        print(f"Image 1 - Brightness: {chars1['brightness']:.1f}, Contrast: {chars1['contrast']:.1f}")
        print(f"Image 2 - Brightness: {chars2['brightness']:.1f}, Contrast: {chars2['contrast']:.1f}")
        
        # Unify images using ASCII art techniques
        print(f"Applying unification method: {unify_method}")
        low_img, top_img = unify_images(low_img, top_img, unify_method)
        
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
    low_img_resized = low_img.resize((400, 400))
    top_img_resized = top_img.resize((400, 400))
    
    # Apply customizable transformations to lower_image (yin)
    if low_img_flip_horizontal:
        low_img_resized = low_img_resized.transpose(Image.FLIP_LEFT_RIGHT)
    if low_img_flip_vertical:
        low_img_resized = low_img_resized.transpose(Image.FLIP_TOP_lower)
    if low_img_rotation != 0:
        low_img_resized = low_img_resized.rotate(-low_img_rotation, expand=True)  # Negative for clockwise
    
    # Apply customizable transformations to top image (yang)
    if top_img_flip_horizontal:
        top_img_resized = top_img_resized.transpose(Image.FLIP_LEFT_RIGHT)
    if top_img_flip_vertical:
        top_img_resized = top_img_resized.transpose(Image.FLIP_TOP_lower)
    if top_img_rotation != 0:
        top_img_resized = top_img_resized.rotate(-top_img_rotation, expand=True)  # Negative for clockwise
    
    # Resize back to 400x400 after rotation (in case expand=True changed size)
    low_img_resized = low_img_resized.resize((400, 400))
    top_img_resized = top_img_resized.resize((400, 400))
    
    low_img_array = np.array(low_img_resized)
    top_img_array = np.array(top_img_resized)
    
    # Fill regions with appropriate images
    for i in range(400):
        for j in range(400):
            if outside_circle[i, j]:
                # Outside the main circle - white background
                result_img[i, j] = [255, 255, 255]
            elif upper_eye[i, j]:
                # Upper eye - use top image (opposite of the region it's in)
                result_img[i, j] = top_img_array[i, j]
            elif lower_eye[i, j]:
                # Lower eye - use lower_image (opposite of the region it's in)
                result_img[i, j] = low_img_array[i, j]
            elif left_half[i, j] and not upper_circle[i, j]:
                # Left half minus the upper circle - use lower_image
                result_img[i, j] = low_img_array[i, j]
            elif not left_half[i, j] and not lower_circle[i, j]:
                # Right half minus the lower circle - use top image
                result_img[i, j] = top_img_array[i, j]
            elif upper_circle[i, j]:
                # Upper circle - use top image
                result_img[i, j] = top_img_array[i, j]
            elif lower_circle[i, j]:
                # Lower circle - use lower_image
                result_img[i, j] = low_img_array[i, j]
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
    
    # Save to output directory
    output_filename = get_next_output_filename(unify_method)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_filename}")
    
    # Show the image
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
            print("Name them 'lower_image.png' and 'top_image.png' or specify custom paths.")
    except Exception as e:
        print(f"Note: {e}")

# Run the yin-yang visualization
if __name__ == "__main__":
    import sys
    
    print("Yin-Yang Image Generator")
    print("=======================")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Command line mode - use arguments
        lower_image_path = sys.argv[1] if len(sys.argv) > 1 else None
        top_image_path = sys.argv[2] if len(sys.argv) > 2 else None
        method = sys.argv[3] if len(sys.argv) > 3 else 'brightness_match'
        
        print(f"Using command line args: {lower_image_path}, {top_image_path}, {method}")
        yin_yang_with_images(R=1.0, lower_image_path=lower_image_path, top_image_path=top_image_path, unify_method=method)
    else:
        # Auto mode - use defaults with first two available images and run all methods
        available_images = find_available_images()
        if len(available_images) >= 2:
            print(f"Found {len(available_images)} images. Using: {available_images[0]} and {available_images[1]}")
            print("Creating yin-yangs with all methods...")
            
            methods = ['brightness_match', 'histogram_match', 'color_balance', 'edge_enhance']
            for method in methods:
                print(f"\nGenerating with {method}...")
                yin_yang_with_images(R=1.0, lower_image_path=available_images[0], top_image_path=available_images[1], unify_method=method)
        else:
            print("Not enough images found. Creating sample images...")
            sample_images = create_sample_images()
            
            methods = ['brightness_match', 'histogram_match', 'color_balance', 'edge_enhance']
            for method in methods:
                print(f"\nGenerating with {method}...")
                yin_yang_with_images(R=1.0, lower_image_path=sample_images[0], top_image_path=sample_images[1], unify_method=method)
    
    print("\nUsage:")
    print("  Default mode: python yinyang.py (generates all 4 methods automatically)")
    print("  Custom mode:  python yinyang.py lower_image.png top_image.png [method]")
    print("  Methods: brightness_match, histogram_match, color_balance, edge_enhance")
    print("  Output files: yinyang1_brightness_match.png, yinyang1_histogram_match.png, etc.")
    
    print("\nTip: To use your own images, simply place any .jpg, .png, or other image files")
    print("in this directory and run the script again. The first two images found")
    print("will be used automatically!")
