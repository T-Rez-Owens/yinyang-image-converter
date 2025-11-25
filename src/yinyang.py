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

def yin_yang_with_images(R=1.0, image1_path=None, image2_path=None, unify_method='brightness_match', 
                        img1_rotation=50, img2_rotation=230, 
                        img1_flip_horizontal=False, img1_flip_vertical=False,
                        img2_flip_horizontal=True, img2_flip_vertical=False):
    """
    Create a yin-yang symbol using two different images as fill patterns.
    
    Parameters:
    R: Radius of the yin-yang
    image1_path: Path to first image (for yin half - left side)
    image2_path: Path to second image (for yang half - right side)
    unify_method: Method to unify images ('histogram_match', 'color_balance', 'brightness_match', 'edge_enhance')
    img1_rotation: Degrees to rotate image1 (default: 110 - base 20° + 90° adjustment)
    img2_rotation: Degrees to rotate image2 (default: 245 - base 200° + 45° adjustment + horizontal flip) 
    img1_flip_horizontal: Flip image1 horizontally (default: False)
    img1_flip_vertical: Flip image1 vertically (default: False)
    img2_flip_horizontal: Flip image2 horizontally (default: True - creates mirror effect for yin-yang)
    img2_flip_vertical: Flip image2 vertically (default: False)
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
    
    # Apply customizable transformations to image1 (yin)
    if img1_flip_horizontal:
        img1_resized = img1_resized.transpose(Image.FLIP_LEFT_RIGHT)
    if img1_flip_vertical:
        img1_resized = img1_resized.transpose(Image.FLIP_TOP_BOTTOM)
    if img1_rotation != 0:
        img1_resized = img1_resized.rotate(-img1_rotation, expand=True)  # Negative for clockwise
    
    # Apply customizable transformations to image2 (yang)
    if img2_flip_horizontal:
        img2_resized = img2_resized.transpose(Image.FLIP_LEFT_RIGHT)
    if img2_flip_vertical:
        img2_resized = img2_resized.transpose(Image.FLIP_TOP_BOTTOM)
    if img2_rotation != 0:
        img2_resized = img2_resized.rotate(-img2_rotation, expand=True)  # Negative for clockwise
    
    # Resize back to 400x400 after rotation (in case expand=True changed size)
    img1_resized = img1_resized.resize((400, 400))
    img2_resized = img2_resized.resize((400, 400))
    
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
            print("Name them 'image1.png' and 'image2.png' or specify custom paths.")
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
        image1_path = sys.argv[1] if len(sys.argv) > 1 else None
        image2_path = sys.argv[2] if len(sys.argv) > 2 else None
        method = sys.argv[3] if len(sys.argv) > 3 else 'brightness_match'
        
        print(f"Using command line args: {image1_path}, {image2_path}, {method}")
        yin_yang_with_images(R=1.0, image1_path=image1_path, image2_path=image2_path, unify_method=method)
    else:
        # Auto mode - use defaults with first two available images and run all methods
        available_images = find_available_images()
        if len(available_images) >= 2:
            print(f"Found {len(available_images)} images. Using: {available_images[0]} and {available_images[1]}")
            print("Creating yin-yangs with all methods...")
            
            methods = ['brightness_match', 'histogram_match', 'color_balance', 'edge_enhance']
            for method in methods:
                print(f"\nGenerating with {method}...")
                yin_yang_with_images(R=1.0, image1_path=available_images[0], image2_path=available_images[1], unify_method=method)
        else:
            print("Not enough images found. Creating sample images...")
            sample_images = create_sample_images()
            
            methods = ['brightness_match', 'histogram_match', 'color_balance', 'edge_enhance']
            for method in methods:
                print(f"\nGenerating with {method}...")
                yin_yang_with_images(R=1.0, image1_path=sample_images[0], image2_path=sample_images[1], unify_method=method)
    
    print("\nUsage:")
    print("  Default mode: python yinyang.py (generates all 4 methods automatically)")
    print("  Custom mode:  python yinyang.py image1.png image2.png [method]")
    print("  Methods: brightness_match, histogram_match, color_balance, edge_enhance")
    print("  Output files: yinyang1_brightness_match.png, yinyang1_histogram_match.png, etc.")
    
    print("\nTip: To use your own images, simply place any .jpg, .png, or other image files")
    print("in this directory and run the script again. The first two images found")
    print("will be used automatically!")
