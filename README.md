# Yin-Yang Image Converter 🐉⚫⚪

Transform any two images into beautiful yin-yang symbols with advanced image processing techniques and intelligent preview systems.

## ✨ Features

- **🎯 Smart Preview Modes**: Two intelligent preview systems to find the perfect settings
- **⚙️ Configuration-Driven**: No terminal prompts - just edit `config.ini` and run  
- **🎨 Multiple Artistic Styles**: 4 different image unification methods
- **🔄 Rotation Preview**: See both images rotated through all angles simultaneously
- **📊 Method Comparison**: Compare all unification methods side-by-side
- **🖼️ Smart Image Processing**: Automatic brightness, contrast, and color balancing
- **🎛️ Customizable Transformations**: Independent rotation and flip controls for each image
- **📁 Dual Output System**: Previews in `temp_preview/`, final results in `output/`
- **⚡ High-Quality Output**: 300 DPI PNG files with descriptive naming

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/T-Rez-Owens/yinyang-image-converter.git
cd yinyang-image-converter
pip install -r requirements.txt
```

### 2. Add Your Images
- Drop your **lower/yin** image in: `src/lower_image/`
- Drop your **top/yang** image in: `src/top_image/`

### 3. Find Perfect Settings (Preview Mode)
```bash
cd src
# First, find the best rotations
python createYinYangFromImages.py  # Uses rotation_preview mode by default
```

### 4. Lock In Your Settings
Edit `src/config.ini` with your preferred angles:
```ini
lower_image_rotation = 90    # Your favorite lower rotation
top_image_rotation = 180     # Your favorite top rotation 
mode = methods               # Switch to methods mode
```

### 5. Generate Final Comparison
```bash
python createYinYangFromImages.py  # Now generates method comparisons + final result
```

**Result**: `temp_preview/` has all your options, `output/` has your final yin-yang!

## 🎨 Preview Modes & Output

### 🔄 Rotation Preview Mode
**temp_preview/** folder contains:
- `yinyang_both_rot_000.png` - Both images at 0°
- `yinyang_both_rot_045.png` - Both images at 45°  
- `yinyang_both_rot_090.png` - Both images at 90°
- ...up to 315° (8 total previews)

**output/** folder contains:
- Your final image with exact config settings

### 📊 Methods Mode  
**temp_preview/** folder contains:
- `yinyang_brightness_match.png` - Balanced brightness and contrast
- `yinyang_histogram_match.png` - Matched tonal distribution  
- `yinyang_color_balance.png` - Harmonized colors for unity
- `yinyang_edge_enhance.png` - Enhanced edges with artistic flair

**output/** folder contains:
- Your final image with chosen method and rotations

## ⚙️ Configuration

Edit `src/config.ini` to customize all settings:

### Image Settings
```ini
[images]
# Leave empty for auto-detection from lower_image/top_image folders
lower_image_path = 
top_image_path = 

lower_image_directory = lower_image
top_image_directory = top_image
```

### Transformations
```ini
[transformations]
# Rotation in degrees (positive = clockwise)
lower_image_rotation = 90       # Lower image (yin)
top_image_rotation = 180        # Top image (yang) 

# Flip settings for fine-tuning
lower_image_flip_horizontal = false
lower_image_flip_vertical = false
top_image_flip_horizontal = true    # Creates mirror effect
top_image_flip_vertical = false
```

### Processing Modes
```ini
[processing]
# Two intelligent modes:
mode = rotation_preview    # Find best rotations (generates 8 sync'd previews)
# mode = methods           # Compare unification methods (generates 4 method previews)

method = brightness_match  # Primary method for final output
rotation_increment = 45    # Preview angles: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

# Output quality
resolution = 400
radius = 1.0
```

### Output Settings
```ini
[output]
output_directory = ../output        # Final results go here
dpi = 300                          # High quality 
show_images = false                # Disabled for preview modes
filename_prefix = yinyang
```

## 📁 Project Structure

```
yinyang-image-converter/
├── src/                         # Source code
│   ├── lower_image/            # Your yin images (drag & drop here)
│   ├── top_image/              # Your yang images (drag & drop here)  
│   ├── config.ini              # Main configuration file
│   ├── config_loader.py        # Configuration management system
│   ├── createYinYangFromImages.py # Main script (recommended)
│   └── __pycache__/            # Python cache (auto-generated)
├── output/                      # Final yin-yang results
├── temp_preview/               # Preview comparisons (rotation/method options)
├── creationRelics/             # Legacy/obsolete files (excluded from analysis)
├── requirements.txt            # Python dependencies
└── README.md                   # This documentation
```

## 🎯 How It Works

### Smart Preview Workflow

#### 🔄 Rotation Preview Mode (`mode = rotation_preview`)
1. **Load Images**: Auto-detect from `lower_image/` and `top_image/` folders
2. **Generate Sync Previews**: Both images rotate together (0°, 45°, 90°...315°) 
3. **Save Previews**: 8 comparison images in `temp_preview/`
4. **Create Final**: Your config rotations (lower_image_rotation, top_image_rotation) in `output/`
5. **You Choose**: Pick best angles, update config, switch to methods mode

#### 📊 Methods Mode (`mode = methods`) 
1. **Load Images**: Same auto-detection system
2. **Apply Your Rotations**: Uses your chosen rotations from config
3. **Generate Method Previews**: All 4 unification methods in `temp_preview/`  
4. **Create Final**: Your chosen method in `output/`
5. **You Choose**: Best-looking unification method

### The Yin-Yang Process
1. **Geometric Precision**: Mathematical yin-yang topology with proper proportions
2. **Smart Transformations**: Independent rotation and flip controls per image  
3. **Image Unification**: Brightness, histogram, color, or edge enhancement
4. **Region Mapping**: Images mapped to yin/yang regions with complementary "eyes"
5. **High-Quality Export**: 300 DPI PNG with descriptive filenames

### Intelligent Defaults
- **Rotation Preview**: Both images rotate together for quick comparison
- **Methods Preview**: Your rotation settings + all unification methods  
- **Dual Output**: Previews for exploration, final result for production

## 🎨 Artistic Methods

### Brightness Match
Balances brightness and contrast between images for cohesive lighting.

### Histogram Match  
Matches the tonal distribution of one image to another for consistency.

### Color Balance
Harmonizes color saturation and hues for unified palette.

### Edge Enhance
Applies edge enhancement filters for artistic, sketch-like effects.

## 🛠️ Advanced Usage

### Two-Step Workflow
```bash
# Step 1: Find best rotations
python createYinYangFromImages.py  # rotation_preview mode
# Check temp_preview/, pick your favorite angles

# Step 2: Compare methods with your rotations  
# Edit config.ini: set rotations + mode = methods
python createYinYangFromImages.py  # methods mode
# Check temp_preview/, pick your favorite method
```

### Custom Image Paths
Edit `config.ini` for specific files:
```ini
[images]
lower_image_path = /path/to/your/yin/image.png
top_image_path = /path/to/your/yang/image.png
```

### Quick Single Method
```ini
[processing]
mode = methods
method = edge_enhance  # Just generate one method comparison
```

### Fine-Tune Rotation Steps
```ini
[processing]
mode = rotation_preview
rotation_increment = 30  # More preview options: 0°, 30°, 60°, 90°...
```

## 📸 Tips for Best Results

### 🖼️ Image Selection
- **High Contrast Images**: Work best for distinct yin-yang separation
- **Complementary Content**: Images that balance each other visually  
- **Square Aspect Ratio**: Optimal for circular yin-yang format
- **Sufficient Resolution**: 400px+ recommended for quality output

### 🔄 Using Preview Modes Effectively
- **Start with Rotation Preview**: Find angles where both images look good together
- **Use Methods Mode**: Compare all unification techniques with your chosen rotations
- **Iterate Quickly**: Edit config.ini and re-run to test different combinations
- **Check Both Folders**: `temp_preview/` for options, `output/` for final result

### ⚙️ Configuration Tips
- **rotation_increment = 30**: More preview options (12 instead of 8)
- **rotation_increment = 90**: Fewer options (4 instead of 8) for quick testing
- **Independent Rotations**: Lower and top can be completely different angles
- **Flip Combinations**: Try different flip settings for unique compositions

## 🔧 Development

### Main Script
```bash
cd src
python createYinYangFromImages.py  # Modern config-driven version
```

### Legacy Files
```bash
# These are in creationRelics/ (excluded from VS Code analysis)
python creationRelics/yinyang.py        # Original interactive version  
python creationRelics/yinyang2.py       # Development iteration
```

### Virtual Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### VS Code Configuration
The project includes `.vscode/settings.json` with:
- Excluded legacy files from analysis
- Python path optimization  
- Search exclusions for generated files

## 📋 Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Pillow (PIL)
- SciPy
- Requests

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🎯 About

Created for artists, designers, and anyone who wants to create unique yin-yang compositions from their own images. The project combines traditional yin-yang symbolism with modern image processing techniques.

**Perfect for:**
- Digital art projects
- Logo creation  
- Philosophical/spiritual artwork
- Creative photo manipulation
- Educational demonstrations

---

**Made with 🐍 Python and ❤️ for creative expression**