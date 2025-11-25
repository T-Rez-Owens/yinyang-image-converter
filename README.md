# Yin-Yang Image Converter 🐉⚫⚪

Transform any two images into beautiful yin-yang symbols with advanced image processing techniques.

## ✨ Features

- **Configuration-Driven**: No terminal prompts - just edit `config.ini` and run
- **Multiple Artistic Styles**: 4 different image unification methods
- **Smart Image Processing**: Automatic brightness, contrast, and color balancing
- **Customizable Transformations**: Rotation and flip controls for perfect composition
- **High-Quality Output**: 300 DPI PNG files with sequential numbering
- **User-Friendly Structure**: Simple drag-and-drop workflow

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

### 3. Generate
```bash
cd src
python yinyang_config.py
```

That's it! Check the `output/` folder for your yin-yang creations.

## 🎨 Output Examples

The generator creates 4 artistic variations:
- **`brightness_match`** - Balanced brightness and contrast
- **`histogram_match`** - Matched tonal distribution  
- **`color_balance`** - Harmonized colors for unity
- **`edge_enhance`** - Enhanced edges with artistic flair

## ⚙️ Configuration

Edit `src/config.ini` to customize:

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
lower_image_rotation = 65      # lower image (yin)
top_image_rotation = 245     # Top image (yang) 

# Flip settings
lower_image_flip_horizontal = false
lower_image_flip_vertical = false
top_image_flip_horizontal = true    # Creates mirror effect
top_image_flip_vertical = false
```

### Processing Options
```ini
[processing]
# Which methods to generate (comma-separated)
methods = brightness_match, histogram_match, color_balance, edge_enhance

# Generate all methods (true) or just first (false)
generate_all_methods = true

# Output quality
resolution = 400
radius = 1.0
```

### Output Settings
```ini
[output]
output_directory = ../output
dpi = 300
show_images = true          # Display after generation
filename_prefix = yinyang
```

## 📁 Project Structure

```
yinyang-image-converter/
├── src/                    # Source code
│   ├── lower_image/       # Your yin images (drag & drop here)
│   ├── top_image/         # Your yang images (drag & drop here)
│   ├── config.ini        # Configuration file
│   ├── config_loader.py  # Configuration management
│   ├── yinyang_config.py # Main script (recommended)
│   └── yinyang.py        # Legacy interactive version
├── output/               # Generated yin-yang images
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🎯 How It Works

### The Yin-Yang Process
1. **Load Images**: From `lower_image/` and `top_image/` folders
2. **Apply Transformations**: Rotation and flips based on config
3. **Unify Images**: Balance brightness, colors, or enhance edges
4. **Generate Topology**: Create precise yin-yang geometry
5. **Composite**: Map images to yin/yang regions with proper eyes
6. **Export**: High-quality PNG with method suffix

### Default Transformations
- **lower Image (Yin)**: 65° clockwise rotation
- **Top Image (Yang)**: 245° rotation + horizontal flip
- Creates perfect complementary opposition

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

### Command Line Mode
```bash
# Use specific images and method
python yinyang_config.py lower_image.png top_image.png brightness_match
```

### Custom Image Paths
Edit `config.ini`:
```ini
[images]
lower_image_path = /path/to/your/yin/image.png
top_image_path = /path/to/your/yang/image.png
```

### Single Method Generation
```ini
[processing]
methods = edge_enhance
generate_all_methods = false
```

## 📸 Tips for Best Results

- **High Contrast Images**: Work best for distinct yin-yang separation
- **Complementary Content**: Images that balance each other visually
- **Square Aspect Ratio**: Optimal for circular yin-yang format
- **Sufficient Resolution**: 400px+ recommended for quality output

## 🔧 Development

### Legacy Interactive Mode
```bash
cd src
python yinyang.py  # Interactive prompts (original version)
```

### Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

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