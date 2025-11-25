"""
Configuration loader for Yin-Yang Image Converter
Reads settings from config.ini file
"""

import configparser
import os
from pathlib import Path

class YinYangConfig:
    def __init__(self, config_file="config.ini"):
        self.config = configparser.ConfigParser()
        
        # Set default values
        self.set_defaults()
        
        # Load config file if it exists
        if os.path.exists(config_file):
            self.config.read(config_file)
        else:
            print(f"Config file {config_file} not found, using defaults")
    
    def set_defaults(self):
        """Set default configuration values"""
        self.config['images'] = {
            'image1_path': '',
            'image2_path': '', 
            'bottomimage_directory': 'bottomimage',
            'topimage_directory': 'topimage'
        }
        
        self.config['transformations'] = {
            'image1_rotation': '65',
            'image2_rotation': '245',
            'image1_flip_horizontal': 'false',
            'image1_flip_vertical': 'false', 
            'image2_flip_horizontal': 'true',
            'image2_flip_vertical': 'false'
        }
        
        self.config['processing'] = {
            'methods': 'brightness_match, histogram_match, color_balance, edge_enhance',
            'generate_all_methods': 'true',
            'resolution': '400',
            'radius': '1.0'
        }
        
        self.config['output'] = {
            'output_directory': '../output',
            'dpi': '300', 
            'show_images': 'true',
            'filename_prefix': 'yinyang'
        }
    
    def get_image_paths(self):
        """Get image paths from config"""
        img1 = self.config['images']['image1_path'].strip()
        img2 = self.config['images']['image2_path'].strip()
        
        if img1 and img2:
            return img1, img2
        return None, None
    
    def get_image_directories(self):
        """Get bottom and top image directories"""
        return {
            'bottomimage': self.config['images']['bottomimage_directory'],
            'topimage': self.config['images']['topimage_directory']
        }
    
    def get_transformations(self):
        """Get transformation settings"""
        return {
            'img1_rotation': int(self.config['transformations']['image1_rotation']),
            'img2_rotation': int(self.config['transformations']['image2_rotation']),
            'img1_flip_horizontal': self.config.getboolean('transformations', 'image1_flip_horizontal'),
            'img1_flip_vertical': self.config.getboolean('transformations', 'image1_flip_vertical'),
            'img2_flip_horizontal': self.config.getboolean('transformations', 'image2_flip_horizontal'), 
            'img2_flip_vertical': self.config.getboolean('transformations', 'image2_flip_vertical')
        }
    
    def get_methods(self):
        """Get processing methods"""
        methods_str = self.config['processing']['methods']
        methods = [m.strip() for m in methods_str.split(',')]
        generate_all = self.config.getboolean('processing', 'generate_all_methods')
        
        if not generate_all:
            methods = [methods[0]]  # Use only first method
            
        return methods
    
    def get_processing_settings(self):
        """Get processing settings"""
        return {
            'resolution': int(self.config['processing']['resolution']),
            'radius': float(self.config['processing']['radius'])
        }
    
    def get_output_settings(self):
        """Get output settings"""
        return {
            'output_directory': self.config['output']['output_directory'],
            'dpi': int(self.config['output']['dpi']),
            'show_images': self.config.getboolean('output', 'show_images'),
            'filename_prefix': self.config['output']['filename_prefix']
        }