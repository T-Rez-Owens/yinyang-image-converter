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
            'lower_image_path': '',
            'top_image_path': '', 
            'lower_image_directory': 'lower_image',
            'top_image_directory': 'top_image'
        }
        
        self.config['transformations'] = {
            'lower_image_rotation': '65',
            'top_image_rotation': '245',
            'lower_image_flip_horizontal': 'false',
            'lower_image_flip_vertical': 'false', 
            'top_image_flip_horizontal': 'true',
            'top_image_flip_vertical': 'false'
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
        low_img = self.config['images']['lower_image_path'].strip()
        top_img = self.config['images']['top_image_path'].strip()
        
        if low_img and top_img:
            return low_img, top_img
        return None, None
    
    def get_image_directories(self):
        """Get lower and top image directories"""
        return {
            'lower_image': self.config['images']['lower_image_directory'],
            'top_image': self.config['images']['top_image_directory']
        }
    
    def get_transformations(self):
        """Get transformation settings"""
        return {
            'low_img_rotation': int(self.config['transformations']['lower_image_rotation']),
            'top_img_rotation': int(self.config['transformations']['top_image_rotation']),
            'low_img_flip_horizontal': self.config.getboolean('transformations', 'lower_image_flip_horizontal'),
            'low_img_flip_vertical': self.config.getboolean('transformations', 'lower_image_flip_vertical'),
            'top_img_flip_horizontal': self.config.getboolean('transformations', 'top_image_flip_horizontal'), 
            'top_img_flip_vertical': self.config.getboolean('transformations', 'top_image_flip_vertical')
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