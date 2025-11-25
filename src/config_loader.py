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
            'mode': 'rotation_preview',
            'method': 'brightness_match',
            'rotation_increment': '45',
            'resolution': '400',
            'radius': '1.0'
        }
        
        self.config['output'] = {
            'output_directory': '../temp_preview',
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
    
    def get_processing_mode(self):
        """Get processing mode (methods or rotation_preview)"""
        return self.config.get('processing', 'mode', fallback='rotation_preview')
    
    def get_unification_method(self):
        """Get single unification method for rotation preview"""
        return self.config.get('processing', 'method', fallback='brightness_match')
    
    def get_rotation_increments(self):
        """Get rotation increments for preview mode"""
        increment = int(self.config.get('processing', 'rotation_increment', fallback='45'))
        return list(range(0, 360, increment))
    
    def get_processing_settings(self):
        """Get processing settings"""
        return {
            'resolution': int(self.config['processing']['resolution']),
            'radius': float(self.config['processing']['radius']),
            'mode': self.get_processing_mode(),
            'method': self.get_unification_method(),
            'rotations': self.get_rotation_increments()
        }
    
    def get_output_settings(self):
        """Get output settings"""
        return {
            'output_directory': self.config['output']['output_directory'],
            'dpi': int(self.config['output']['dpi']),
            'show_images': self.config.getboolean('output', 'show_images'),
            'filename_prefix': self.config['output']['filename_prefix']
        }