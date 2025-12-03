import numpy as np
import tifffile
from pathlib import Path
import asyncio
import matplotlib.pyplot as plt

class PixelMapGenerator:
    """Class for generating and loading pixel maps from TIFF stacks."""
    
    @staticmethod
    async def generate_or_load(stack_path, map_path, progress_callback=None):
        """
        Generate or load a pixel map.
        
        Args:
            stack_path (Path): Path to the TIFF stack file
            map_path (Path): Path to save/load the pixel map
            progress_callback (callable): Callback function for progress updates
            
        Returns:
            np.ndarray: The pixel map
        """
        
        # Check if map already exists
        if map_path.exists():
            if progress_callback:
                await progress_callback(50.0, "Loading existing pixel map...")
            
            try:
                pixel_map = np.load(map_path)
                if progress_callback:
                    await progress_callback(100.0, "Pixel map loaded successfully")
                return pixel_map
            except Exception as e:
                if progress_callback:
                    await progress_callback(0.0, f"Failed to load existing map: {str(e)}")
                # Fall through to generation
        
        # Generate new pixel map
        if progress_callback:
            await progress_callback(10.0, "Loading TIFF stack...")
        
        try:
            # Load the TIFF stack
            stack = tifffile.imread(stack_path)
            
            if progress_callback:
                await progress_callback(30.0, "Processing stack...")
            
            # Determine baseline and measurement periods based on stack length
            stack_length = stack.shape[0]
            if stack_length == 1520:
                stim_start = 726
                baseline_range = (426, 716)  # 300 to 10 frames before stimulation
                measurement_range = (743, 1033)  # 10 to 300 frames after stimulation
            elif stack_length == 2890:
                stim_start = 1381
                baseline_range = (1081, 1371)  # 300 to 10 frames before stimulation
                measurement_range = (1398, 1688)  # 10 to 300 frames after stimulation
            else:
                raise ValueError(f"Unsupported stack length: {stack_length}")
            
            if progress_callback:
                await progress_callback(50.0, f"Using baseline frames {baseline_range[0]}-{baseline_range[1]} and measurement frames {measurement_range[0]}-{measurement_range[1]}...")
            
            # Calculate baseline for each pixel
            baseline = np.mean(stack[baseline_range[0]:baseline_range[1]], axis=0)
            
            if progress_callback:
                await progress_callback(70.0, "Calculating dF/F for measurement period...")
            
            # Calculate dF/F for measurement period only
            measurement_frames = stack[measurement_range[0]:measurement_range[1]]
            dff_measurement = (measurement_frames - baseline) / baseline
            
            if progress_callback:
                await progress_callback(90.0, "Computing pixel map...")
            
            # Take mean dF/F across measurement frames for each pixel
            pixel_map = np.mean(dff_measurement, axis=0)
            
            # Save the generated map
            np.save(map_path, pixel_map)
            
            if progress_callback:
                await progress_callback(95.0, "Saving pixel map visualization...")
            
            # Save PNG visualization
            png_path = map_path.with_suffix('.png')
            PixelMapGenerator._save_pixel_map_png(pixel_map, png_path)
            
            if progress_callback:
                await progress_callback(100.0, "Pixel map generated and saved")
            
            return pixel_map
            
        except Exception as e:
            if progress_callback:
                await progress_callback(0.0, f"Error generating pixel map: {str(e)}")
            raise e
    
    @staticmethod
    def _save_pixel_map_png(pixel_map, png_path):
        """
        Save a PNG visualization of the pixel map with jet colormap and colorbar.
        
        Args:
            pixel_map (np.ndarray): The pixel map data
            png_path (Path): Path where to save the PNG file
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Display the pixel map with jet colormap
        im = ax.imshow(pixel_map, cmap='jet', aspect='equal')
        
        # Add colorbar on the right side
        cbar = plt.colorbar(im, ax=ax, label='bPAC dF/F')
        
        # Set title and remove axis ticks
        ax.set_title('bPAC dF/F Pixel Map', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Adjust layout to prevent clipping
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close figure to free memory 