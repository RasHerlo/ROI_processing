import numpy as np
import tifffile
from pathlib import Path
import asyncio

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
            
            # Calculate dF/F for each pixel
            # Assuming first 10% of frames are baseline
            baseline_frames = int(stack.shape[0] * 0.1)
            baseline = np.mean(stack[:baseline_frames], axis=0)
            
            if progress_callback:
                await progress_callback(60.0, "Calculating dF/F...")
            
            # Calculate dF/F for each frame
            dff_stack = (stack - baseline) / baseline
            
            if progress_callback:
                await progress_callback(80.0, "Computing pixel map...")
            
            # Take mean dF/F across all frames for each pixel
            pixel_map = np.mean(dff_stack, axis=0)
            
            # Save the generated map
            np.save(map_path, pixel_map)
            
            if progress_callback:
                await progress_callback(100.0, "Pixel map generated and saved")
            
            return pixel_map
            
        except Exception as e:
            if progress_callback:
                await progress_callback(0.0, f"Error generating pixel map: {str(e)}")
            raise e 