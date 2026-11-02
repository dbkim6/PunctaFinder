## custom functions used in puncta_analysis_pipeline.ipynb

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import disk, opening, remove_small_objects
from skimage.measure import label
import PF_functions as PF
import os
import glob
import re

## use PF.PunctaFinder function to count puncta on cell bodies lebeled by tracer (FB or CTB - single channel image)
def analyze_puncta_image(output_file_path, egfp_file_path, csv_save_path, T_local, T_global, T_CV):
    print("Loading image stacks...")
    # 1. Load the filled blob shapes from the previous step
    filled_stack = tiff.imread(output_file_path)
    # 2. Load the matching EGFP channel stack
    egfp_stack = tiff.imread(egfp_file_path)
    
    # Safety checks for dimensions
    if filled_stack.ndim == 2:
        filled_stack = np.expand_dims(filled_stack, axis=0)
        egfp_stack = np.expand_dims(egfp_stack, axis=0)
        
    assert filled_stack.shape == egfp_stack.shape, "Image dimensions do not match!"
    
    num_slices = filled_stack.shape[0]
    puncta_counts_list = []
    
    # 5. Loop through all slices (Z-levels)
    for z in range(num_slices):
        slice_cells = filled_stack[z]
        slice_egfp = egfp_stack[z]
        
        # Identify unique cells in this specific slice (ignoring background 0)
        unique_cells = np.unique(slice_cells)
        unique_cells = unique_cells[unique_cells != 0]
        
        print(f"Processing slice {z+1}/{num_slices}... Found {len(unique_cells)} cells.")
        
        # Loop through each individual cell in the current slice
        for cell_id in unique_cells:
            # Step 1: Generate the binary mask for this specific cell
            cell_mask = (slice_cells == cell_id)
            cell_size = np.sum(cell_mask)

            # --- DEFENSIVE FIX 1: Ignore tiny noise artifacts/debris ---
            if cell_size < 10: 
                print(f"Slice {z+1}, Cell ID {cell_id}: Skipped (Too small: {cell_size} pixels).")
                continue
            
            # Step 2: Apply the cell mask to the EGFP channel
            # Pixels outside the cell are set to 0
            EGFP_mask = np.zeros_like(slice_egfp)
            EGFP_mask[cell_mask] = slice_egfp[cell_mask]

            # if > 30% of the cell area (by pixels) has intensity > 700 in EGFP_mask, skip the cell
            if np.sum(EGFP_mask > 700) > 0.3 * np.sum(cell_mask):
                print(f"Slice {z+1}, Cell ID {cell_id} area: {np.sum(cell_mask)} pixels, EGFP positive pixels: {np.sum(EGFP_mask > 700)}.")
                print(f"Slice {z+1}, Cell ID {cell_id}: Skipped due to >30% of pixels with intensity > 700.")
                continue

            else:
                # --- DIAGNOSTIC PRINT: Keep this to catch the hanging cell ---
                # print(f"Slice {z+1}, Cell ID {cell_id} ({cell_size} px) -> Entering PunctaFinder...", end="", flush=True)
                
                # If the script hangs, the last printed line will tell you exactly which cell broke it
                # result_df = PF.PunctaFinder(EGFP_mask, cell_mask, 5, 0, T_local, T_global, T_CV)
                # print(" Done.") # Will only print if PunctaFinder successfully returns
                
                # Step 3: Run your predefined PunctaFinder function
                # Expecting a 1-row DataFrame where column index 1 (the 2nd column) holds the count
                result_df = PF.PunctaFinder(EGFP_mask, cell_mask, 6, 0, T_local, T_global, T_CV)
                
                # Step 4: Extract the puncta value from the 2nd column (index 1) of the first row
                result = result_df.iloc[0, 1]
                zero_puncta_cells = 0
                if result is None:
                    print(f"Slice {z+1}, Cell ID {cell_id}: No puncta detected.")
                    zero_puncta_cells = zero_puncta_cells + 1
                    continue
                else:
                    puncta_value = len(result)
                    print(f"Slice {z+1}, Cell ID {cell_id}: Detected Puncta = {puncta_value}")
                
                # Record it to our master collector
                puncta_counts_list.append(puncta_value)
            
    # Step 4 & 6: Create a single-column DataFrame and save to CSV
    final_df = pd.DataFrame(puncta_counts_list) #, columns=["Detected_Puncta"]
    
    final_df.to_csv(csv_save_path, index=False)
    print(f"\nAnalysis complete! Total cells analyzed: {len(final_df)+zero_puncta_cells}. Total cells with detected puncta: {len(final_df)}. Cells skipped due to high EGFP intensity: {zero_puncta_cells}.")
    print(f"Results saved to: {csv_save_path}")

## PV+ cell body (tdTomato) counting: segmentation with overlap exclusion across Z-slices
def count_neurons_with_overlap_exclusion(input_path, output_csv_path, axon_max_width, min_soma_area):
    print(f"Loading fluorescence stack: {input_path}")
    stack = tiff.imread(input_path)
    
    if stack.ndim == 2:
        stack = np.expand_dims(stack, axis=0)
        
    num_slices = stack.shape[0]
    
    radius = int(np.ceil(axon_max_width / 2)) + 1
    selem = disk(radius)
    
    total_unique_cells = 0
    previous_slice_mask = None
    slice_counts = []

    print("\nProcessing slices and calculating overlap exclusions...")
    for z in range(num_slices):
        slice_data = stack[z]
        
        # Default tracking if the slice is completely empty
        if np.max(slice_data) == 0:
            slice_counts.append({"Slice": z + 1, "New_Cells_Discovered": 0})
            previous_slice_mask = np.zeros_like(slice_data, dtype=bool)
            continue
            
        # Step 1-4: Clean up image to isolate cell bodies (somas)
        thresh = threshold_otsu(slice_data)
        binary_mask = slice_data > thresh
        axon_free_mask = opening(binary_mask, selem)
        filled_somas = binary_fill_holes(axon_free_mask)
        clean_somas_current = remove_small_objects(filled_somas, min_size=min_soma_area)
        
        # Label each individual cell body on the CURRENT slice
        labeled_current, num_features = label(clean_somas_current, return_num=True)
        
        new_cells_in_slice = 0
        
        # Step 5: Check every detected cell for overlap with the previous slice
        for cell_idx in range(1, num_features + 1):
            cell_mask = (labeled_current == cell_idx)
            cell_area = np.sum(cell_mask)
            
            # If it's the very first slice, there is no previous data to overlap with
            if z == 0 or previous_slice_mask is None:
                new_cells_in_slice += 1
            else:
                # Calculate how many pixels of this cell overlap with any cell from the last slice
                overlap_pixels = np.sum(cell_mask & previous_slice_mask)
                overlap_ratio = overlap_pixels / cell_area
                
                # Exclude if overlap is less than 50%
                if overlap_ratio < 0.50:
                    new_cells_in_slice += 1
                    
        total_unique_cells += new_cells_in_slice
        slice_counts.append({
            "Slice": z + 1, 
            "New_Cells_Discovered": new_cells_in_slice
        })
        
        print(f" -> Slice {z+1}/{num_slices}: Counted {new_cells_in_slice} new cells (Excluded {num_features - new_cells_in_slice} continuations)")
        
        # Save current mask to compare against the next slice
        previous_slice_mask = clean_somas_current

    print(f"\n>> Grand Total Unique Neuronal Cell Bodies: {total_unique_cells} <<")

    image_PV_count = total_unique_cells

    return image_PV_count
    
    # # Save a simple per-slice breakdown to CSV
    # df = pd.DataFrame(slice_counts)
    # # Append the final total summary row at the bottom of the CSV
    # df = pd.concat([df, pd.DataFrame([{"Slice": "TOTAL UNIQUE NEURONS", "New_Cells_Discovered": total_unique_cells}])], ignore_index=True)
    
    # df.to_csv(output_csv_path, index=False)
    # print(f"Summary saved to: {output_csv_path}") # summary csv not needed; edit so final count can be written directly into sypCountSheet

## compile ressults into .csv file that can be used in PV_Syp_AnlysesNotebook
def compile_analysis_data(directory_path, csv_file_path, PV_count):
    """
    Consolidates cell puncta data arrays and global neuron counts into a master Excel file
    using only Pandas.
    """
   # 1. Load the existing CSV file if it exists, otherwise start with a blank DataFrame
    if os.path.exists(csv_file_path):
        master_df = pd.read_csv(csv_file_path)
    else:
        master_df = pd.DataFrame(columns=["Image Name", "Slide #", "Section #", "# of Synapses", "Cell Type", "# of Neurons"])

    csv_pattern = os.path.join(directory_path, "*_puncta_counts.csv")
    csv_files = glob.glob(csv_pattern)
    csv_files.sort(reverse=True) 

    new_rows = []

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        
        match = re.search(r"^(.*)_(FB|CTB)_puncta_counts\.csv$", filename)
        if not match:
            continue  
            
        base_image_name = match.group(1)      
        tracer_type = match.group(2)          
        cell_type_string = f"{tracer_type}_subtype"
        
        data_df = pd.read_csv(csv_file, header=None)
        puncta_counts = data_df[0].tolist()
        
        for val in puncta_counts:
            new_rows.append({
                "Image Name": base_image_name,
                "Slide #": None,       
                "Section #": None,
                "# of Synapses": int(val),
                "Cell Type": cell_type_string,
                "Basket Cells": None   
            })

    if new_rows:
        new_data_df = pd.DataFrame(new_rows)
        new_data_df.at[0, "Basket Cells"] = int(PV_count)
        master_df = pd.concat([master_df, new_data_df], ignore_index=True)

    # Save out as a flat CSV file into the same directory as the puncta count files
    csv_file_path = os.path.join(directory_path, csv_file_path)
    master_df.to_csv(csv_file_path, index=False)