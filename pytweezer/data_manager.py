import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tifffile

DATA_DIR = "C:\\Users\\tweez\\Box\\Working folder\\caftweezers\\Data\\"

@dataclass
class ExperimentalData:
    """A structured container for experimental dataset contents."""
    info: str = ""
    pickles: Dict[str, Any] = field(default_factory=dict)
    images: Dict[str, List[np.ndarray]] = field(default_factory=dict)

def _get_path_from_date(date_folder: str) -> Path:
    """
    Helper function to convert a date string like '22Aug26' into the 
    new nested directory structure: DATA_DIR / '2026' / 'Aug' / '22Aug26'
    """
    try:
        # Parse the string to extract year and month
        dt = datetime.strptime(date_folder, "%d%b%y")
        year_str = dt.strftime("%Y")
        month_str = dt.strftime("%b")
        return Path(DATA_DIR) / year_str / month_str / date_folder
    except ValueError:
        raise ValueError(f"Date folder '{date_folder}' does not match format 'DDMonYY' (e.g., '22Aug26')")


def list_datasets(date_folder: str) -> List[str]:
    """
    Lists and returns all dataset directories within a specific date folder.
    Automatically resolves the Year/Month/Date path structure.
    """
    date_path = _get_path_from_date(date_folder)
    
    if not date_path.exists() or not date_path.is_dir():
        print(f"Error: Date folder not found at {date_path}")
        return []
        
    datasets = [item.name for item in date_path.iterdir() if item.is_dir()]
    
    print(f"Datasets found in {date_path}:")
    if not datasets:
        print("  (No dataset folders found)")
    else:
        for ds in sorted(datasets):
            print(f"  - {ds}/")
            
    return sorted(datasets)


def load_dataset(date_folder: str, dataset_title: str) -> ExperimentalData:
    """
    Extracts text, pickle files, and .tif images from a specific dataset directory.
    Automatically resolves the Year/Month/Date path structure.
    """
    dataset_path = _get_path_from_date(date_folder) / dataset_title
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
    # --- Print directory contents ---
    print(f"Loading dataset: {dataset_title}")
            
    data = ExperimentalData()
    
    # 1. Extract and print the contextual .txt information
    txt_files = list(dataset_path.glob("*.txt"))
    if txt_files:
        with open(txt_files[0], 'r', encoding='utf-8') as f:
            data.info = f.read()
            print("\n--- Experimental Info ---\n")
            print(data.info.strip())
            print("\n" + "-" * 40)

    dirs = []
    files = []
    for item in dataset_path.iterdir():
        if item.is_dir():
            dirs.append(item.name)
        else:
            files.append(item.name)

    print("Contents:")
    if dirs:
        print("  Subdirectories:")
        for d in sorted(dirs):
            print(f"    - {d}/")
    if files:
        print("  Files:")
        for f in sorted(files):
            print(f"    - {f}")
    print("-" * 40)
    
    # 2. Extract data from all .pkl files (and .pickle for backwards compatibility)
    pickle_files = list(dataset_path.glob("*.pkl")) + list(dataset_path.glob("*.pickle"))
    for p_file in pickle_files:
        with open(p_file, 'rb') as f:
            data.pickles[p_file.name] = pickle.load(f)
            
    # 3. Extract .tif images from any subdirectories using tifffile
    for item in dataset_path.iterdir():
        if item.is_dir():
            tif_files = sorted(list(item.glob("*.tif")) + list(item.glob("*.tiff")))
            
            if tif_files:
                image_list = []
                for tif_file in tif_files:
                    img_array = tifffile.imread(str(tif_file))
                    image_list.append(img_array)
                    
                data.images[item.name] = image_list
                
    return data


def save_dataset(
    dataset_title: str, 
    context_info: str = "", 
    pickles: Dict[str, Any] = None, 
    images: Dict[str, List[np.ndarray]] = None
) -> None:
    """
    Saves text, pickle variables, and image arrays into a new dataset directory.
    Automatically generates the Year/Month/Date path structure for today.
    Prevents overwriting by appending a suffix to the title if needed.
    """
    if pickles is None:
        pickles = {}
    if images is None:
        images = {}
        
    now = datetime.now()
    year_str = now.strftime("%Y")
    month_str = now.strftime("%b")
    date_folder = now.strftime("%d%b%y")
    
    # Establish the base date directory, e.g. DATA_DIR/2026/Aug/22Aug26
    date_path = Path(DATA_DIR) / year_str / month_str / date_folder
    
    # Check for existing dataset and append suffix if necessary
    original_title = dataset_title
    counter = 1
    dataset_path = date_path / dataset_title
    
    while dataset_path.exists():
        dataset_title = f"{original_title}_{counter}"
        dataset_path = date_path / dataset_title
        counter += 1
        
    # This creates the year, month, date, and dataset folders all at once if they don't exist
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    if counter > 1:
        print(f"Dataset '{original_title}' already exists. Renaming to '{dataset_title}'.")
    
    # 1. Write the experiment_info.txt
    info_file = dataset_path / "experiment_info.txt"
    with open(info_file, "w", encoding='utf-8') as f:
        f.write(f"Dataset Title: {dataset_title}\n")
        f.write(f"Date: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(context_info)
        
    # 2. Save pickle variables with forced .pkl extension
    for filename, variable in pickles.items():
        if filename.endswith(".pickle"):
            filename = filename[:-7] + ".pkl"
        elif not filename.endswith(".pkl"):
            filename += ".pkl"
            
        with open(dataset_path / filename, 'wb') as f:
            pickle.dump(variable, f)
            
    # 3. Save image arrays using tifffile
    for folder_name, img_list in images.items():
        img_folder = dataset_path / folder_name
        img_folder.mkdir(exist_ok=True)
        
        for i, img_array in enumerate(img_list):
            img_path = img_folder / f"image_{i:03d}.tif"
            tifffile.imwrite(str(img_path), img_array)
            
    print(f"Successfully saved to {dataset_path}")