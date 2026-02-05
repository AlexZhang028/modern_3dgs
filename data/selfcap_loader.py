import cv2
import numpy as np
import os

def read_selfcap_cameras(path):
    """
    Read SelfCap/EasyVolcap style extri.yml and intri.yml.
    Returns dictionaries for extrinsics and intrinsics.
    """
    extri_path = os.path.join(path, "extri.yml")
    intri_path = os.path.join(path, "intri.yml")

    # Try to find files in optimized/ if not in root
    if not os.path.exists(extri_path):
        extri_path = os.path.join(path, "optimized", "extri.yml")
    
    if not os.path.exists(intri_path):
        intri_path = os.path.join(path, "optimized", "intri.yml")

    if not os.path.exists(extri_path):
        raise FileNotFoundError(f"Could not find extri.yml in {path} or {os.path.join(path, 'optimized')}")
    
    # Read extri.yml manually line by line to get names if 'names' node is missing
    # or just iterate potential keys if we can find a way.
    # OpenCV FileStorage doesn't provide a generic 'keys()' method easily in python wrapper for root?
    # Actually we can't iterate root keys easily with cv2.FileStorage Python API unless we know them
    # OR we parse the YAML text manually to find keys.
    
    # Helper to parse YAML keys manually
    def parse_yaml_keys(file_path):
        keys = set()
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.endswith(':'):
                    key = line[:-1]
                    # We look for R_{name} or Rot_{name} or T_{name}
                    if key.startswith("R_") or key.startswith("Rot_") or key.startswith("T_"):
                        name = key.split('_')[1]
                        keys.add(name)
        return sorted(list(keys))

    names = []
    fs = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    names_node = fs.getNode("names")
    if not names_node.empty():
        for i in range(names_node.size()):
            names.append(names_node.at(i).string())
    else:
        # Fallback: Parse text to find keys
        names = parse_yaml_keys(extri_path)
    
    extrinsics = {}
    for name in names:
        # Order of preference: Rot (3x3), R (Rodrigues)
        Rot_node = fs.getNode(f"Rot_{name}")
        R_node = fs.getNode(f"R_{name}")
        T_node = fs.getNode(f"T_{name}")
        
        if T_node.empty():
             continue
        
        tvec = T_node.mat().flatten()
        
        if not Rot_node.empty():
            R = Rot_node.mat()
        elif not R_node.empty():
            rvec = R_node.mat().flatten()
            R, _ = cv2.Rodrigues(rvec)
        else:
            continue
        
        extrinsics[name] = {
            "R": R,
            "T": tvec
        }
    
    fs.release()

    # Read intrinsics
    intrinsics = {}
    if os.path.exists(intri_path):
        fs = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
        for name in names:
             K_node = fs.getNode(f"K_{name}")
             D_node = fs.getNode(f"D_{name}") 
             
             if not K_node.empty():
                 K = K_node.mat()
                 D = np.zeros(5)
                 if not D_node.empty():
                     D = D_node.mat().flatten()
                 
                 intrinsics[name] = {
                     "K": K,
                     "D": D,
                     "width": 0, # Will be filled from images
                     "height": 0
                 }
        fs.release()
    
    return names, extrinsics, intrinsics
