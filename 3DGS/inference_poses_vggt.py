import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import numpy as np
import torch
import argparse
import glob
from scipy.spatial.transform import Rotation
from PIL import Image


from ezexr import imread
import cv2

import sys
vggt_path = os.path.abspath("vggt")
sys.path.append(vggt_path)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

from plyfile import PlyData, PlyElement



def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--images_dir", type=str, default=None, help="Directory to panohdr, should be the train folder")    
    

    return parser

def resize_hdr(img_hdr, size=512):
    H1, W1, _ = img_hdr.shape
    if size == 224:
        pass
    else:
        # resize long side to 512
        S = max(img_hdr.shape)
        if S > size:
            interp = cv2.INTER_LANCZOS4
        elif S <= size:
            interp = cv2.INTER_CUBIC
        new_size = (int(round(W1*size/S)), int(round(H1*size/S)))
        img_hdr = cv2.resize(img_hdr, new_size, interpolation=interp) 
    H, W, _ = img_hdr.shape
    W2 = W//16*16
    H2 = H//16*16
    img_hdr = cv2.resize(img_hdr, (W2,H2), interpolation=cv2.INTER_LINEAR)
    return img_hdr

def load_model(device=None):
    """Load and initialize the VGGT model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = VGGT.from_pretrained("facebook/VGGT-1B")

    # model = VGGT()
    # _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    # model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    
    model.eval()
    model = model.to(device)
    return model, device

def process_images(image_dir, model, device):
    """Process images with VGGT and return predictions."""
    image_names = glob.glob(os.path.join(image_dir, "*"))
    image_names = sorted([f for f in image_names if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(image_names)} images")
    
    if len(image_names) == 0:
        raise ValueError(f"No images found in {image_dir}")

    original_images = []
    for img_path in image_names:
        img = Image.open(img_path).convert('RGB')
        original_images.append(np.array(img))
    
    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")
    
    print("Running inference...")
    dtype = torch.float16
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    print("Converting pose encoding to camera parameters...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    
    print("Computing 3D points from depth maps...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points
    
    predictions["original_images"] = original_images
    
    S, H, W = world_points.shape[:3]
    normalized_images = np.zeros((S, H, W, 3), dtype=np.float32)
    
    for i, img in enumerate(original_images):
        resized_img = cv2.resize(img, (W, H))
        normalized_images[i] = resized_img / 255.0
    
    predictions["images"] = normalized_images
    
    return predictions, image_names, original_images

def extrinsic_to_colmap_format(extrinsics):
    """Convert extrinsic matrices to COLMAP format (quaternion + translation)."""
    num_cameras = extrinsics.shape[0]
    quaternions = []
    translations = []
    
    for i in range(num_cameras):
        # VGGT's extrinsic is camera-to-world (R|t) format
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        
        # Convert rotation matrix to quaternion
        # COLMAP quaternion format is [qw, qx, qy, qz]
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # scipy returns [x, y, z, w]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]
        
        quaternions.append(quat)
        translations.append(t)
    
    return np.array(quaternions), np.array(translations)

def hash_point(point, scale=100):
    """Create a hash for a 3D point by quantizing coordinates."""
    quantized = tuple(np.round(point * scale).astype(int))
    return hash(quantized)


def filter_and_prepare_points(predictions, conf_threshold, mask_sky=False, mask_black_bg=False, 
                             mask_white_bg=False, stride=1, prediction_mode="Depthmap and Camera Branch", imagesHDR=None):
    """
    Filter points based on confidence and prepare for COLMAP format.
    Implementation matches the conventions in the original VGGT code.
    """
    
    if prediction_mode == "Depthmap and Camera Branch":
        print("Using Depthmap and Camera Branch")
        pred_world_points = predictions["world_points_from_depth"]
    else:
        print("Using pts_cloud")
        pred_world_points = predictions["world_points"]
    pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    colors_rgb = predictions["images"] 
    
    S, H, W = pred_world_points.shape[:3]
    if colors_rgb.shape[:3] != (S, H, W):
        print(f"Reshaping colors_rgb from {colors_rgb.shape} to match {(S, H, W, 3)}")
        reshaped_colors = np.zeros((S, H, W, 3), dtype=np.float32)
        for i in range(S):
            if i < len(colors_rgb):
                reshaped_colors[i] = cv2.resize(colors_rgb[i], (W, H))
        colors_rgb = reshaped_colors
    
    colors_rgb = (colors_rgb * 255).astype(np.uint8)
    
    if imagesHDR is not None:
        colors_hdr = imagesHDR
        colors_hdr_flat = colors_hdr.reshape(-1, 3)
        
        
    vertices_3d = pred_world_points.reshape(-1, 3)
    conf = pred_world_points_conf.reshape(-1)
    colors_rgb_flat = colors_rgb.reshape(-1, 3)

    

    if len(conf) != len(colors_rgb_flat):
        print(f"WARNING: Shape mismatch between confidence ({len(conf)}) and colors ({len(colors_rgb_flat)})")
        min_size = min(len(conf), len(colors_rgb_flat))
        conf = conf[:min_size]
        vertices_3d = vertices_3d[:min_size]
        colors_rgb_flat = colors_rgb_flat[:min_size]
    
    if conf_threshold == 0.0:
        conf_thres_value = 0.0
    else:
        conf_thres_value = np.percentile(conf, conf_threshold)
    
    print(f"Using confidence threshold: {conf_threshold}% (value: {conf_thres_value:.4f})")
    conf_mask = (conf >= conf_thres_value) & (conf > 1e-5)
    
    if mask_black_bg:
        black_bg_mask = colors_rgb_flat.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask
    
    
    filtered_vertices = vertices_3d[conf_mask]
    filtered_colors = colors_rgb_flat[conf_mask]
    
    
    if len(filtered_vertices) == 0:
        print("Warning: No points remaining after filtering. Using default point.")
        filtered_vertices = np.array([[0, 0, 0]])
        filtered_colors = np.array([[200, 200, 200]])
    
    print(f"Filtered to {len(filtered_vertices)} points")
    
    points3D = []
    point_indices = {}
    image_points2D = [[] for _ in range(len(pred_world_points))]
    
    print(f"Preparing points for COLMAP format with stride {stride}...")
    
    new_pts_3d = []
    new_colors_rgb = []
    
    total_points = 0
    for img_idx in range(S):
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                flat_idx = img_idx * H * W + y * W + x
                
                if flat_idx >= len(conf):
                    continue
                
                if mask_black_bg and colors_rgb_flat[flat_idx].sum() < 16:
                    continue
                
                if conf[flat_idx] < conf_thres_value or conf[flat_idx] <= 1e-5:
                    continue
                
                point3D = vertices_3d[flat_idx]
                rgb = colors_rgb_flat[flat_idx]
                
                if imagesHDR is not None:
                    rgb = colors_hdr_flat[flat_idx]
                
                if not np.all(np.isfinite(point3D)):
                    continue
                
                new_pts_3d.append(point3D)
                new_colors_rgb.append(rgb)
                
                point_hash = hash_point(point3D, scale=100)
                
                if point_hash not in point_indices:
                    point_idx = len(points3D)
                    point_indices[point_hash] = point_idx
                    
                    point_entry = {
                        "id": point_idx,
                        "xyz": point3D,
                        "rgb": rgb,
                        "error": 1.0,
                        "track": [(img_idx, len(image_points2D[img_idx]))]
                    }
                    points3D.append(point_entry)
                    total_points += 1
                else:
                    point_idx = point_indices[point_hash]
                    points3D[point_idx]["track"].append((img_idx, len(image_points2D[img_idx])))
                
                image_points2D[img_idx].append((x, y, point_indices[point_hash]))
    
    print(f"Prepared {len(points3D)} 3D points with {sum(len(pts) for pts in image_points2D)} observations for COLMAP")
    
    return new_pts_3d, new_colors_rgb, image_points2D
    

def write_colmap_cameras_txt(file_path, intrinsics, image_width, image_height, orig_width, orig_height):
    """Write camera intrinsics to COLMAP cameras.txt format."""
    with open(file_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(intrinsics)}\n")
        
        
        print(f"Image size: {image_width}x{image_height}, Original size: {orig_width}x{orig_height}")


        for i, intrinsic in enumerate(intrinsics):
            camera_id = i + 1  # COLMAP uses 1-indexed camera IDs
            model = "PINHOLE" 
            factorx = orig_width / 2 / intrinsic[0, 2]
            factory = orig_height / 2 / intrinsic[1, 2]
            
            fx = intrinsic[0, 0] * factorx
            fy = intrinsic[1, 1] * factory
            cx = orig_width / 2
            cy = orig_height / 2
            
            f.write(f"{camera_id} {model} {orig_width} {orig_height} {fx} {fy} {cx} {cy}\n")

def write_colmap_images_txt(file_path, quaternions, translations, image_points2D, image_names):
    """Write camera poses and keypoints to COLMAP images.txt format."""
    with open(file_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        num_points = sum(len(points) for points in image_points2D)
        avg_points = num_points / len(image_points2D) if image_points2D else 0
        f.write(f"# Number of images: {len(quaternions)}, mean observations per image: {avg_points:.1f}\n")
        
        for i in range(len(quaternions)):
            image_id = i + 1 
            camera_id = i + 1  
          
            qw, qx, qy, qz = quaternions[i]
            tx, ty, tz = translations[i]
            
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {os.path.basename(image_names[i])}\n")
            
            points_line = " ".join([f"{x} {y} {point3d_id+1}" for x, y, point3d_id in image_points2D[i]])
            f.write(f"{points_line}\n")
            
def storePlyHDR(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    
def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

if __name__ == "__main__":
    parser = get_args_parser()
    args, _ = parser.parse_known_args()

    img_folder_path = os.path.join(args.images_dir, 'ldr_rgb')
    train_img_list = sorted(os.listdir(img_folder_path))
    device = args.device
    
    output_colmap_path=img_folder_path.replace("ldr_rgb", "sparse/0")
    os.makedirs(output_colmap_path, exist_ok=True)
    
    model, device = load_model(device)
    model.float()
    
    
    predictions, image_names, original_images = process_images(img_folder_path, model, device)
    
    print("Converting camera parameters to COLMAP format...")
    quaternions, translations = extrinsic_to_colmap_format(predictions["extrinsic"])
    
    height, width = predictions["depth"].shape[1:3]
    #load HDR images
    img_hdr_folder_path = os.path.join(args.images_dir, 'hdr')
    imgs_hdr = []
    for img_file in image_names:
        img_hdr_file = os.path.basename(img_file)[:-4] + ".exr"
        img_hdr_path = os.path.join(img_hdr_folder_path, img_hdr_file)
        img_hdr = imread(img_hdr_path).astype(np.float32)
        img_hdr = cv2.resize(img_hdr, (width,height), interpolation=cv2.INTER_CUBIC)
        imgs_hdr.append(img_hdr)
    images_HDR = np.array(imgs_hdr)
    
    conf_threshold = 0
    stride = 2
    print(f"Filtering points with confidence threshold {conf_threshold}% and stride {stride}...")
    new_pts_3d, new_colors_rgb, image_points2D = filter_and_prepare_points(
        predictions, 
        conf_threshold, 
        mask_sky=False, 
        mask_black_bg=True,
        mask_white_bg=False,
        stride=stride,
        prediction_mode="Depthmap and Camera Branch",
        imagesHDR = images_HDR
    )
    
    height, width = predictions["depth"].shape[1:3]
    
    write_colmap_cameras_txt(
        os.path.join(output_colmap_path, "cameras.txt"), 
        predictions["intrinsic"], width, height, original_images[0].shape[1], original_images[0].shape[0])
    write_colmap_images_txt(
        os.path.join(output_colmap_path, "images.txt"), 
        quaternions, translations, image_points2D, image_names)
    
    
    new_pts_3d = np.array(new_pts_3d).astype(np.float32)
    new_colors_rgb = np.array(new_colors_rgb).astype(np.float32)
    
    storePlyHDR(os.path.join(output_colmap_path, "points3D.ply"), new_pts_3d, new_colors_rgb)
