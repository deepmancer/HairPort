import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_landmarks_on_mesh(mesh_path, landmarks_3d_path, output_path=None):
    mesh = trimesh.load(mesh_path, force='mesh')
    landmarks = np.load(landmarks_3d_path)
    
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], 
                c='gray', alpha=0.1, s=0.1)
    ax1.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], 
                c='red', s=50, marker='o')
    ax1.set_title('Landmarks on Mesh (Front)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], 
                c='gray', alpha=0.1, s=0.1)
    ax2.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], 
                c='red', s=50, marker='o')
    ax2.view_init(elev=0, azim=90)
    ax2.set_title('Landmarks on Mesh (Side)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2], 
                c='gray', alpha=0.1, s=0.1)
    ax3.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], 
                c='red', s=50, marker='o')
    ax3.view_init(elev=90, azim=0)
    ax3.set_title('Landmarks on Mesh (Top)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def export_landmarks_to_obj(landmarks_3d_path, output_path):
    landmarks = np.load(landmarks_3d_path)
    num_landmarks = len(landmarks)
    
    with open(output_path, 'w') as f:
        f.write(f"# {num_landmarks} Facial Landmarks\n")
        for i, lmk in enumerate(landmarks):
            f.write(f"v {lmk[0]:.6f} {lmk[1]:.6f} {lmk[2]:.6f}\n")
        
        # Only add connectivity lines if we have the standard 68-landmark format
        if num_landmarks == 68:
            # Jaw line (0-16)
            jaw_indices = list(range(1, 18))
            for i in range(len(jaw_indices) - 1):
                f.write(f"l {jaw_indices[i]} {jaw_indices[i+1]}\n")
            
            # Right eyebrow (17-21)
            right_eyebrow = list(range(18, 23))
            for i in range(len(right_eyebrow) - 1):
                f.write(f"l {right_eyebrow[i]} {right_eyebrow[i+1]}\n")
            
            # Left eyebrow (22-26)
            left_eyebrow = list(range(23, 28))
            for i in range(len(left_eyebrow) - 1):
                f.write(f"l {left_eyebrow[i]} {left_eyebrow[i+1]}\n")
            
            # Nose bridge (27-30)
            nose_bridge = list(range(28, 32))
            for i in range(len(nose_bridge) - 1):
                f.write(f"l {nose_bridge[i]} {nose_bridge[i+1]}\n")
            
            # Nose bottom (31-35)
            nose_bottom = [32, 33, 34, 35, 36, 32]
            for i in range(len(nose_bottom) - 1):
                f.write(f"l {nose_bottom[i]} {nose_bottom[i+1]}\n")
            
            # Right eye (36-41)
            right_eye = [37, 38, 39, 40, 41, 42, 37]
            for i in range(len(right_eye) - 1):
                f.write(f"l {right_eye[i]} {right_eye[i+1]}\n")
            
            # Left eye (42-47)
            left_eye = [43, 44, 45, 46, 47, 48, 43]
            for i in range(len(left_eye) - 1):
                f.write(f"l {left_eye[i]} {left_eye[i+1]}\n")
            
            # Outer lip (48-59)
            outer_lip = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 49]
            for i in range(len(outer_lip) - 1):
                f.write(f"l {outer_lip[i]} {outer_lip[i+1]}\n")
            
            # Inner lip (60-67)
            inner_lip = [61, 62, 63, 64, 65, 66, 67, 68, 61]
            for i in range(len(inner_lip) - 1):
                f.write(f"l {inner_lip[i]} {inner_lip[i+1]}\n")
        else:
            # For non-standard landmark counts, just create a simple line strip
            for i in range(1, num_landmarks):
                f.write(f"l {i} {i+1}\n")
    
    print(f"Landmarks exported to OBJ: {output_path} ({num_landmarks} landmarks)")


def print_landmark_statistics(landmarks_3d_path):
    landmarks = np.load(landmarks_3d_path)
    
    print("\n=== Landmark Statistics ===")
    print(f"Number of landmarks: {len(landmarks)}")
    print(f"Shape: {landmarks.shape}")
    print(f"\nBounding box:")
    print(f"  X: [{landmarks[:, 0].min():.4f}, {landmarks[:, 0].max():.4f}]")
    print(f"  Y: [{landmarks[:, 1].min():.4f}, {landmarks[:, 1].max():.4f}]")
    print(f"  Z: [{landmarks[:, 2].min():.4f}, {landmarks[:, 2].max():.4f}]")
    print(f"\nCentroid: [{landmarks[:, 0].mean():.4f}, {landmarks[:, 1].mean():.4f}, {landmarks[:, 2].mean():.4f}]")
    
    distances = []
    for i in range(len(landmarks) - 1):
        dist = np.linalg.norm(landmarks[i + 1] - landmarks[i])
        distances.append(dist)
    
    print(f"\nAverage inter-landmark distance: {np.mean(distances):.4f}")
    print(f"Max inter-landmark distance: {np.max(distances):.4f}")
    print(f"Min inter-landmark distance: {np.min(distances):.4f}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize 3D landmarks')
    parser.add_argument('--mesh_path', type=str, required=True)
    parser.add_argument('--landmarks_path', type=str, required=True)
    parser.add_argument('--output_viz', type=str, default='landmarks_visualization.png')
    parser.add_argument('--output_obj', type=str, default='landmarks.obj')
    
    args = parser.parse_args()
    
    print_landmark_statistics(args.landmarks_path)
    
    visualize_landmarks_on_mesh(
        args.mesh_path,
        args.landmarks_path,
        args.output_viz
    )
    
    export_landmarks_to_obj(
        args.landmarks_path,
        args.output_obj
    )
