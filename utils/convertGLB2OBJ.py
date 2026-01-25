import trimesh
import numpy as np


def convert_glb_to_obj(
    input_glb_path, output_obj_path, component_names=None, normalize=True
):
    """
    Convert GLB file to OBJ, with option to select specific components and normalize.
    """
    try:
        # Load the GLB file
        scene = trimesh.load(input_glb_path)

        # Debug: Print scene information
        print("\nScene Information:")
        print(f"Scene graph nodes: {list(scene.graph.nodes)}")
        print(f"Scene geometries: {list(scene.geometry.keys())}")
        print(
            f"Graph node names: {[scene.graph[node][1] for node in scene.graph.nodes_geometry]}"
        )

        # Get all available components
        available_components = list(scene.geometry.keys())
        print(f"\nAvailable components: {available_components}")

        # If no specific components are requested, use all components
        if component_names is None:
            component_names = available_components

        print(f"\nRequested components: {component_names}")

        # Create a list to store all meshes
        meshes = []

        # Debug: Print graph structure
        print("\nScene Graph Structure:")
        for node in scene.graph.nodes_geometry:
            print(f"Node: {node}")
            print(f"Transform: {scene.graph[node][0]}")
            print(f"Name: {scene.graph[node][1]}")

        for name in component_names:
            print(f"\nProcessing component: {name}")
            # Get the geometry
            if name not in scene.geometry:
                print(f"Warning: Component {name} not found in geometries")
                continue

            geometry = scene.geometry[name]
            print(f"Found geometry with {len(geometry.vertices)} vertices")

            # Get all instances of this geometry
            instances = [
                node
                for node in scene.graph.nodes_geometry
                if scene.graph[node][1] == name
            ]
            print(f"Found {len(instances)} instances of this geometry")

            if not instances:
                # Try direct geometry export if no instances found
                print("No instances found, adding geometry directly")
                meshes.append(geometry)
                continue

            for instance in instances:
                print(f"Processing instance: {instance}")
                # Get the transform for this instance
                transform = scene.graph[instance][0]

                # Create a copy of the geometry
                mesh_copy = geometry.copy()

                # Apply the transform
                mesh_copy.apply_transform(transform)

                # Add to our list of meshes
                meshes.append(mesh_copy)

        # Check if we have any meshes
        print(f"\nTotal meshes collected: {len(meshes)}")
        if not meshes:
            raise ValueError("No valid meshes found in the selected components")

        combined_mesh = trimesh.util.concatenate(meshes)

        # Normalize if requested
        if normalize:
            combined_mesh = normalize_mesh(combined_mesh)

        # Export to OBJ
        combined_mesh.export(output_obj_path, file_type="obj")
        print(f"\nSuccessfully exported to {output_path}")

        return True

    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False


def normalize_mesh(mesh):
    """
    Normalize a mesh to fit within a unit cube centered at origin
    """
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    mesh.apply_translation(-center)
    scale = 1.0 / np.max(mesh.bounds[1] - mesh.bounds[0])
    mesh.apply_scale(scale)
    return mesh


# Example usage
if __name__ == "__main__":
    input_path = "/localhome/mza143/Downloads/human_model/low_poly_jumpsuit.glb"
    output_path = "/localhome/mza143/Downloads/human_model/body_02.obj"

    selected_components = [
        "Clotch_up_Clotch_up_0",
        "Clotch_up_Hand_0",
        "Clotch_up_Material #106_0",
        "Clotch_up_Material #105_0",
        "Clotch_up_Tactical_belt_0",
        "Clotch_up_Material #108_0",
    ]

    convert_glb_to_obj(input_path, output_path, None, normalize=True)
