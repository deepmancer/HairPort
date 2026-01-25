#!/bin/bash

echo "========================================="
echo "3D Facial Landmark Estimation Framework"
echo "========================================="
echo ""

if [ "$1" == "test" ]; then
    echo "Running framework tests..."
    python fit_lmk/test_framework.py

elif [ "$1" == "example" ]; then
    echo "Creating example batch config..."
    python fit_lmk/batch_process.py --create_example
    echo ""
    echo "Edit batch_config.json and run:"
    echo "  bash run.sh batch"

elif [ "$1" == "batch" ]; then
    echo "Running batch processing..."
    if [ ! -f "batch_config.json" ]; then
        echo "Error: batch_config.json not found"
        echo "Run: bash run.sh example"
        exit 1
    fi
    python fit_lmk/batch_process.py --config batch_config.json --device cuda

elif [ "$1" == "single" ]; then
    if [ -z "$2" ]; then
        echo "Error: mesh path required"
        echo "Usage: bash run.sh single <mesh_path>"
        exit 1
    fi
    
    MESH_PATH="$2"
    OUTPUT_DIR="${3:-./output_landmarks}"
    
    echo "Processing single mesh: $MESH_PATH"
    echo "Output directory: $OUTPUT_DIR"
    
    python fit_lmk/run_standalone.py \
        --mesh_path "$MESH_PATH" \
        --cam_loc 0.0 -1.45 0.0 \
        --cam_rot 1.5708 0.0 0.0 \
        --ortho_scale 1.1 \
        --output_dir "$OUTPUT_DIR" \
        --resolution 1024 \
        --device cuda
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Generating visualization..."
        python fit_lmk/visualize.py \
            --mesh_path "$MESH_PATH" \
            --landmarks_path "$OUTPUT_DIR/landmarks_3d.npy" \
            --output_viz "$OUTPUT_DIR/visualization.png" \
            --output_obj "$OUTPUT_DIR/landmarks.obj"
    fi

elif [ "$1" == "visualize" ]; then
    if [ -z "$2" ] || [ -z "$3" ]; then
        echo "Error: mesh path and landmarks path required"
        echo "Usage: bash run.sh visualize <mesh_path> <landmarks_path>"
        exit 1
    fi
    
    MESH_PATH="$2"
    LANDMARKS_PATH="$3"
    OUTPUT_VIZ="${4:-visualization.png}"
    OUTPUT_OBJ="${5:-landmarks.obj}"
    
    echo "Visualizing landmarks..."
    python fit_lmk/visualize.py \
        --mesh_path "$MESH_PATH" \
        --landmarks_path "$LANDMARKS_PATH" \
        --output_viz "$OUTPUT_VIZ" \
        --output_obj "$OUTPUT_OBJ"

else
    echo "Usage:"
    echo "  bash run.sh test              - Run framework tests"
    echo "  bash run.sh example           - Create example batch config"
    echo "  bash run.sh batch             - Run batch processing"
    echo "  bash run.sh single <mesh>     - Process single mesh"
    echo "  bash run.sh visualize <mesh> <lmk> - Visualize landmarks"
    echo ""
    echo "Examples:"
    echo "  bash run.sh test"
    echo "  bash run.sh single mesh.glb"
    echo "  bash run.sh single mesh.glb ./my_output"
    echo "  bash run.sh visualize mesh.glb ./output/landmarks_3d.npy"
    exit 1
fi
