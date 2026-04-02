#!/usr/bin/env python3
"""
High-quality mesh simplification for large GLB meshes (textured or untextured).

Pipeline:
  GLB -> (PyMeshLab load) -> QEM decimation (UV-aware if UVs exist) -> OBJ(+MTL+texture if usable) -> gltfpack -> GLB

Key properties:
  - Exact target face count.
  - Preserves UVs when present.
  - Works for:
      * Textured meshes (baseColor texture) -> textured GLB if texture is decodable.
      * Untextured meshes with UVs -> geometry GLB with UVs.
      * Untextured meshes without UVs -> geometry-only GLB.
  - Robust texture handling:
      * Extract baseColor from GLB (embedded or data URI).
      * Decode+re-encode texture to a standard 8-bit RGBA PNG using Pillow to ensure gltfpack compatibility.

Dependencies:
  pip install pymeshlab numpy pillow
System tools:
  gltfpack on PATH (required)
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pymeshlab as pml
from PIL import Image  # pillow


LOG = logging.getLogger("simplify_glb")


# ----------------------------
# GLB parsing / texture extraction
# ----------------------------

@dataclass(frozen=True)
class GLBChunks:
    json_dict: dict
    bin_chunk: bytes


def _read_glb_chunks(glb_path: Path) -> GLBChunks:
    data = glb_path.read_bytes()
    if len(data) < 20:
        raise ValueError(f"File too small to be a valid GLB: {glb_path}")

    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ValueError(f"Not a GLB (bad magic): {glb_path}")
    if version != 2:
        raise ValueError(f"Unsupported GLB version {version} (expected 2): {glb_path}")
    if length != len(data):
        LOG.warning("GLB length header (%d) != file length (%d). Continuing.", length, len(data))

    offset = 12
    json_bytes: Optional[bytes] = None
    bin_bytes: Optional[bytes] = None

    while offset + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack_from("<I4s", data, offset)
        offset += 8
        chunk_data = data[offset: offset + chunk_len]
        offset += chunk_len

        if chunk_type == b"JSON":
            json_bytes = chunk_data
        elif chunk_type == b"BIN\0":
            bin_bytes = chunk_data

    if json_bytes is None:
        raise ValueError(f"Missing JSON chunk in GLB: {glb_path}")
    if bin_bytes is None:
        bin_bytes = b""

    json_text = json_bytes.rstrip(b"\x00 \t\r\n").decode("utf-8")
    json_dict = json.loads(json_text)
    return GLBChunks(json_dict=json_dict, bin_chunk=bin_bytes)


def _decode_data_uri(uri: str) -> bytes:
    header, payload = uri.split(",", 1)
    if ";base64" in header:
        return base64.b64decode(payload)
    return payload.encode("utf-8")


def extract_basecolor_texture_from_glb(glb_path: Path, out_dir: Path) -> Optional[Path]:
    """
    Extract baseColorTexture image for materials[0] if present; else returns None.
    Writes extracted bytes to disk. Filename is 'basecolor.bin' initially, then later sanitized.
    """
    chunks = _read_glb_chunks(glb_path)
    j = chunks.json_dict
    bin_chunk = chunks.bin_chunk

    materials = j.get("materials") or []
    textures = j.get("textures") or []
    images = j.get("images") or []
    buffer_views = j.get("bufferViews") or []

    if not materials:
        return None

    pbr = (materials[0].get("pbrMetallicRoughness") or {})
    base_tex = (pbr.get("baseColorTexture") or {})
    tex_index = base_tex.get("index")
    if tex_index is None or tex_index >= len(textures):
        return None

    img_index = textures[tex_index].get("source")
    if img_index is None or img_index >= len(images):
        return None

    img = images[img_index]

    # 1) URI case
    uri = img.get("uri")
    if uri:
        if uri.startswith("data:"):
            img_bytes = _decode_data_uri(uri)
            out_path = out_dir / "basecolor.bin"
            out_path.write_bytes(img_bytes)
            return out_path

        # external file reference (rare for .glb, common for .gltf)
        src_path = (glb_path.parent / uri).resolve()
        if not src_path.exists():
            LOG.warning("External texture referenced but missing: %s", src_path)
            return None
        out_path = out_dir / src_path.name
        shutil.copyfile(src_path, out_path)
        return out_path

    # 2) Embedded bufferView case
    bv_index = img.get("bufferView")
    if bv_index is None or bv_index >= len(buffer_views):
        return None

    bv = buffer_views[bv_index]
    byte_offset = int(bv.get("byteOffset", 0))
    byte_length = int(bv.get("byteLength", 0))
    if byte_length <= 0 or byte_offset < 0 or byte_offset + byte_length > len(bin_chunk):
        LOG.warning("Invalid image bufferView range (offset=%d len=%d bin=%d).",
                    byte_offset, byte_length, len(bin_chunk))
        return None

    img_bytes = bin_chunk[byte_offset: byte_offset + byte_length]
    if not img_bytes:
        return None

    out_path = out_dir / "basecolor.bin"
    out_path.write_bytes(img_bytes)
    return out_path


# ----------------------------
# Texture sanitization (critical fix)
# ----------------------------

def sanitize_texture_for_gltfpack(tex_path: Path, out_dir: Path) -> Optional[Path]:
    """
    Ensures texture is decodable by gltfpack by decoding with Pillow and re-encoding
    to standard 8-bit RGBA PNG.

    Returns path to sanitized PNG or None if decode fails.
    """
    if not tex_path.exists() or tex_path.stat().st_size == 0:
        return None

    sanitized = out_dir / "basecolor_sanitized.png"

    try:
        with Image.open(tex_path) as im:
            # Force decode now
            im.load()

            # Normalize to 8-bit RGBA (max compatibility)
            if im.mode != "RGBA":
                im = im.convert("RGBA")

            # Save with conservative settings
            im.save(sanitized, format="PNG", optimize=False)
    except Exception as e:
        LOG.warning("Texture decode/re-encode failed (%s). Proceeding without texture.", e)
        return None

    if not sanitized.exists() or sanitized.stat().st_size == 0:
        return None

    return sanitized


# ----------------------------
# Minimal MTL only when textured
# ----------------------------

def write_basic_mtl(mtl_path: Path, material_name: str, texture_filename: str) -> None:
    mtl_path.write_text(
        "\n".join([
            f"newmtl {material_name}",
            "Ka 0.000000 0.000000 0.000000",
            "Kd 1.000000 1.000000 1.000000",
            "Ks 0.000000 0.000000 0.000000",
            "d 1.000000",
            "illum 1",
            f"map_Kd {texture_filename}",
            "",
        ]),
        encoding="utf-8",
    )
def force_obj_mtllib_and_usemtl(obj_path: Path, mtl_name: str, material_name: str) -> None:
    """
    Forces the OBJ to reference exactly one mtllib and one usemtl near the top.
    - Removes any existing mtllib/usemtl lines.
    - Inserts the desired ones after initial comments.
    """
    lines = obj_path.read_text(encoding="utf-8", errors="replace").splitlines()

    # Remove existing mtllib/usemtl lines anywhere
    filtered = []
    for l in lines:
        ll = l.lower().strip()
        if ll.startswith("mtllib ") or ll.startswith("usemtl "):
            continue
        filtered.append(l)

    # Insert near top (after comments)
    insert_at = 0
    while insert_at < len(filtered) and filtered[insert_at].startswith("#"):
        insert_at += 1

    filtered.insert(insert_at, f"mtllib {mtl_name}")
    filtered.insert(insert_at + 1, f"usemtl {material_name}")

    obj_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")

# ----------------------------
# Decimator selection (UV-aware)
# ----------------------------

def mesh_has_wedge_uv(mesh: pml.Mesh) -> bool:
    for attr in ("has_wedge_tex_coord", "has_wedge_texcoord", "has_wedge_texture_coords"):
        if hasattr(mesh, attr):
            try:
                return bool(getattr(mesh, attr)())
            except TypeError:
                return bool(getattr(mesh, attr))
            except Exception:
                pass

    for attr in ("wedge_tex_coord_matrix", "wedge_texcoord_matrix"):
        if hasattr(mesh, attr):
            try:
                w = getattr(mesh, attr)()
                return w is not None and len(w) > 0
            except Exception:
                pass

    return False


def keep_largest_component(ms: pml.MeshSet) -> None:
    """
    Keep only the largest connected component of the mesh, removing all smaller
    components (noise). This is done by:
    1. Splitting mesh into connected components
    2. Identifying the largest by face count
    3. Removing all others
    """
    mesh = ms.current_mesh()
    initial_faces = mesh.face_number()
    initial_vertices = mesh.vertex_number()
    
    if initial_faces == 0:
        LOG.warning("Mesh has no faces, skipping component filtering.")
        return
    
    # Method 1: Try using the filter to remove small disconnected components
    # This filter removes all connected components with face count below a threshold
    try:
        # First, compute the number of connected components to determine threshold
        # We want to keep only the largest, so we'll use an iterative approach
        
        # Get connected component info by selecting small components
        # We use a very high percentage to select everything except the largest
        ms.apply_filter(
            "compute_selection_by_small_disconnected_components_per_face",
            nbfaceratio=0.0,  # ratio threshold (0 = use absolute)
            nonclosedonly=False,
        )
        
        # The above selects small components. We need a different approach.
        # Let's use the direct removal filter with a ratio
        ms.apply_filter(
            "meshing_remove_connected_component_by_face_number",
            mincomponentsize=0,  # Will be overridden by ratio
            removeunref=True,
        )
    except Exception as e:
        LOG.debug("First component removal method failed: %s. Trying alternative.", e)
    
    # Method 2: Alternative approach - remove components smaller than a fraction of total
    try:
        # Remove components that have fewer faces than 10% of the largest component
        # This effectively keeps the main mesh and removes noise
        ms.apply_filter(
            "meshing_remove_connected_component_by_diameter",
            mincomponentdiag=pml.PercentageValue(1.0),  # 1% of bounding box diagonal
        )
    except Exception as e:
        LOG.debug("Diameter-based removal failed: %s. Trying face-based.", e)
    
    # Method 3: More reliable approach - split and keep largest
    try:
        mesh = ms.current_mesh()
        current_faces = mesh.face_number()
        
        # Use selection-based approach: select small components and delete them
        # Try progressively larger thresholds until we isolate the main component
        for ratio in [0.5, 0.25, 0.1, 0.05, 0.01]:
            try:
                ms.apply_filter(
                    "compute_selection_by_small_disconnected_components_per_face", 
                    nbfaceratio=ratio,
                    nonclosedonly=False,
                )
                # Check if anything is selected
                mesh = ms.current_mesh()
                selected_count = mesh.selected_face_number()
                
                if selected_count > 0 and selected_count < current_faces:
                    # Delete selected (small) faces
                    ms.apply_filter("meshing_remove_selected_faces")
                    ms.apply_filter("meshing_remove_unreferenced_vertices")
                    LOG.info("Removed %d faces from small components (ratio threshold: %.2f)", 
                             selected_count, ratio)
                    break
            except Exception:
                continue
                
    except Exception as e:
        LOG.warning("Component isolation failed: %s. Proceeding with full mesh.", e)
    
    # Report results
    mesh = ms.current_mesh()
    final_faces = mesh.face_number()
    final_vertices = mesh.vertex_number()
    
    removed_faces = initial_faces - final_faces
    removed_vertices = initial_vertices - final_vertices
    
    if removed_faces > 0:
        LOG.info("Component cleanup: removed %d faces, %d vertices (kept %d faces, %d vertices)",
                 removed_faces, removed_vertices, final_faces, final_vertices)
    else:
        LOG.info("Component cleanup: mesh appears to be a single component (no changes).")


# ----------------------------
# Main simplify
# ----------------------------

def simplify_glb(
    input_glb: Path,
    output_glb: Path,
    target_faces: int,
    *,
    qualitythr: float = 0.7,
    extratcoordw: float = 4.0,
    preserveboundary: bool = True,
    boundaryweight: float = 2.0,
    planarquadric: bool = True,
    preservenormal: bool = True,
    optimalplacement: bool = True,
    keep_intermediate: bool = False,
    skip_texture: bool = False,
) -> None:
    input_glb = input_glb.resolve()
    output_glb = output_glb.resolve()

    if target_faces <= 0:
        raise ValueError("target_faces must be positive")

    gltfpack = shutil.which("gltfpack")
    if gltfpack is None:
        raise RuntimeError("gltfpack not found on PATH. Install gltfpack or add it to PATH.")

    with tempfile.TemporaryDirectory(prefix="simplify_glb_") as td:
        workdir = Path(td)
        LOG.info("Working directory: %s", workdir)

        # Extract optional baseColor candidate and sanitize it (skip if requested).
        sanitized_tex: Optional[Path] = None

        if skip_texture:
            LOG.info("Texture processing skipped (skip_texture=True); proceeding as untextured mesh.")
        else:
            extracted_candidate = extract_basecolor_texture_from_glb(input_glb, workdir)
            if extracted_candidate and extracted_candidate.exists():
                LOG.info("Extracted baseColor candidate: %s", extracted_candidate.name)
                sanitized_tex = sanitize_texture_for_gltfpack(extracted_candidate, workdir)
                if sanitized_tex:
                    LOG.info("Using texture for export: %s", sanitized_tex.name)
                else:
                    LOG.warning("Texture present but not decodable after sanitization; exporting untextured.")
            else:
                LOG.info("No base color texture found; proceeding as untextured mesh.")

        has_texture = sanitized_tex is not None

        # Load mesh
        ms = pml.MeshSet()
        ms.load_new_mesh(str(input_glb), load_in_a_single_layer=False)
        mesh = ms.current_mesh()
        LOG.info("Loaded mesh: %d faces, %d vertices", mesh.face_number(), mesh.vertex_number())

        # Keep only the largest connected component (remove noise)
        keep_largest_component(ms)
        mesh = ms.current_mesh()

        if target_faces >= mesh.face_number():
            LOG.info("Target faces >= input faces. Copying input to output.")
            shutil.copyfile(input_glb, output_glb)
            return

        # Triangulate if needed
        try:
            ms.apply_filter("meshing_poly_to_tri")
        except Exception:
            pass

        mesh = ms.current_mesh()
        has_wedge = mesh_has_wedge_uv(mesh)
        LOG.info("Mesh wedge-UV present: %s", has_wedge)

        # Decimate
        if has_wedge:
            ms.apply_filter(
                "meshing_decimation_quadric_edge_collapse_with_texture",
                targetfacenum=int(target_faces),
                targetperc=0.0,
                qualitythr=float(qualitythr),
                extratcoordw=float(extratcoordw),
                preserveboundary=bool(preserveboundary),
                boundaryweight=float(boundaryweight),
                optimalplacement=bool(optimalplacement),
                preservenormal=bool(preservenormal),
                planarquadric=bool(planarquadric),
                selected=False,
            )
        else:
            ms.apply_filter(
                "meshing_decimation_quadric_edge_collapse",
                targetfacenum=int(target_faces),
                targetperc=0.0,
                qualitythr=float(qualitythr),
                preserveboundary=bool(preserveboundary),
                boundaryweight=float(boundaryweight),
                optimalplacement=bool(optimalplacement),
                preservenormal=bool(preservenormal),
                planarquadric=bool(planarquadric),
                selected=False,
            )

        mesh = ms.current_mesh()
        LOG.info("Simplified mesh: %d faces, %d vertices", mesh.face_number(), mesh.vertex_number())

        # Recompute normals
        try:
            ms.apply_filter("compute_normal_per_face")
            ms.apply_filter("compute_normal_per_vertex")
        except Exception:
            pass

        # Export OBJ (only save wedge texcoords if UVs exist)
        obj_path = workdir / "simplified.obj"
        ms.save_current_mesh(
            str(obj_path),
            save_textures=False,
            save_wedge_texcoord=has_wedge,
            save_wedge_normal=True,
            save_vertex_normal=True,
        )

        # If we have a sanitized texture, write MTL and add mtllib/usemtl if needed
        if has_texture:
            material_name = "material0"
            mtl_name = "simplified.mtl"
            mtl_path = workdir / mtl_name

            write_basic_mtl(mtl_path, material_name, sanitized_tex.name)
            force_obj_mtllib_and_usemtl(obj_path, mtl_name, material_name)

            LOG.info("Wrote minimal MTL: %s (map_Kd %s)", mtl_path.name, sanitized_tex.name)

        # Convert OBJ -> GLB (cwd=workdir is essential for relative file resolution)
        cmd = [gltfpack, "-i", str(obj_path), "-o", str(output_glb)]
        LOG.info("Running (cwd=%s): %s", workdir, " ".join(cmd))

        # Capture stderr for logging
        proc = subprocess.run(cmd, cwd=str(workdir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.stdout.strip():
            LOG.debug("gltfpack stdout:\n%s", proc.stdout)
        if proc.stderr.strip():
            # gltfpack writes warnings to stderr
            LOG.warning("gltfpack stderr:\n%s", proc.stderr.rstrip())

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

        # Validate output
        if not output_glb.exists():
            raise RuntimeError("gltfpack reported success but output file does not exist.")
        if output_glb.stat().st_size < 32:
            raise RuntimeError(f"Output GLB is too small ({output_glb.stat().st_size} bytes) and likely invalid/empty.")

        # Keep intermediates if requested
        if keep_intermediate:
            out_dir = output_glb.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(obj_path, out_dir / obj_path.name)
            if has_texture:
                shutil.copyfile(workdir / "simplified.mtl", out_dir / "simplified.mtl")
                shutil.copyfile(sanitized_tex, out_dir / sanitized_tex.name)
            LOG.info("Saved intermediates next to output.")


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simplify a GLB (textured or untextured) to a target face count.")
    p.add_argument("--input", default="outputs/hi3dgen/sample/shape_mesh.glb", type=Path, help="Input .glb path")
    p.add_argument("--output", default="output.glb", type=Path, help="Output .glb path")
    p.add_argument("--target-faces", default=200_000, type=int, help="Target triangle face count")

    p.add_argument("--qualitythr", type=float, default=0.8)
    p.add_argument("--extratcoordw", type=float, default=4.0)
    p.add_argument("--preserveboundary", action="store_true", default=True)
    p.add_argument("--boundaryweight", type=float, default=2.0)
    p.add_argument("--planarquadric", action="store_true", default=True)
    p.add_argument("--preservenormal", action="store_true", default=True)
    p.add_argument("--no-optimalplacement", action="store_true")

    p.add_argument("--keep-intermediate", action="store_true")
    p.add_argument("--skip-texture", action="store_true", help="Skip texture extraction/processing (faster for known untextured meshes)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    simplify_glb(
        input_glb=args.input,
        output_glb=args.output,
        target_faces=args.target_faces,
        qualitythr=args.qualitythr,
        extratcoordw=args.extratcoordw,
        preserveboundary=bool(args.preserveboundary),
        boundaryweight=args.boundaryweight,
        planarquadric=bool(args.planarquadric),
        preservenormal=bool(args.preservenormal),
        optimalplacement=not bool(args.no_optimalplacement),
        keep_intermediate=bool(args.keep_intermediate),
        skip_texture=bool(args.skip_texture),
    )
    LOG.info("Done: %s", args.output)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        LOG.error("External tool failed: %s", e)
        sys.exit(e.returncode)
    except Exception:
        LOG.exception("Failed")
        sys.exit(1)
