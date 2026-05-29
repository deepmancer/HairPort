"""hairport.stages — Pipeline stage wrappers with consistent interfaces.

Each sub-module exposes:
- A ``Stage`` class with ``run(**kwargs)`` for programmatic use.
- A ``main()`` function for CLI invocation.
- An ``if __name__ == '__main__'`` guard.

Stages
------
1. baldify       — Generate bald version of portrait images
2. caption       — Generate text captions / outpainting for bald images
3. shape_mesh    — Texture canonical shape meshes via MVAdapter
4. landmark_3d   — Estimate 3D facial landmarks from the supported frontal view
5. align_view    — Align target hairstyle to source view
6. render_view   — Render target-hair alignment views
7. enhance_view  — Enhance rendered views (FLUX.2 Klein + CodeFormer)
8. blend_hair    — Warp + blend hair onto bald heads (Poisson)
9. transfer_hair — Final hair transfer (FLUX.2 Klein 9B)
"""

__all__ = [
    "baldify",
    "caption",
    "shape_mesh",
    "landmark_3d",
    "align_view",
    "render_view",
    "enhance_view",
    "blend_hair",
    "transfer_hair",
]
