import os
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

from torchvision.transforms.functional import normalize

from config.paths import PathConfig
from modules.CodeFormer.basicsr.utils.download_util import load_file_from_url
from modules.CodeFormer.basicsr.utils import img2tensor, tensor2img, imwrite
from modules.CodeFormer.basicsr.utils.registry import ARCH_REGISTRY
from modules.CodeFormer.basicsr.archs.rrdbnet_arch import RRDBNet
from modules.CodeFormer.basicsr.archs.realesrgan import RealESRGANer
from modules.CodeFormer.facelib.utils.face_restoration_helper import FaceRestoreHelper
from modules.CodeFormer.facelib.utils.misc import is_gray


class CodeFormerEnhancer:
    _PRETRAIN_URLS = {
        "codeformer":  "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "detection":   "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "parsing":     "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
        "realesrgan":  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "ultrasharp":  "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth",
    }

    def __init__(
        self,
        device: str = "cuda",
        ultrasharp: bool = True,
        bg_tile: int = 100,
        bg_tile_pad: int = 10,
        face_size: int = 512,
    ):
        # pick available device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.ultrasharp = ultrasharp

        self.CODEFORMER_ROOT = Path("modules/CodeFormer")
        # ensure CodeFormer weights are downloaded under modules/CodeFormer/weights
        self._ensure_weights()

        # init background upscaler
        self._init_realesrgan(bg_tile, bg_tile_pad)

        # init CodeFormer network
        self._init_codeformer(face_size)

    def _ensure_weights(self):
        mapping = {
            "codeformer":   "weights/CodeFormer",
            "detection":    "weights/facelib",
            "parsing":      "weights/facelib",
            "realesrgan":   "weights/realesrgan",
            "ultrasharp":   "weights/ultrasharp_4x",
        }
        for name, url in self._PRETRAIN_URLS.items():
            subdir = self.CODEFORMER_ROOT / mapping[name]
            subdir.mkdir(parents=True, exist_ok=True)
            fname = url.split("/")[-1]
            dest = subdir / fname
            if not dest.exists():
                load_file_from_url(url, model_dir=str(subdir), file_name=fname)

    def _init_realesrgan(self, tile: int, tile_pad: int):
        """Sets up RealESRGANer for background (and optionally face) upscaling."""
        rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        if self.ultrasharp:
            file_name = "4x-UltraSharp.pth"
            model_dir = str(self.CODEFORMER_ROOT / "weights" / "ultrasharp_4x" / file_name)
            
        else:
            file_name = "RealESRGAN_x4plus.pth"
            model_dir = str(self.CODEFORMER_ROOT / "weights" / "realesrgan" / file_name)
            
        self.bg_upsampler = RealESRGANer(
            scale=4,
            model_path=model_dir,
            model=rrdb,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=0,
            half=True,
            device=self.device,
        )

    def _init_codeformer(self, face_size: int):
        """Loads the CodeFormer network for face restoration."""
        ckpt = torch.load(
            str(self.CODEFORMER_ROOT / "weights" / "CodeFormer" / "codeformer.pth"),
            map_location="cpu"
        )["params_ema"]
        self.codeformer = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)
        self.codeformer.load_state_dict(ckpt)
        self.codeformer.eval()
        self.face_size = face_size

    def _normalize_input_image(self, pil_img: Image.Image) -> tuple[np.ndarray, bool, bool]:
        if isinstance(pil_img, np.ndarray):
            if pil_img.dtype == np.float32 or pil_img.dtype == np.float64:
                pil_img = (pil_img * 255).astype(np.uint8)
            pil_img = Image.fromarray(pil_img)

        is_grayscale = False
        if pil_img.mode == 'RGBA':
            alpha = np.array(pil_img.getchannel('A'))
            rgb = pil_img.convert('RGB')
            img_array = np.array(rgb)
            has_alpha = True
        elif pil_img.mode == 'RGB':
            img_array = np.array(pil_img)
            alpha = None
            has_alpha = False
        elif pil_img.mode == 'L':
            img_array = np.array(pil_img)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            alpha = None
            has_alpha = False
            is_grayscale = True
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR), alpha, has_alpha, is_grayscale
        elif pil_img.mode in ('LA', 'PA'):
            alpha = np.array(pil_img.getchannel('A')) if 'A' in pil_img.getbands() else None
            rgb = pil_img.convert('RGB')
            img_array = np.array(rgb)
            has_alpha = alpha is not None
        else:
            img_array = np.array(pil_img.convert('RGB'))
            alpha = None
            has_alpha = False
        
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        is_grayscale = False
        
        return img_bgr, alpha, has_alpha, is_grayscale

    def enhance(
        self,
        pil_img: Image.Image,
        face_align: bool = True,
        background_enhance: bool = True,
        face_upsample: bool = True,
        upscale: int = 2,
        codeformer_fidelity: float = 0.5,
        only_largest_face: bool = True,
    ) -> Image.Image:
        img_bgr, alpha_channel, has_alpha, is_grayscale = self._normalize_input_image(pil_img)
        h, w = img_bgr.shape[:2]

        helper = FaceRestoreHelper(
            upscale,
            face_size=self.face_size,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            use_parse=True,
            device=self.device,
        )

        has_aligned = not face_align
        if face_align:
            helper.read_image(img_bgr)
            helper.get_face_landmarks_5(only_center_face=False)
            
            # Filter to keep only the largest face if requested
            if only_largest_face and len(helper.all_landmarks_5) > 1:
                # Compute face area from landmarks for each detected face
                largest_idx = 0
                largest_area = 0
                for idx, landmarks in enumerate(helper.all_landmarks_5):
                    # landmarks is a 5x2 array of facial keypoints
                    # Compute bounding box area from the landmarks
                    x_coords = landmarks[:, 0]
                    y_coords = landmarks[:, 1]
                    width = x_coords.max() - x_coords.min()
                    height = y_coords.max() - y_coords.min()
                    area = width * height
                    if area > largest_area:
                        largest_area = area
                        largest_idx = idx
                
                # Keep only the largest face
                helper.all_landmarks_5 = [helper.all_landmarks_5[largest_idx]]
                if hasattr(helper, 'det_faces') and helper.det_faces is not None and len(helper.det_faces) > largest_idx:
                    helper.det_faces = helper.det_faces[largest_idx:largest_idx+1]
            
            helper.align_warp_face()
        else:
            resized = cv2.resize(img_bgr, (self.face_size, self.face_size), interpolation=cv2.INTER_LINEAR)
            helper.is_gray = is_gray(resized, threshold=10) or is_grayscale
            helper.cropped_faces = [resized]

        for cropped in helper.cropped_faces:
            tensor = img2tensor(cropped / 255.0, bgr2rgb=True, float32=True)
            normalize(tensor, (0.5,)*3, (0.5,)*3, inplace=True)
            tensor = tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.codeformer(tensor, w=codeformer_fidelity, adain=True)[0]
            restored = tensor2img(output, rgb2bgr=True, min_max=(-1,1)).astype("uint8")
            helper.add_restored_face(restored, cropped)

        if background_enhance and upscale > 1:
            bg_up = self.bg_upsampler.enhance(img_bgr, outscale=upscale)[0]
        else:
            bg_up = img_bgr

        if not has_aligned:
            helper.get_inverse_affine(None)
            if face_upsample and self.bg_upsampler is not None:
                result_bgr = helper.paste_faces_to_input_image(
                    upsample_img=bg_up,
                    draw_box=False,
                    face_upsampler=self.bg_upsampler,
                )
            else:
                result_bgr = helper.paste_faces_to_input_image(
                    upsample_img=bg_up,
                    draw_box=False,
                )
        else:
            result_bgr = restored

        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        
        if has_alpha and alpha_channel is not None:
            if alpha_channel.shape != result_rgb.shape[:2]:
                alpha_resized = cv2.resize(alpha_channel, (result_rgb.shape[1], result_rgb.shape[0]), 
                                          interpolation=cv2.INTER_LINEAR)
            else:
                alpha_resized = alpha_channel
            result_rgba = np.dstack([result_rgb, alpha_resized])
            return Image.fromarray(result_rgba, mode='RGBA')
        
        return Image.fromarray(result_rgb)
