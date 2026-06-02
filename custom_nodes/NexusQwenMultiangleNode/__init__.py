from __future__ import annotations

import math


def _build_camera_info(horizontal_angle: int, vertical_angle: int, zoom: float) -> dict:
    az_rad = math.radians(horizontal_angle % 360)
    el_rad = math.radians(vertical_angle)
    visual_dist = 2.6 - (zoom / 10.0) * 2.0
    center_y = 0.5
    return {
        "position": {
            "x": visual_dist * math.sin(az_rad) * math.cos(el_rad),
            "y": center_y + visual_dist * math.sin(el_rad),
            "z": visual_dist * math.cos(az_rad) * math.cos(el_rad),
        },
        "target": {"x": 0.0, "y": center_y, "z": 0.0},
        "zoom": 1,
        "cameraType": "perspective",
    }


def _prompt(horizontal_angle: int, vertical_angle: int, zoom: float) -> str:
    h_angle = horizontal_angle % 360
    if h_angle < 22.5 or h_angle >= 337.5:
        h_direction = "front view"
    elif h_angle < 67.5:
        h_direction = "front-right quarter view"
    elif h_angle < 112.5:
        h_direction = "right side view"
    elif h_angle < 157.5:
        h_direction = "back-right quarter view"
    elif h_angle < 202.5:
        h_direction = "back view"
    elif h_angle < 247.5:
        h_direction = "back-left quarter view"
    elif h_angle < 292.5:
        h_direction = "left side view"
    else:
        h_direction = "front-left quarter view"

    if vertical_angle < -15:
        v_direction = "low-angle shot"
    elif vertical_angle < 15:
        v_direction = "eye-level shot"
    elif vertical_angle < 45:
        v_direction = "elevated shot"
    else:
        v_direction = "high-angle shot"

    if zoom < 2:
        distance = "wide shot"
    elif zoom < 6:
        distance = "medium shot"
    else:
        distance = "close-up"
    return f"<sks> {h_direction} {v_direction} {distance}"


class NexusQwenMultiangleCameraNode:
    RETURN_TYPES = ("STRING", "LOAD3D_CAMERA")
    RETURN_NAMES = ("prompt", "camera_info")
    FUNCTION = "execute"
    CATEGORY = "image/multiangle"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "horizontal_angle": ("INT", {"default": 54, "min": 0, "max": 360, "step": 1}),
                "vertical_angle": ("INT", {"default": 29, "min": -30, "max": 60, "step": 1}),
                "zoom": ("FLOAT", {"default": 2.1, "min": 0.0, "max": 10.0, "step": 0.1}),
                "default_prompts": ("BOOLEAN", {"default": False}),
                "camera_view": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    def execute(self, horizontal_angle, vertical_angle, zoom, default_prompts=False, camera_view=False, image=None):
        horizontal = max(0, min(360, int(horizontal_angle)))
        vertical = max(-30, min(60, int(vertical_angle)))
        zoom_value = max(0.0, min(10.0, float(zoom)))
        return (_prompt(horizontal, vertical, zoom_value), _build_camera_info(horizontal, vertical, zoom_value))


NODE_CLASS_MAPPINGS = {
    "QwenMultiangleCameraNode": NexusQwenMultiangleCameraNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenMultiangleCameraNode": "Qwen Multiangle Camera",
}
