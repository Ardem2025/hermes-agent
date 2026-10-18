"""OmniRoute image generation and image-to-image editing backend.

Routes image generation and image-to-image editing through local OmniRoute gateway to
Antigravity's gemini-3.1-flash-image model and Codex gpt-5.6-sol model.
"""

from __future__ import annotations

import concurrent.futures
import logging
import mimetypes
import os
import requests
from typing import Any, Dict, List, Optional

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "codex/gpt-5.6-sol"
DEFAULT_ANTIGRAVITY_MODEL = "antigravity/gemini-3.1-flash-image"
DEFAULT_CODEX_MODEL = "codex/gpt-5.6-sol"
DEFAULT_BASE_URL = "http://127.0.0.1:20128/v1"

_SIZES = {
    "landscape": "1024x1024",
    "square": "1024x1024",
    "portrait": "1024x1024",
}


def _load_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_api_key(cfg_key: Optional[str]) -> str:
    if cfg_key and not cfg_key.startswith("${"):
        return cfg_key
    
    if cfg_key and cfg_key.startswith("${") and cfg_key.endswith("}"):
        env_var = cfg_key[2:-1]
        val = os.environ.get(env_var)
        if val:
            return val

    for env_name in ("OMNIROUTE_LOCAL_API_KEY", "OMNIROUTE_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(env_name)
        if val:
            return val

    env_file = "/home/srvadm/.openclaw/.env"
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("OMNIROUTE_LOCAL_API_KEY=") or line.startswith("OMNIROUTE_API_KEY="):
                        return line.strip().split("=", 1)[1]
        except Exception as e:
            logger.debug("Error reading .openclaw/.env: %s", e)

    return "omniroute-local"


class OmniRouteImageGenProvider(ImageGenProvider):
    """OmniRoute local gateway image generation and edit backend."""

    @property
    def name(self) -> str:
        return "omniroute"

    @property
    def display_name(self) -> str:
        return "OmniRoute (Antigravity + Codex)"

    def is_available(self) -> bool:
        # Always available — local gateway manages credentials
        return True

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "max_reference_images": 1,
        }

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": DEFAULT_MODEL,
                "display": "GPT 5.6 Sol (Codex Image)",
                "speed": "~15-30s",
                "strengths": "High prompt adherence and detail inpainting via Codex",
                "price": "free (Codex OAuth)",
                "modalities": ["text", "image"],
                "max_reference_images": 1,
            },
            {
                "id": DEFAULT_ANTIGRAVITY_MODEL,
                "display": "Gemini 3.1 Flash Image",
                "speed": "~5-8s",
                "strengths": "Fast image generation and image-to-image editing via Antigravity",
                "price": "free (Antigravity OAuth)",
                "modalities": ["text", "image"],
                "max_reference_images": 1,
            },
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OmniRoute",
            "badge": "free",
            "tag": "Local OmniRoute gateway → Dual Antigravity/Codex Image Generation & Editing",
            "env_vars": [],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        image_input = (
            kwargs.get("image_path")
            or kwargs.get("image")
            or kwargs.get("input_image")
            or kwargs.get("image_url")
        )

        if not prompt and not image_input:
            return error_response(
                error="Prompt or input image is required",
                error_type="invalid_argument",
                provider="omniroute",
                aspect_ratio=aspect,
            )

        cfg = _load_config()
        primary_model = cfg.get("model", DEFAULT_MODEL)
        base_url = cfg.get("base_url", DEFAULT_BASE_URL)
        api_key = _resolve_api_key(cfg.get("api_key"))

        size = _SIZES.get(aspect, "1024x1024")

        models_to_run = [
            {"name": "codex", "model": DEFAULT_MODEL},
            {"name": "antigravity", "model": DEFAULT_ANTIGRAVITY_MODEL},
        ]

        if "antigravity" in primary_model.lower() or "gemini" in primary_model.lower():
            models_to_run = [
                {"name": "antigravity", "model": DEFAULT_ANTIGRAVITY_MODEL},
                {"name": "codex", "model": DEFAULT_MODEL},
            ]

        def call_omniroute_image(target_provider: str, target_model: str) -> Dict[str, Any]:
            try:
                headers = {}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                if image_input:
                    endpoint = f"{base_url.rstrip('/')}/images/edits"
                    if isinstance(image_input, str) and os.path.isfile(image_input):
                        mime_type, _ = mimetypes.guess_type(image_input)
                        if not mime_type or not mime_type.startswith("image/"):
                            ext = os.path.splitext(image_input)[1].lower()
                            if ext in (".jpg", ".jpeg"):
                                mime_type = "image/jpeg"
                            elif ext == ".webp":
                                mime_type = "image/webp"
                            else:
                                mime_type = "image/png"
                        with open(image_input, "rb") as fh:
                            files = {"image": (os.path.basename(image_input), fh.read(), mime_type)}
                            data = {
                                "model": target_model,
                                "prompt": prompt,
                                "size": size,
                            }
                            resp = requests.post(endpoint, headers=headers, data=data, files=files, timeout=180)
                    else:
                        payload = {
                            "model": target_model,
                            "prompt": prompt,
                            "size": size,
                            "image": image_input,
                        }
                        resp = requests.post(endpoint, headers=headers, json=payload, timeout=180)

                    if resp.status_code != 200:
                        return {
                            "success": False,
                            "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
                            "model": target_model,
                            "provider": target_provider,
                        }

                    res_json = resp.json()
                    data_list = res_json.get("data") or []
                    if not data_list:
                        return {
                            "success": False,
                            "error": "No image data returned from edit endpoint",
                            "model": target_model,
                            "provider": target_provider,
                        }

                    first = data_list[0]
                    b64 = first.get("b64_json")
                    url = first.get("url")

                    if b64:
                        saved_path = save_b64_image(b64, prefix=f"omniroute_{target_provider}")
                        image_ref = str(saved_path)
                    elif url:
                        if isinstance(url, str) and url.startswith("data:image/") and ";base64," in url:
                            try:
                                b64_part = url.split(";base64,")[1]
                                saved_path = save_b64_image(b64_part, prefix=f"omniroute_{target_provider}")
                                image_ref = str(saved_path)
                            except Exception as e:
                                logger.error("Failed to parse base64 from data URL: %s", e)
                                image_ref = url
                        else:
                            image_ref = url
                    else:
                        return {
                            "success": False,
                            "error": "Response returned neither base64 nor URL",
                            "model": target_model,
                            "provider": target_provider,
                        }

                    return {
                        "success": True,
                        "image": image_ref,
                        "model": target_model,
                        "provider": target_provider,
                    }
                else:
                    endpoint = f"{base_url.rstrip('/')}/images/generations"
                    payload = {
                        "model": target_model,
                        "prompt": prompt,
                        "size": size,
                        "n": 1,
                    }
                    resp = requests.post(endpoint, headers=headers, json=payload, timeout=180)

                    if resp.status_code != 200:
                        return {
                            "success": False,
                            "error": f"HTTP {resp.status_code}: {resp.text[:300]}",
                            "model": target_model,
                            "provider": target_provider,
                        }

                    res_json = resp.json()
                    data_list = res_json.get("data") or []
                    if not data_list:
                        return {
                            "success": False,
                            "error": "No image data returned from generation endpoint",
                            "model": target_model,
                            "provider": target_provider,
                        }

                    first = data_list[0]
                    b64 = first.get("b64_json")
                    url = first.get("url")

                    if b64:
                        saved_path = save_b64_image(b64, prefix=f"omniroute_{target_provider}")
                        image_ref = str(saved_path)
                    elif url:
                        if isinstance(url, str) and url.startswith("data:image/") and ";base64," in url:
                            try:
                                b64_part = url.split(";base64,")[1]
                                saved_path = save_b64_image(b64_part, prefix=f"omniroute_{target_provider}")
                                image_ref = str(saved_path)
                            except Exception as e:
                                logger.error("Failed to parse base64 from data URL: %s", e)
                                image_ref = url
                        else:
                            image_ref = url
                    else:
                        return {
                            "success": False,
                            "error": "Response returned neither base64 nor URL",
                            "model": target_model,
                            "provider": target_provider,
                        }

                    return {
                        "success": True,
                        "image": image_ref,
                        "model": target_model,
                        "provider": target_provider,
                    }
            except Exception as exc:
                return {
                    "success": False,
                    "error": str(exc),
                    "model": target_model,
                    "provider": target_provider,
                }

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_model = {
                executor.submit(call_omniroute_image, item["name"], item["model"]): item
                for item in models_to_run
            }
            done, not_done = concurrent.futures.wait(future_to_model.keys(), timeout=180.0)

            for future in future_to_model:
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    item = future_to_model[future]
                    results.append({
                        "success": False,
                        "error": f"Task crashed: {e}",
                        "model": item["model"],
                        "provider": item["name"],
                    })

        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        if not successful:
            combined_err = " | ".join([f"{r['provider']}: {r['error']}" for r in failed])
            return error_response(
                error=f"OmniRoute dual image generation failed: {combined_err}",
                error_type="api_error",
                provider="omniroute",
                model=primary_model,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        primary_res = successful[0]

        all_images = [r["image"] for r in successful]
        additional_images = [r["image"] for r in successful[1:]]

        extra_data: Dict[str, Any] = {
            "images": results,
            "dual_mode": True,
            "mode": "edit" if image_input else "generation",
            "all_images": all_images,
            "additional_images": additional_images,
        }

        if len(successful) > 1:
            primary_path = all_images[0]
            secondary_path = ", ".join(additional_images)
            media_tags = " ".join([f"MEDIA:{img}" for img in all_images])
            extra_data["message"] = (
                f"Dual generation completed: {len(all_images)} images created.\n"
                f"Primary: {primary_path}\n"
                f"Secondary: {secondary_path}\n"
                f"IMPORTANT: Include MEDIA tags for ALL generated images in your response so the user receives both photos: {media_tags}"
            )

        return success_response(
            image=primary_res["image"],
            model=primary_res["model"],
            prompt=prompt,
            aspect_ratio=aspect,
            provider="omniroute",
            extra=extra_data,
        )


def register(ctx) -> None:
    """Plugin entry point."""
    ctx.register_image_gen_provider(OmniRouteImageGenProvider())
