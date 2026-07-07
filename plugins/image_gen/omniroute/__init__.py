"""OmniRoute image generation backend.

Routes image generation through local OmniRoute gateway to
Antigravity's gemini-3.1-flash-image model and Codex gpt-5.5 model.
"""

from __future__ import annotations

import logging
import os
import concurrent.futures
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

DEFAULT_MODEL = "antigravity/gemini-3.1-flash-image"
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


class OmniRouteImageGenProvider(ImageGenProvider):
    """OmniRoute local gateway image generation backend."""

    @property
    def name(self) -> str:
        return "omniroute"

    @property
    def display_name(self) -> str:
        return "OmniRoute (Antigravity + Codex)"

    def is_available(self) -> bool:
        # Always available — local gateway doesn't need API key
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": DEFAULT_MODEL,
                "display": "Gemini 3.1 Flash Image",
                "speed": "~10s",
                "strengths": "Fast image generation via Antigravity",
                "price": "free (Antigravity OAuth)",
            },
            {
                "id": "codex/gpt-5.5",
                "display": "GPT 5.5 (Codex Image)",
                "speed": "~15s",
                "strengths": "Incredibly high prompt adherence and detail",
                "price": "free (Codex OAuth)",
            }
        ]

    def default_model(self) -> Optional[str]:
        return DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "OmniRoute",
            "badge": "free",
            "tag": "Local OmniRoute gateway → Dual Antigravity/Codex Image Generation",
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

        if not prompt:
            return error_response(
                error="Prompt is required",
                error_type="invalid_argument",
                provider="omniroute",
                aspect_ratio=aspect,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed",
                error_type="missing_dependency",
                provider="omniroute",
                aspect_ratio=aspect,
            )

        cfg = _load_config()
        primary_model = cfg.get("model", DEFAULT_MODEL)
        base_url = cfg.get("base_url", DEFAULT_BASE_URL)
        api_key = cfg.get("api_key", os.environ.get("OPENAI_API_KEY", "omniroute-local"))
        
        # Resolve ${OPENAI_API_KEY} template
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "omniroute-local")

        size = _SIZES.get(aspect, "1024x1024")

        # Set up p_model and s_model
        if "codex" in primary_model.lower() or "gpt-5.5" in primary_model.lower():
            p_model = primary_model
            s_model = "antigravity/gemini-3.1-flash-image"
        else:
            p_model = primary_model
            s_model = "codex/gpt-5.5"

        models_to_run = [
            {"name": "antigravity", "model": "antigravity/gemini-3.1-flash-image"},
            {"name": "codex", "model": "codex/gpt-5.5"}
        ]
        # Keep configured model first in results
        if p_model == "codex/gpt-5.5":
            models_to_run = [
                {"name": "codex", "model": "codex/gpt-5.5"},
                {"name": "antigravity", "model": "antigravity/gemini-3.1-flash-image"}
            ]

        def call_openai_image(target_provider: str, target_model: str) -> Dict[str, Any]:
            try:
                client = openai.OpenAI(base_url=base_url, api_key=api_key)
                response = client.images.generate(
                    model=target_model,
                    prompt=prompt,
                    size=size,  # type: ignore
                    n=1,
                )
                data = getattr(response, "data", None) or []
                if not data:
                    return {
                        "success": False, 
                        "error": "No image data returned", 
                        "model": target_model, 
                        "provider": target_provider
                    }
                
                first = data[0]
                b64 = getattr(first, "b64_json", None)
                url = getattr(first, "url", None)
                
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
                        "provider": target_provider
                    }
                
                return {
                    "success": True, 
                    "image": image_ref, 
                    "model": target_model, 
                    "provider": target_provider
                }
            except Exception as exc:
                return {
                    "success": False, 
                    "error": str(exc), 
                    "model": target_model, 
                    "provider": target_provider
                }

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_model = {
                executor.submit(call_openai_image, item["name"], item["model"]): item 
                for item in models_to_run
            }
            # Wait for all tasks to complete, max 45s
            done, not_done = concurrent.futures.wait(future_to_model.keys(), timeout=45.0)
            
            for future in future_to_model:
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    item = future_to_model[future]
                    results.append({
                        "success": False,
                        "error": f"Process crashed: {e}",
                        "model": item["model"],
                        "provider": item["name"]
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

        # Primary is the first successful image
        primary_res = successful[0]
        
        return success_response(
            image=primary_res["image"],
            model=primary_res["model"],
            prompt=prompt,
            aspect_ratio=aspect,
            provider="omniroute",
            extra={
                "images": results,
                "dual_mode": True
            }
        )


def register(ctx) -> None:
    """Plugin entry point."""
    ctx.register_image_gen_provider(OmniRouteImageGenProvider())
