"""OmniRoute image generation backend.

Routes image generation through local OmniRoute gateway to
Antigravity's gemini-3.1-flash-image model and Codex gpt-5.5 model.
"""

from __future__ import annotations

import logging
import os
import concurrent.futures
import threading
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


def send_telegram_message(token: str, chat_id: str, text: str, thread_id: Optional[str] = None):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    if thread_id and str(thread_id).strip():
        try:
            payload["message_thread_id"] = int(thread_id)
        except ValueError:
            pass
    try:
        import requests
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        logger.info("Direct telegram message sent successfully")
    except Exception as e:
        logger.error("Failed to send direct telegram message: %s", e)


def send_telegram_photo(token: str, chat_id: str, photo_path: str, caption: str, thread_id: Optional[str] = None):
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    if not os.path.exists(photo_path):
        logger.error("Photo path does not exist for direct send: %s", photo_path)
        return
    payload = {
        "chat_id": chat_id,
        "caption": caption[:1024]
    }
    if thread_id and str(thread_id).strip():
        try:
            payload["message_thread_id"] = int(thread_id)
        except ValueError:
            pass
    try:
        import requests
        with open(photo_path, "rb") as f:
            files = {"photo": f}
            r = requests.post(url, data=payload, files=files, timeout=30)
            r.raise_for_status()
            logger.info("Direct telegram photo sent successfully")
    except Exception as e:
        logger.error("Failed to send direct telegram photo: %s", e)


def bg_wait_and_send(future, token, chat_id, thread_id, target_provider, target_model, prompt):
    try:
        logger.info("Background thread waiting for Codex generated image...")
        res = future.result(timeout=120.0)
        if res and res.get("success"):
            image_path = res.get("image")
            caption = f"Вариант 2 (высокодетализированный вариант Codex)"
            send_telegram_photo(token, chat_id, image_path, caption, thread_id)
        else:
            err = res.get("error", "Unknown error")
            logger.error("Background image generation failed: %s", err)
            send_telegram_message(token, chat_id, f"⚠️ Не удалось сгенерировать второй вариант (Codex): {err}", thread_id)
    except Exception as e:
        logger.error("Error in bg_wait_and_send: %s", e)
        send_telegram_message(token, chat_id, f"⚠️ Произошел технический сбой при генерации второго варианта: {e}", thread_id)


class OmniRouteImageGenProvider(ImageGenProvider):
    """OmniRoute local gateway image generation backend."""

    @property
    def name(self) -> str:
        return "omniroute"

    @property
    def display_name(self) -> str:
        return "OmniRoute (Antigravity + Codex)"

    def is_available(self) -> bool:
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
        
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "omniroute-local")

        size = _SIZES.get(aspect, "1024x1024")

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

        primary_item = models_to_run[0]
        second_item = models_to_run[1]

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        try:
            future_to_model = {
                executor.submit(call_openai_image, item["name"], item["model"]): item 
                for item in models_to_run
            }

            primary_future = None
            second_future = None
            for fut, item in future_to_model.items():
                if item == primary_item:
                    primary_future = fut
                elif item == second_item:
                    second_future = fut

            # Check if running within a Telegram Gateway context
            from gateway.session_context import get_session_env
            token = os.environ.get("TELEGRAM_BOT_TOKEN")
            chat_id = get_session_env("HERMES_SESSION_CHAT_ID")
            thread_id = get_session_env("HERMES_SESSION_THREAD_ID")
            
            is_telegram = bool(token and chat_id)

            if is_telegram:
                logger.info("Running in Telegram context. Setting up fast image generation optimization.")
                done, not_done = concurrent.futures.wait(future_to_model.keys(), timeout=10.0)
                
                if primary_future.done() and second_future.done():
                    try:
                        res1 = primary_future.result()
                    except Exception as e:
                        res1 = {"success": False, "error": str(e), "model": primary_item["model"], "provider": primary_item["name"]}
                    try:
                        res2 = second_future.result()
                    except Exception as e:
                        res2 = {"success": False, "error": str(e), "model": second_item["model"], "provider": second_item["name"]}
                    
                    if res1.get("success") and res2.get("success"):
                        logger.info("Both models completed fast. Returning dual images.")
                        return success_response(
                            image=res1["image"],
                            model=res1["model"],
                            prompt=prompt,
                            aspect_ratio=aspect,
                            provider="omniroute",
                            extra={
                                "images": [res1, res2],
                                "dual_mode": True
                            }
                        )
                
                if not primary_future.done():
                    try:
                        res1 = primary_future.result(timeout=25.0)
                    except Exception as e:
                        res1 = {
                            "success": False,
                            "error": f"Primary model crashed or timed out: {e}",
                            "model": primary_item["model"],
                            "provider": primary_item["name"]
                        }
                else:
                    try:
                        res1 = primary_future.result()
                    except Exception as e:
                        res1 = {
                            "success": False,
                            "error": str(e),
                            "model": primary_item["model"],
                            "provider": primary_item["name"]
                        }

                if not res1.get("success") and second_future.done():
                    try:
                        res2 = second_future.result()
                        if res2.get("success"):
                            logger.info("Primary model failed but secondary succeeded fast. Returning secondary.")
                            return success_response(
                                image=res2["image"],
                                model=res2["model"],
                                prompt=prompt,
                                aspect_ratio=aspect,
                                provider="omniroute",
                                extra={
                                    "images": [res2],
                                    "dual_mode": False
                                }
                            )
                    except Exception:
                        pass

                if res1.get("success"):
                    if second_future.done():
                        try:
                            res2 = second_future.result()
                        except Exception as e:
                            res2 = {"success": False, "error": str(e), "model": second_item["model"], "provider": second_item["name"]}
                        
                        if res2.get("success"):
                            logger.info("Both finished by now. Returning dual images.")
                            return success_response(
                                image=res1["image"],
                                model=res1["model"],
                                prompt=prompt,
                                aspect_ratio=aspect,
                                provider="omniroute",
                                extra={
                                    "images": [res1, res2],
                                    "dual_mode": True
                                }
                            )
                        else:
                            logger.info("Secondary model failed. Returning only primary.")
                            return success_response(
                                image=res1["image"],
                                model=res1["model"],
                                prompt=prompt,
                                aspect_ratio=aspect,
                                provider="omniroute",
                                extra={
                                    "images": [res1],
                                    "dual_mode": False
                                }
                            )
                    else:
                        logger.info("Primary image ready, secondary is still generating. Initiating background wait.")
                        send_telegram_message(
                            token, 
                            chat_id, 
                            "<i>Первый эскиз готов! Второе изображение (высокодетализированный вариант Codex) еще создаётся и будет отправлено через минуту...</i>", 
                            thread_id
                        )
                        t = threading.Thread(
                            target=bg_wait_and_send,
                            args=(second_future, token, chat_id, thread_id, second_item["name"], second_item["model"], prompt),
                            daemon=True
                        )
                        t.start()
                        
                        return success_response(
                            image=res1["image"],
                            model=res1["model"],
                            prompt=prompt,
                            aspect_ratio=aspect,
                            provider="omniroute",
                            extra={
                                "images": [res1],
                                "dual_mode": False
                            }
                        )
                
                err_msg = res1.get("error", "Unknown error")
                return error_response(
                    error=f"Primary model generation failed: {err_msg}",
                    error_type="api_error",
                    provider="omniroute",
                    model=primary_model,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )

            else:
                # Synchronous fallback for CLI / Tests
                results = []
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
        finally:
            executor.shutdown(wait=False)


def register(ctx) -> None:
    """Plugin entry point."""
    ctx.register_image_gen_provider(OmniRouteImageGenProvider())