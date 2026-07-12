"""ส่ง JSON output ไปยัง GAS Endpoint (TOR Phase 1)"""

import json
import time

import requests


def post_to_gas(payload: dict, url: str, timeout_s: float = 30.0,
                max_retries: int = 3) -> dict:
    """POST JSON → GAS พร้อม retry แบบ exponential backoff

    คืน {"ok": bool, "status_code": int|None, "response": str|None,
    "attempts": int}
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url, json=payload, timeout=timeout_s,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code < 400:
                return {"ok": True, "status_code": resp.status_code,
                        "response": resp.text[:500], "attempts": attempt}
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        if attempt < max_retries:
            time.sleep(2 ** (attempt - 1))
    return {"ok": False, "status_code": None,
            "response": last_error, "attempts": max_retries}


def save_json(payload: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
