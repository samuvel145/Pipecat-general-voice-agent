"""
pipeline/jll_client.py
Async HTTP client for the JLL integration proxy.

Routes (from integration (1).js):
  GET  {JLL_PROXY_URL}/proxy/search             -> search_properties
  GET  {JLL_PROXY_URL}/proxy/property-details   -> get_property_details
  GET  {JLL_PROXY_URL}/proxy/areas-by-budget    -> areas_by_budget
  POST {JLL_PROXY_URL}/proxy/callback           -> submit_callback
  POST {JLL_PROXY_URL}/proxy/site-visit         -> schedule_site_visit
"""

import json
import logging
from typing import Any

import httpx

from config import settings

log = logging.getLogger("JLL-CLIENT")

_BASE = settings.JLL_PROXY_URL.rstrip("/")
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=10.0)
    return _client


async def search_properties(
    city: str,
    property_type: str = "",
    location: str = "",
    min_price: int | None = None,
    max_price: int | None = None,
    bedrooms: str = "",
    page: int = 1,
    limit: int = 3,
    exclude_slugs: list[str] | None = None,
    sort_preference: str = "",
    call_id: str = "",
) -> dict:
    """GET /proxy/search"""
    params: dict[str, Any] = {"city": city, "page": str(page), "limit": str(limit)}
    if property_type:
        params["property_type"] = property_type
    if location:
        params["location"] = location
    if min_price is not None:
        params["min_price"] = str(min_price)
    if max_price is not None:
        params["max_price"] = str(max_price)
    if bedrooms:
        params["bedrooms"] = bedrooms
    if sort_preference:
        params["sort_preference"] = sort_preference
    if exclude_slugs:
        params["exclude_slugs"] = json.dumps(exclude_slugs)
    if call_id:
        params["call_id"] = call_id

    log.info(f"[search_properties] city={city} type={property_type} loc={location} budget={min_price}-{max_price} page={page}")
    try:
        r = await _get_client().get(f"{_BASE}/proxy/search", params=params)
        r.raise_for_status()
        result = r.json()
        log.info(f"[search_properties] returned {len(result.get('data') or [])} results")
        return result
    except httpx.HTTPStatusError as e:
        log.error(f"[search_properties] HTTP {e.response.status_code}: {e.response.text[:200]}")
        return {"success": False, "error": str(e), "data": []}
    except Exception as e:
        log.error(f"[search_properties] failed: {e}")
        return {"success": False, "error": str(e), "data": []}


async def get_property_details(property_id: str, city: str = "", call_id: str = "") -> dict:
    """GET /proxy/property-details"""
    params: dict[str, Any] = {"property_id": property_id}
    if city:
        params["city"] = city
    if call_id:
        params["call_id"] = call_id
    log.info(f"[get_property_details] property_id={property_id}")
    try:
        r = await _get_client().get(f"{_BASE}/proxy/property-details", params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"[get_property_details] failed: {e}")
        return {"success": False, "error": str(e)}


async def areas_by_budget(
    city: str,
    property_type: str = "",
    min_price: int | None = None,
    max_price: int | None = None,
    call_id: str = "",
) -> dict:
    """GET /proxy/areas-by-budget"""
    params: dict[str, Any] = {"city": city}
    if property_type:
        params["property_type"] = property_type
    if min_price is not None:
        params["min_price"] = str(min_price)
    if max_price is not None:
        params["max_price"] = str(max_price)
    if call_id:
        params["call_id"] = call_id
    log.info(f"[areas_by_budget] city={city} budget={min_price}-{max_price}")
    try:
        r = await _get_client().get(f"{_BASE}/proxy/areas-by-budget", params=params)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"[areas_by_budget] failed: {e}")
        return {"success": False, "error": str(e), "areas": []}


async def submit_callback(
    name: str, phone: str, city: str = "", property_id: str = "", call_id: str = ""
) -> dict:
    """POST /proxy/callback"""
    payload = {"name": name, "phone": phone, "city": city, "property_id": property_id, "call_id": call_id}
    log.info(f"[submit_callback] name={name} phone={phone}")
    try:
        r = await _get_client().post(f"{_BASE}/proxy/callback", json=payload)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"[submit_callback] failed: {e}")
        return {"success": False, "error": str(e)}


async def schedule_site_visit(
    name: str, phone: str, property_id: str, preferred_date: str = "", call_id: str = ""
) -> dict:
    """POST /proxy/site-visit"""
    payload = {"name": name, "phone": phone, "property_id": property_id, "preferred_date": preferred_date, "call_id": call_id}
    log.info(f"[schedule_site_visit] property_id={property_id} name={name}")
    try:
        r = await _get_client().post(f"{_BASE}/proxy/site-visit", json=payload)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"[schedule_site_visit] failed: {e}")
        return {"success": False, "error": str(e)}


async def close_client() -> None:
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
    _client = None
