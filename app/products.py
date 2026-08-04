"""Product catalog sourced from Shopify's public products.json endpoint.

Polo's Pet Closet does not expose Admin API credentials, so instead of a
maintained JSON export we pull the storefront's public (unauthenticated)
products.json feed directly: https://{shop}/products.json
This is paginated at up to 250 products per page and refreshed on a TTL
so we don't hammer the store on every chat turn.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import requests

from .config import settings

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass
class Product:
    id: int
    title: str
    handle: str
    url: str
    description: str
    vendor: str
    product_type: str
    tags: list[str]
    price_min: float | None
    price_max: float | None
    available: bool
    image: str | None

    def as_card(self) -> dict:
        """Compact representation returned to the frontend / LLM tool output."""
        price = (
            f"${self.price_min:.2f}"
            if self.price_min == self.price_max
            else f"${self.price_min:.2f} - ${self.price_max:.2f}"
        )
        return {
            "id": self.id,
            "title": self.title,
            "price": price,
            "url": self.url,
            "image": self.image,
            "available": self.available,
            "product_type": self.product_type,
        }


def _clean_html(raw_html: str | None) -> str:
    text = _TAG_RE.sub(" ", raw_html or "")
    return _WS_RE.sub(" ", text).strip()


def _normalize(raw: dict) -> Product:
    variants = raw.get("variants", [])
    prices = [float(v["price"]) for v in variants if v.get("price") is not None]
    images = raw.get("images") or []
    return Product(
        id=raw["id"],
        title=raw.get("title", ""),
        handle=raw.get("handle", ""),
        url=f"https://{settings.SHOP_DOMAIN}/products/{raw.get('handle', '')}",
        description=_clean_html(raw.get("body_html")),
        vendor=raw.get("vendor", ""),
        product_type=raw.get("product_type", ""),
        tags=raw.get("tags") or [],
        price_min=min(prices) if prices else None,
        price_max=max(prices) if prices else None,
        available=any(v.get("available") for v in variants),
        image=images[0]["src"] if images else None,
    )


def _fetch_all_raw() -> list[dict]:
    """Paginate through /products.json until an empty page is returned."""
    products: list[dict] = []
    page = 1
    while True:
        resp = requests.get(
            settings.PRODUCTS_JSON_URL,
            params={"limit": 250, "page": page},
            timeout=10,
        )
        resp.raise_for_status()
        batch = resp.json().get("products", [])
        if not batch:
            break
        products.extend(batch)
        page += 1
        if page > 40:  # hard safety cap (10k products)
            break
    return products


class ProductCatalog:
    """Thread-safe, TTL-cached view of the Shopify catalog."""

    def __init__(self, ttl_minutes: int = 30):
        self._ttl = timedelta(minutes=ttl_minutes)
        self._lock = threading.Lock()
        self._last_fetched: datetime | None = None
        self._products: list[Product] = []

    def _refresh_if_stale(self) -> None:
        stale = (
            self._last_fetched is None
            or datetime.now() - self._last_fetched > self._ttl
        )
        if not stale:
            return
        with self._lock:
            # re-check inside the lock in case another thread just refreshed
            if (
                self._last_fetched is not None
                and datetime.now() - self._last_fetched <= self._ttl
            ):
                return
            raw_products = _fetch_all_raw()
            self._products = [_normalize(p) for p in raw_products]
            self._last_fetched = datetime.now()

    def all(self) -> list[Product]:
        self._refresh_if_stale()
        return self._products

    def force_refresh(self) -> None:
        with self._lock:
            self._last_fetched = None
        self._refresh_if_stale()

    def search(
        self,
        query_terms: list[str],
        product_types: list[str] | None = None,
        only_available: bool = True,
        limit: int = 5,
    ) -> list[Product]:
        """Simple weighted keyword match across title / description / type.

        The live feed has no tags on most items, so we can't rely on tag
        filtering alone — match against title, description and product_type.
        Swap this for an embeddings-based search later if matching quality
        needs to improve (see README).
        """
        terms = [t.lower() for t in query_terms if t.strip()]
        if not terms:
            return []

        scored: list[tuple[float, Product]] = []
        for p in self.all():
            if only_available and not p.available:
                continue
            if product_types and p.product_type not in product_types:
                continue

            title_l = p.title.lower()
            desc_l = p.description.lower()
            type_l = p.product_type.lower()
            tags_l = " ".join(p.tags).lower()

            score = 0.0
            for term in terms:
                score += title_l.count(term) * 3.0
                score += type_l.count(term) * 2.0
                score += tags_l.count(term) * 2.0
                score += desc_l.count(term) * 1.0

            if score > 0:
                scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:limit]]


catalog = ProductCatalog(ttl_minutes=settings.PRODUCT_CACHE_TTL_MINUTES)
