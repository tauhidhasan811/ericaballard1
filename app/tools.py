"""Tools the Polo agent can call: live product search + store policy FAQs."""
from __future__ import annotations

from langchain_core.tools import tool

from .config import settings
from .products import catalog


@tool
def search_products(query_terms: list[str], product_type: str | None = None, limit: int = 5) -> list[dict]:
    """Search Polo's Pet Closet's live product catalog.

    Use this whenever you have enough information about the dog (breed,
    size, age, or a specific need like 'cooling vest' or 'puppy chew toy')
    to recommend real products. Always base recommendations on the returned
    results — never invent products or prices.

    Args:
        query_terms: Keywords to match, e.g. ["small breed", "sweater", "chihuahua"].
        product_type: Optional exact product_type filter if known.
        limit: Max number of products to return (default 5).
    """
    results = catalog.search(
        query_terms=query_terms,
        product_types=[product_type] if product_type else None,
        only_available=True,
        limit=limit,
    )
    if not results:
        # retry once without the type filter, in case it was too narrow
        results = catalog.search(query_terms=query_terms, only_available=True, limit=limit)
    return [p.as_card() for p in results]


@tool
def get_shipping_policy() -> str:
    """Return Polo's Pet Closet's shipping policy. Use for shipping questions."""
    return settings.SHIPPING_POLICY


@tool
def get_return_policy() -> str:
    """Return Polo's Pet Closet's return/refund policy. Use for return questions."""
    return settings.RETURN_POLICY


ALL_TOOLS = [search_products, get_shipping_policy, get_return_policy]
