"""Central configuration for the Polo assistant."""
from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Shopify store (public storefront JSON — no Admin API access required)
    SHOP_DOMAIN: str = os.getenv("SHOP_DOMAIN", "polospetcloset.myshopify.com")
    PRODUCTS_JSON_URL: str = f"https://{SHOP_DOMAIN}/products.json"
    PRODUCT_CACHE_TTL_MINUTES: int = int(os.getenv("PRODUCT_CACHE_TTL_MINUTES", "30"))

    # LLM
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4.1")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.6"))

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Store policy snippets used for FAQ answers (fill in real copy)
    SHIPPING_POLICY: str = os.getenv(
        "SHIPPING_POLICY",
        "Orders ship within 1-3 business days. Standard shipping takes 5-7 business days.",
    )
    RETURN_POLICY: str = os.getenv(
        "RETURN_POLICY",
        "Returns are accepted within 30 days of delivery for unused items in original packaging.",
    )


settings = Settings()
