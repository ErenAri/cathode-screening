"""
CathodeScreen Python SDK — AI-powered battery cathode screening.

Usage:
    from cathode_screen import CathodeClient

    client = CathodeClient(api_key="your-key")
    result = client.predict("LiCoO2.cif")
    print(result.decision)      # "KEEP" / "MAYBE" / "KILL"
    print(result.ehull)         # 0.023 eV/atom
    print(result.capacity)      # 274.0 mAh/g
    print(result.voltage)       # 3.90 V

    # Async client
    async with AsyncCathodeClient(api_key="your-key") as client:
        result = await client.predict("LiCoO2.cif")

    # Multi-tenant
    client = CathodeClient(api_key="sk-...", org_id="acme-corp")
"""

from cathode_screen.client import (
    AsyncCathodeClient,
    AsyncJob,
    CathodeAPIError,
    CathodeClient,
    CathodeProperties,
    PredictionResult,
)

__version__ = "1.3.0"
__all__ = [
    "CathodeClient",
    "AsyncCathodeClient",
    "PredictionResult",
    "CathodeProperties",
    "AsyncJob",
    "CathodeAPIError",
    "__version__",
]
