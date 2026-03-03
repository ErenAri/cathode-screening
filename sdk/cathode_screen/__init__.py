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
"""

from cathode_screen.client import CathodeClient, PredictionResult

__version__ = "0.1.0"
__all__ = ["CathodeClient", "PredictionResult", "__version__"]
