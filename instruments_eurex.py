"""Eurex govvie instrument registry entries — registered into tlob.instruments at import.

This module owns ALL instrument economics for tlob-eurex (the tlob library
ships zero entries). Import it (for side effects) before any tlob code that
resolves fees/sessions/lots.
"""
from __future__ import annotations

from tlob.instruments import InstrumentSpec, Session, register_instrument

_SESSION_EUREX = Session(tz="Europe/Berlin", open_hour=8, close_hour=17)

# Eurex euro government-bond futures, continuous front (.v.0), Databento XEUR.EOBI.
# point_value = €1000 per index point; fee €0.40/side is the exchange+clearing
# placeholder carried over from the original tlob-a2a entries.
for _root, _tick in (
    ("FGBL", 0.01),   # Bund 10y DE
    ("FGBM", 0.01),   # Bobl 5y DE
    ("FGBS", 0.005),  # Schatz 2y DE
    ("FGBX", 0.02),   # Buxl 30y DE
    ("FBTP", 0.01),   # BTP 10y IT
    ("FOAT", 0.01),   # OAT 10y FR
):
    register_instrument(_root, InstrumentSpec(
        asset_class="eurex_rates", currency="EUR", point_value=1000.0,
        fee_per_side_native=0.40, tick_size=_tick, session=_SESSION_EUREX,
        volume_unit="contracts"))
