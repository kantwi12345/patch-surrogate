"""
Diagnostic: build geometry only, then pause so we can inspect the
substrate material's actual permittivity value in the AEDT GUI before
any solve happens.
"""

from src.sim.schema import DesignParams
from src.sim.hfss_model import build_geometry

design = DesignParams(
    L=0.02000, W=0.02520, h=0.001524, eps_r=3.00, y0=0.00710,
)

hfss, fr_est = build_geometry(
    design, non_graphical=False, create_port=True, create_airbox=True, use_inset=True,
)

# Print what pyaedt itself thinks the permittivity value is
mat = hfss.materials.material_keys["patch_substrate"]
print(f"\n=== Material check ===")
print(f"Material name: patch_substrate")
print(f"Permittivity (via pyaedt): {mat.permittivity.value}")

input("\nInspect the substrate material in the AEDT GUI now, then press Enter to close...")
hfss.release_desktop(close_projects=True, close_desktop=True)