# ============================================================
#   EARTHLENS AI – DEPENDENCY CHECKER
#   Run: python check_deps.py
# ============================================================

import importlib
import subprocess
import sys

# ── package_name : import_name ────────────────────────────────
DEPENDENCIES = {
    # UI
    "streamlit"                  : "streamlit",
    # Satellite & Geospatial
    "sentinelhub"                : "sentinelhub",
    "landsatxplore"              : "landsatxplore",
    "rasterio"                   : "rasterio",
    "pyproj"                     : "pyproj",
    "shapely"                    : "shapely",
    "geopandas"                  : "geopandas",
    # Numerical
    "numpy"                      : "numpy",
    "scipy"                      : "scipy",
    # Image Processing
    "Pillow"                     : "PIL",
    "opencv-python-headless"     : "cv2",
    "scikit-image"               : "skimage",
    # Data
    "pandas"                     : "pandas",
    "xarray"                     : "xarray",
    "netCDF4"                    : "netCDF4",
    # ML
    "scikit-learn"               : "sklearn",
    "joblib"                     : "joblib",
    # Visualization
    "folium"                     : "folium",
    "streamlit-folium"           : "streamlit_folium",
    "plotly"                     : "plotly",
    "matplotlib"                 : "matplotlib",
    "branca"                     : "branca",
    # HTTP
    "requests"                   : "requests",
    "httpx"                      : "httpx",
    "aiohttp"                    : "aiohttp",
    # Config
    "python-dotenv"              : "dotenv",
    "pydantic"                   : "pydantic",
    # File formats
    "h5py"                       : "h5py",
    "zarr"                       : "zarr",
    # Cache
    "cachetools"                 : "cachetools",
    "diskcache"                  : "diskcache",
    # Dev
    "jupyter"                    : "jupyter",
    "pytest"                     : "pytest",
    "pytest-cov"                 : "pytest_cov",
    # Logging
    "loguru"                     : "loguru",
    "tqdm"                       : "tqdm",
    "rich"                       : "rich",
}

# ── colours ──────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def check():
    installed   = []
    missing     = []

    print(f"\n{BOLD}{CYAN}{'═'*58}")
    print(f"  🛰️  EARTHLENS AI — DEPENDENCY CHECKER")
    print(f"{'═'*58}{RESET}\n")
    print(f"  {'Package':<32} {'Status':<12} {'Version'}")
    print(f"  {'-'*54}")

    for pkg_name, import_name in DEPENDENCIES.items():
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "n/a")
            print(f"  {GREEN}✅  {pkg_name:<28}{RESET}  {'installed':<12}  {CYAN}{version}{RESET}")
            installed.append(pkg_name)
        except ImportError:
            print(f"  {RED}❌  {pkg_name:<28}{RESET}  {RED}NOT FOUND{RESET}")
            missing.append(pkg_name)

    # ── Summary ───────────────────────────────────────────────
    total = len(DEPENDENCIES)
    print(f"\n{BOLD}{'═'*58}{RESET}")
    print(f"  {GREEN}✅  Installed : {len(installed)}/{total}{RESET}")
    print(f"  {RED}❌  Missing   : {len(missing)}/{total}{RESET}")
    print(f"{BOLD}{'═'*58}{RESET}\n")

    # ── Auto-install prompt ───────────────────────────────────
    if missing:
        print(f"{YELLOW}⚠️  Missing packages:{RESET}")
        for m in missing:
            print(f"    • {m}")

        print(f"\n{BOLD}Install missing packages?{RESET}")
        print(f"  {CYAN}[1]{RESET} Yes — install all missing now")
        print(f"  {CYAN}[2]{RESET} No  — just show pip command")
        print(f"  {CYAN}[3]{RESET} Skip")

        choice = input("\nYour choice (1/2/3): ").strip()

        pip_cmd = [sys.executable, "-m", "pip", "install"] + missing

        if choice == "1":
            print(f"\n{CYAN}Installing...{RESET}\n")
            subprocess.check_call(pip_cmd)
            print(f"\n{GREEN}{BOLD}✅ Done! Re-run this script to verify.{RESET}\n")

        elif choice == "2":
            print(f"\n{CYAN}Run this command:{RESET}")
            print(f"  pip install {' '.join(missing)}\n")

        else:
            print(f"\n{YELLOW}Skipped. Run manually:{RESET}")
            print(f"  pip install -r requirements.txt\n")
    else:
        print(f"{GREEN}{BOLD}🎉 All dependencies are installed! EarthLens AI is ready.{RESET}\n")


if __name__ == "__main__":
    check()
