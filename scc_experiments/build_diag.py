"""Build the detector with Python-driven MSVC compilation."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPENCV = Path("E:/OPENCV/opencv/build")

SRC = ROOT / "src"
BUILD = ROOT / "build"
BIN = BUILD / "bin"

SOURCES = [
    "adaptApproxPolyDP.cpp",
    "Contours.cpp",
    "EllipseNonMaximumSuppression.cpp",
    "FLED.cpp",
    "FLED_Export.cpp",
    "FLED_drawAndWriteFunctions.cpp",
    "FLED_Initialization.cpp",
    "FLED_PrivateFunctions.cpp",
    "Group.cpp",
    "LinkMatrix.cpp",
    "Node_FC.cpp",
    "Segmentation.cpp",
    "Validation.cpp",
    "main.cpp",
]

VS_DEVCMD = (
    r"D:\Program Files\Microsoft Visual Studio\2022\Community"
    r"\VC\Auxiliary\Build\vcvars64.bat"
)


def get_vs_env() -> dict:
    """Capture MSVC environment by running vcvars64.bat and printing env."""
    script = f'@echo off\r\ncall "{VS_DEVCMD}" >nul 2>&1\r\nset\r\n'
    proc = subprocess.run(
        ["cmd.exe", "/c", script],
        capture_output=True,
        text=True,
        shell=False,
    )
    env = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip()
    return env


def main() -> int:
    BIN.mkdir(parents=True, exist_ok=True)

    print("Setting up MSVC environment...")
    env = get_vs_env()
    if not env.get("INCLUDE"):
        print("ERROR: Could not capture MSVC environment. Try running from VS Developer Command Prompt.")
        return 1
    print(f"  INCLUDE found, {len(env)} env vars total")

    include_dirs = [str(OPENCV / "include"), str(SRC)]
    env["INCLUDE"] = ";".join(include_dirs) + ";" + env.get("INCLUDE", "")
    env["LIB"] = str(OPENCV / "x64" / "vc16" / "lib") + ";" + env.get("LIB", "")

    project_root_macro = str(ROOT).replace("\\", "\\\\")

    objs = []
    for src_name in SOURCES:
        src_path = SRC / src_name
        obj_path = BUILD / src_name.replace(".cpp", ".obj")
        cmd = [
            "cl.exe",
            "/std:c++17", "/O2", "/EHsc", "/MD", "/nologo",
            f'/DAAMED_OPENCV_PROJECT_ROOT="{project_root_macro}"',
            "/c",
            f"/Fo{obj_path}",
            str(src_path),
        ]
        print(f"  Compiling {src_name}...")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        if proc.returncode != 0:
            print(f"  ERROR compiling {src_name}:")
            print(proc.stderr)
            return 1
        objs.append(str(obj_path))

    print("Linking aamed_demo.exe...")
    link_cmd = [
        "link.exe", "/nologo",
        f"/OUT:{BIN / 'aamed_demo.exe'}",
    ] + objs + ["opencv_world4120.lib"]

    proc = subprocess.run(link_cmd, capture_output=True, text=True, env=env, timeout=120)
    if proc.returncode != 0:
        print("ERROR linking:")
        print(proc.stderr)
        return 1

    print(f"Build SUCCESS: {BIN / 'aamed_demo.exe'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
