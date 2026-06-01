"""Build the detector using Python subprocess with manual MSVC paths."""
import subprocess
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MSVC = Path(r"D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207")
SDK = Path(r"C:\Program Files (x86)\Windows Kits\10")
SDK_VER = "10.0.26100.0"
OPENCV = Path(r"E:\OPENCV\opencv\build")

SOURCES = [
    "adaptApproxPolyDP.cpp", "Contours.cpp",
    "EllipseNonMaximumSuppression.cpp", "FLED.cpp", "FLED_Export.cpp",
    "FLED_drawAndWriteFunctions.cpp", "FLED_Initialization.cpp",
    "FLED_PrivateFunctions.cpp", "Group.cpp", "LinkMatrix.cpp",
    "Node_FC.cpp", "Segmentation.cpp", "Validation.cpp", "main.cpp",
]


def main():
    build_dir = ROOT / "build"
    bin_dir = build_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    cl = MSVC / "bin" / "Hostx64" / "x64" / "cl.exe"
    link = MSVC / "bin" / "Hostx64" / "x64" / "link.exe"

    includes = [
        str(ROOT / "src"),
        str(OPENCV / "include"),
        str(MSVC / "include"),
        str(SDK / "Include" / SDK_VER / "ucrt"),
        str(SDK / "Include" / SDK_VER / "shared"),
        str(SDK / "Include" / SDK_VER / "um"),
    ]

    libs = [
        str(MSVC / "lib" / "x64"),
        str(SDK / "Lib" / SDK_VER / "ucrt" / "x64"),
        str(SDK / "Lib" / SDK_VER / "um" / "x64"),
        str(OPENCV / "x64" / "vc16" / "lib"),
    ]

    env = os.environ.copy()
    env["INCLUDE"] = ";".join(includes)
    env["LIB"] = ";".join(libs)

    # Use raw string literal for the project root macro to avoid
    # Python escaping issues
    project_root_macro = str(ROOT).replace("\\", "\\\\")

    objs = []
    cflags = [
        "/std:c++17", "/EHsc", "/MD", "/nologo", "/O2",
        f'/D', f'AAMED_OPENCV_PROJECT_ROOT="{project_root_macro}"',
        "/c",
    ]

    for src_name in SOURCES:
        src_path = ROOT / "src" / src_name
        obj_path = build_dir / src_name.replace(".cpp", ".obj")
        cmd = [str(cl)] + cflags + [f"/Fo{obj_path}", str(src_path)]
        print(f"  [{len(objs)+1}/{len(SOURCES)}] {src_name}")
        result = subprocess.run(cmd, capture_output=True, text=True,
                                env=env, timeout=120)
        if result.returncode != 0:
            print(f"  ERROR: {src_name}")
            print(result.stderr)
            return 1
        objs.append(str(obj_path))

    print(f"Linking {len(objs)} objs -> aamed_demo.exe")
    link_cmd = [str(link), "/nologo",
                f"/OUT:{bin_dir / 'aamed_demo.exe'}"] + objs + [
                    "opencv_world4120.lib", "kernel32.lib", "user32.lib"]
    result = subprocess.run(link_cmd, capture_output=True, text=True,
                            env=env, timeout=120)
    if result.returncode != 0:
        print("LINK ERROR:")
        print(result.stderr)
        return 1

    print(f"SUCCESS: {bin_dir / 'aamed_demo.exe'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
