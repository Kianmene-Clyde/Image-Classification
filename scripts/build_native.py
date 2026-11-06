"""Compiler la bibliothèque C++17 avec MSVC, Clang, GCC ou CMake."""
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "build"
    out.mkdir(exist_ok=True)
    source = str(root / "native" / "signlab.cpp")
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler:
        suffix = "signlab_native.dll" if sys.platform == "win32" else "libsignlab_native.dylib" if sys.platform == "darwin" else "libsignlab_native.so"
        command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-shared", source, "-o", str(out / suffix)]
        if sys.platform != "win32":
            command.insert(1, "-fPIC")
    elif shutil.which("cl"):
        command = ["cl", "/nologo", "/std:c++17", "/O2", "/EHsc", "/LD", source, f"/Fe:{out / 'signlab_native.dll'}"]
    elif shutil.which("cmake"):
        subprocess.run(["cmake", "-S", str(root / "native"), "-B", str(out / "cmake")], check=True)
        subprocess.run(["cmake", "--build", str(out / "cmake"), "--config", "Release"], check=True)
        for path in (out / "cmake").rglob("*signlab_native.*"):
            if path.suffix in {".dll", ".so", ".dylib"}:
                shutil.copy2(path, out / path.name)
        return 0
    else:
        print("Aucun compilateur C++17 trouvé. Windows : ouvrir 'Developer PowerShell for VS' avec les outils C++ installés, puis relancer cette commande. Linux : installer g++. macOS : outils Xcode.", file=sys.stderr)
        return 2
    subprocess.run(command, cwd=out, check=True)
    print("Bibliothèque construite dans", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
