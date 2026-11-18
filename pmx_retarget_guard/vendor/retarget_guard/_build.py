"""Lazy C99 compilation into a user-writable cache; no numerical code changes."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import sys
import re

SOURCES = Path(__file__).resolve().parent / "c"


def build(name, *, force=False):
    if name not in ("guard_core", "guard_core_cloth", "deform_core"):
        raise ValueError("Unknown C library")
    cache = Path(os.environ.get("RETARGET_GUARD_CACHE", str(
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "retarget_guard")))
    cache.mkdir(parents=True, exist_ok=True)
    source = SOURCES / (name + ".c")
    suffix = ".dll" if sys.platform == "win32" else ".so"
    dest = cache / (name + suffix)
    if not force and dest.exists() and dest.stat().st_mtime >= source.stat().st_mtime:
        return dest
    flags = ["-std=c99", "-O3", "-fno-math-errno"]
    if name != "deform_core":
        flags.append("-ffp-contract=fast")
    march = os.environ.get("RETARGET_GUARD_MARCH", "")
    if march:
        flags.append("-march=" + march)
    if sys.platform == "win32":
        flags += ["-shared", "-include", str(SOURCES / "windows_compat.h")]
        exports = re.findall(r"^(?:void\s*\*?|int|double|float)\s+(guard_\w+|deform_\w+)\s*\(",
                             source.read_text(), re.MULTILINE)
        # Pointer-returning declarations can omit whitespace after the star.
        exports += re.findall(r"^void\s*\*(guard_\w+|deform_\w+)\s*\(",
                              source.read_text(), re.MULTILINE)
        exports = [symbol for symbol in exports if symbol != "deform_benchmark"]
        flags += ["-Wl,/export:" + symbol for symbol in sorted(set(exports))]
    else:
        flags += ["-fPIC", "-shared"]
    flags += shlex.split(os.environ.get("CFLAGS", ""))
    compiler = shlex.split(os.environ.get("CC", "clang" if sys.platform == "win32" else "cc"))
    # Unique build file and atomic rename support concurrent first use.
    fd, temporary = tempfile.mkstemp(prefix=name + "-", suffix=suffix, dir=cache)
    os.close(fd)
    try:
        libraries = [] if sys.platform == "win32" else ["-lm"]
        subprocess.run([*compiler, *flags, str(source), *libraries, "-o", temporary], check=True)
        os.replace(temporary, dest)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return dest


def library_path(name):
    return build(name)


if __name__ == "__main__":
    for library in ("guard_core", "guard_core_cloth", "deform_core"):
        print(build(library, force=True))
