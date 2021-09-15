"""System/OS global variables."""

import platform

WINDOWS_PLATFORM = platform.system() == "Windows"
MAC_M1_PLATFORM = (
    platform.platform() == "Darwin" and platform.machine().startswith("arm")
)
