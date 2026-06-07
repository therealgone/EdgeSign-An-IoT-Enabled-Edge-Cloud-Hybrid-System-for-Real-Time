"""
Headless launcher for pi_sender.py.

Runs the exact same code path as pi_sender.py but disables OpenCV preview/window calls
so it works on Raspberry Pi without display (SSH/headless boot).
"""

import os
import runpy
import cv2


def _disable_opencv_gui():
    def _noop_imshow(*_args, **_kwargs):
        return None

    # Keep running loop forever in headless mode (never triggers 'q' key path).
    def _headless_wait_key(_delay=1):
        return -1

    def _noop_destroy(*_args, **_kwargs):
        return None

    cv2.imshow = _noop_imshow
    cv2.waitKey = _headless_wait_key
    cv2.destroyAllWindows = _noop_destroy


def main():
    _disable_opencv_gui()

    here = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(here, "pi_sender.py")
    if not os.path.exists(target):
        raise FileNotFoundError(f"pi_sender.py not found at: {target}")

    print("[HEADLESS] Starting pi_sender.py with OpenCV preview disabled")
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
