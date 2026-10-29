#!/usr/bin/env python3
"""Run each console-control test on a private console, never CTest/editor's console."""
import subprocess
import sys


def run_isolated(helper, event):
    process = None
    try:
        process = subprocess.Popen(
            [helper, event], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        try:
            stdout, stderr = process.communicate(timeout=8)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("isolated {} helper timed out".format(event)) from error
        if process.returncode != 0:
            raise RuntimeError("isolated {} helper failed ({}): {}{}".format(
                event, process.returncode, stdout.decode(errors="replace"),
                stderr.decode(errors="replace")))
        sys.stdout.buffer.write(stdout)
        sys.stderr.buffer.write(stderr)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main():
    if len(sys.argv) != 2:
        raise RuntimeError("expected helper executable path")
    for event in ("ctrl-c", "ctrl-break"):
        run_isolated(sys.argv[1], event)


if __name__ == "__main__":
    main()
