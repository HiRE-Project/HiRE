"""
Small wrappers around wandb so interrupted runs do not hang during Python atexit.
"""

import logging
import os
import signal
import subprocess
from contextlib import contextmanager
from typing import Optional

import wandb

log = logging.getLogger(__name__)


def init_wandb(*args, **kwargs):
    """Initialize wandb with quiet service settings when the installed version supports them."""
    settings = kwargs.pop("settings", None)
    if settings is None:
        settings_kwargs = {"quiet": True}
        try:
            settings = wandb.Settings(**settings_kwargs)
        except TypeError:
            # Older wandb versions may not support the same Settings fields.
            settings = None

    if settings is not None:
        kwargs["settings"] = settings

    return wandb.init(*args, **kwargs)


@contextmanager
def _timeout(seconds: Optional[float]):
    if not seconds or seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _handle_timeout(signum, frame):
        raise TimeoutError(f"wandb finish timed out after {seconds} seconds")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _kill_current_wandb_service():
    """Terminate wandb service processes that belong to this Python process."""
    pid = os.getpid()
    patterns = [
        rf"wandb service .*--pid {pid}\b",
        rf"wandb-service\(.*-{pid}-",
    ]
    for pattern in patterns:
        subprocess.run(
            ["pkill", "-TERM", "-f", pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def finish_wandb(exit_code: Optional[int] = None, timeout: float = 5.0):
    """
    Finish the active wandb run, but do not let upload/service teardown block exit forever.

    On Ctrl-C the normal wandb atexit hook can get stuck waiting for the internal service.
    A short explicit finish keeps successful runs graceful and interrupted runs responsive.
    """
    if wandb.run is None:
        return

    try:
        with _timeout(timeout):
            wandb.finish(exit_code=exit_code, quiet=True)
    except BaseException as exc:
        log.warning("wandb finish did not complete cleanly: %s", exc)
        _kill_current_wandb_service()
