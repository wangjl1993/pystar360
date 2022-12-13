import contextlib
import threading
import pkg_resources as pkg

from pathlib import Path
from pystar360.utilities._logger import d_logger

COMPANY_STRING = """
        ************************************************************************************
        ************************************************************************************
        *** ██████╗ ██╗   ██╗███████╗████████╗ █████╗ ██████╗ ██████╗  ██████╗  ██████╗  ***
        *** ██╔══██╗╚██╗ ██╔╝██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚════██╗██╔════╝ ██╔═████╗ ***
        *** ██████╔╝ ╚████╔╝ ███████╗   ██║   ███████║██████╔╝ █████╔╝███████╗ ██║██╔██║ ***
        *** ██╔═══╝   ╚██╔╝  ╚════██║   ██║   ██╔══██║██╔══██╗ ╚═══██╗██╔═══██╗████╔╝██║ ***
        *** ██║        ██║   ███████║   ██║   ██║  ██║██║  ██║██████╔╝╚██████╔╝╚██████╔╝ ***
        *** ╚═╝        ╚═╝   ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚═════╝  ***
        ***                  溯星360列车智能故障检测系统-PyStar360                       ***
        ************************************************************************************
        ************************************************************************************
        """


def print_product_info(logger=None):
    if logger:
        logger.info(COMPANY_STRING)
    else:
        d_logger.info(COMPANY_STRING)


class TryExcept(contextlib.ContextDecorator):
    # TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg=''):
        self.msg = msg

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if value:
            d_logger.info(f'{self.msg}{value}')
        return True


@TryExcept()
def check_requirements(requirements=(), exclude=()):
    # Check installed dependencies
    if isinstance(requirements, Path):  # requirements.txt file
        file = requirements.resolve()
        assert file.exists(), f"{file} not found, check failed."
        with file.open() as f:
            requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(f) if x.name not in exclude]
    elif isinstance(requirements, str):
        requirements = [requirements]

    s = ''
    n = 0
    for r in requirements:
        try:
            pkg.require(r)
        except (pkg.VersionConflict, pkg.DistributionNotFound):  # exception if requirements not met
            s += f'"{r}"'
            n += 1
            d_logger.info(f"{n} packages missing: {s}")


def threadingDecorator(func):
    def wrap_func(*args, **kwargs):
        th = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        th.start()
        return

    return wrap_func
