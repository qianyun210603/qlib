from typing import cast
from qlib.log import get_module_logger
from arctic.auth import Credential
from arctic.hooks import register_get_auth_hook

try:
    from vnpy.trader.database import SETTINGS, get_database
    from vnpy_arctic.arctic_database import ArcticDatabase
except ImportError:
    pass


logger = get_module_logger("arctic_storage")


def db_symbol_to_qlib(db_symbol: str) -> str:
    """convert db_symbol to qlib symbol

    Parameters
    ----------
    db_symbol : str
        db_symbol

    Returns
    -------
    str
        qlib symbol
    """
    code, exch = db_symbol.split("_")
    exch = "SH" if exch == "SSE" else "SZ"
    return f"{exch}{code}"


def qlib_symbol_to_db(qlib_symbol: str) -> str:
    """convert db_symbol to qlib symbol

    Parameters
    ----------
    db_symbol : str
        db_symbol

    Returns
    -------
    str
        qlib symbol
    """
    exch = "SSE" if qlib_symbol[:2].lower() == "sh" else "SZSE"
    return f"{qlib_symbol[2:]}_{exch}"


def arctic_auth_hook(*_):
    if bool(SETTINGS.get("database.password", "")) and bool(SETTINGS.get("database.user", "")):
        return Credential(
            database="admin",
            user=SETTINGS["database.user"],
            password=SETTINGS["database.password"],
        )
    return None


register_get_auth_hook(arctic_auth_hook)


class ArcticStorageMixin:
    """ArcticStorageMixin, applicable to ArcticXXXStorage
    Subclasses need
    """

    def _get_arctic_store(self):
        db_mgr = get_database()
        return cast(ArcticDatabase, db_mgr).connection
