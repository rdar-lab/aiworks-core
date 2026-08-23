from .executor import WorkerExecutor
from ..models import Session


def create_executor(session: Session) -> WorkerExecutor:
    """Return the appropriate WorkerExecutor for *session*.

    aiworks-core ships only the base WorkerExecutor class; the host
    application must implement and register this function for the session
    types it defines (e.g. type_a, type_b, etc.).

    Raise NotImplementedError when called before the host app registers
    its own implementation.
    """
    raise NotImplementedError(
        f"aiworks-core create_executor() is not implemented. "
        f"The host application must override this function to map "
        f"session.session_type={session.session_type!r} to the appropriate "
        f"WorkerExecutor subclass. "
        f"Register the implementation via aiworks_core.apps.AiWorksCoreConfig "
        f"or by replacing this function at module load time."
    )
