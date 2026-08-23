import httpx
from pathlib import Path


def get_library_path() -> Path:
    """Return the absolute path to the aiworks_core library root directory.

    Use this to locate resources bundled with the library (e.g. schemas,
    templates, static assets) that live inside the aiworks_core package.
    """
    return Path(__file__).resolve().parent


def get_library_resource_path(*segments: str) -> Path:
    """Return an absolute path to a resource file within the aiworks_core library.

    Args:
        *segments: Path components relative to the library root
                   (e.g. "resources/schema/ppt_schema.json")

    Returns:
        Absolute Path to the requested resource

    Example:
        schema_path = get_library_resource_path("resources/schema/ppt_schema.json")
    """
    return get_library_path().joinpath(*segments)


def chunked(iterable, chunk_size):
    """Yield successive chunks of *size* from *iterable*."""
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def close_connections():
    from django.db import connection
    from django.db import close_old_connections

    try:
        connection.close()
    except Exception as _:
        pass

    try:
        close_old_connections()
    except Exception as _:
        pass


def sync_to_async(func):
    from asgiref.sync import sync_to_async as _sync_to_async

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            close_connections()

    return _sync_to_async(wrapper)


def async_to_sync(func):
    from asgiref.sync import async_to_sync as _async_to_sync

    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        finally:
            close_connections()

    return _async_to_sync(wrapper)


def raise_for_status(response: httpx.Response) -> None:
    """Like response.raise_for_status() but includes the response body in the exception."""
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise httpx.HTTPStatusError(
            f"{response.status_code} {response.text}",
            request=exc.request,
            response=response,
        )
