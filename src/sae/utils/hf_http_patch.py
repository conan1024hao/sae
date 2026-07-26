"""Work around huggingface_hub's retry loop closing the client it is still using.

`huggingface_hub.utils._http._http_backoff_base` binds the shared client once, before
its retry loop:

    client = get_session()
    while True:
        try:
            response = client.request(...)
        except retry_on_exceptions as err:
            if isinstance(err, httpx.ConnectError):
                close_session()     # closes the client `client` still points at
        time.sleep(...)             # loop back and reuse the closed client

So any ConnectError that gets retried raises
`RuntimeError: Cannot send a request, as the client has been closed` on the next
attempt instead of retrying. Streaming FineVisionMax throws SSL handshake timeouts
(which are ConnectErrors) regularly, which took down all eight ranks every few
hundred steps, at one point after only 46.

The patch keeps `close_session`'s contract -- the next `get_session()` still builds a
fresh client -- but stops it from closing the instance that in-flight retries are
holding, so the retry can actually run. The displaced client is left to garbage
collection.

Verified against huggingface_hub 1.18.0.
"""

import logging

logger = logging.getLogger(__name__)


def apply() -> bool:
    """Install the patch. Returns False if this hf_hub version does not need it."""
    from huggingface_hub.utils import _http

    if not hasattr(_http, "_GLOBAL_CLIENT") or not hasattr(_http, "close_session"):
        logger.warning("huggingface_hub layout changed; skipping shared-client patch")
        return False

    def close_session_without_closing() -> None:
        _http._GLOBAL_CLIENT = None

    close_session_without_closing.__doc__ = _http.close_session.__doc__
    _http.close_session = close_session_without_closing
    return True
