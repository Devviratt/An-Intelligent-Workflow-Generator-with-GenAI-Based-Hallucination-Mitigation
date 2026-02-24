"""
Request Logging Middleware — structured per-request telemetry.

Logs every HTTP request with method, path, response time (ms),
and status code.  Uses the stdlib logger — no raw prints.
"""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("src.api.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured access-log middleware.

    Emits one JSON-shaped log line per request::

        {"method": "POST", "path": "/api/v1/generate",
         "duration_ms": 134, "status_code": 200}
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            '{"method": "%s", "path": "%s", "duration_ms": %s, "status_code": %s}',
            request.method,
            request.url.path,
            duration_ms,
            response.status_code,
        )

        return response
