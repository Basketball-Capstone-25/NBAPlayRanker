"""ASGI entrypoint for local development and container startup."""

from application.api_coordination.app import app

__all__ = ["app"]