#!/usr/bin/env python3
"""Emit the API's form-field contract as JSON.

For every endpoint that accepts a form/multipart body, lists the accepted field
names. This is the source of truth the R client is checked against: if the
client sends a field not listed here, that's the drift class of bug that took
the platform down. The R repo vendors the emitted file and validates its
requests against it in CI.

Usage:  python scripts/dump_contract.py > dl_api_contract.json
"""
import json
import os
import sys


def _resolve_props(schema: dict, node: dict) -> dict:
    """Return the properties of a schema node, following a $ref if present."""
    if "$ref" in node:
        ref = node["$ref"].split("/")[-1]
        node = schema.get("components", {}).get("schemas", {}).get(ref, {})
    return node.get("properties", {})


def extract_contract(app) -> dict:
    schema = app.openapi()
    contract = {}
    for path, methods in schema.get("paths", {}).items():
        for method, op in methods.items():
            body = op.get("requestBody", {})
            content = body.get("content", {})
            form = content.get("multipart/form-data") or content.get(
                "application/x-www-form-urlencoded"
            )
            if not form:
                continue
            props = _resolve_props(schema, form.get("schema", {}))
            contract.setdefault(path, {})[method.upper()] = sorted(props.keys())
    return contract


def main():
    # Avoid needing a live broker just to import the app.
    os.environ.setdefault("CELERY_BROKER_URL", "memory://")
    os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")
    # Ensure the repo root (parent of scripts/) is importable.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import fastapi_app

    contract = extract_contract(fastapi_app.app)
    json.dump(contract, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
