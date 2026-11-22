#!/bin/bash
set -e
# Ensure log and data directories are writable by appuser when using bind/volume mounts
chown -R appuser:appuser /app/logs /app/data 2>/dev/null || true
exec gosu appuser "$@"
