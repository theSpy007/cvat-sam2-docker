# Local Django settings override for CVAT.
# Mounted into the container at /home/django/cvat/settings/local.py
# and loaded via DJANGO_SETTINGS_MODULE=cvat.settings.local

import os
from .production import *  # noqa: F401, F403

# Allow POST requests from the browser-facing origin (required for CSRF checks)
# The origin includes scheme + host + port, e.g. http://localhost:8080
_base_url = os.getenv("CVAT_BASE_URL", "http://localhost:8080").rstrip("/")
CSRF_TRUSTED_ORIGINS = [_base_url]
