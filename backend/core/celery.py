# celery.py
from __future__ import absolute_import, unicode_literals
import os
import sys
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

app = Celery('core')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.conf.update(
    broker_connection_retry_on_startup=True,
)
app.autodiscover_tasks()

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))