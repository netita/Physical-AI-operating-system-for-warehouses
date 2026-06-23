#!/bin/sh
set -e

sed \
  -e "s|\${ALERT_GMAIL_FROM}|${ALERT_GMAIL_FROM}|g" \
  -e "s|\${ALERT_GMAIL_APP_PASSWORD}|${ALERT_GMAIL_APP_PASSWORD}|g" \
  -e "s|\${ALERT_EMAIL_TO}|${ALERT_EMAIL_TO}|g" \
  /etc/alertmanager/alertmanager.yml.tmpl > /tmp/alertmanager.yml

exec /bin/alertmanager \
  --config.file=/tmp/alertmanager.yml \
  --storage.path=/alertmanager \
  --web.listen-address=0.0.0.0:9093 \
  --web.external-url=http://localhost:9093 \
  --log.level=info
