#!/bin/sh
set -e

envsubst '$EDGE_RATE $EDGE_BURST $EDGE_MAX_BODY $EDGE_CONNECT_TIMEOUT $EDGE_READ_TIMEOUT $EDGE_SEND_TIMEOUT' \
  < /etc/nginx/templates/cathode.conf.template \
  > /etc/nginx/conf.d/default.conf

exec nginx -g "daemon off;"
