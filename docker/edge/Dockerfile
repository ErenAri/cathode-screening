FROM nginx:1.27-alpine

COPY docker/edge/nginx.conf.template /etc/nginx/templates/cathode.conf.template
COPY docker/edge/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENV EDGE_RATE=10r/s
ENV EDGE_BURST=20
ENV EDGE_MAX_BODY=2m
ENV EDGE_CONNECT_TIMEOUT=5s
ENV EDGE_READ_TIMEOUT=60s
ENV EDGE_SEND_TIMEOUT=60s

ENTRYPOINT ["/entrypoint.sh"]
