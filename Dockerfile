FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ETH_BTC_MTF_LIVE.py strategy_config.py strategies.json README.md docker_entrypoint.sh healthcheck.sh ./
ENV BOT_INITIAL_EQUITY=1000 \
    TG_ENABLED=false \
    TG_BOT_TOKEN="" \
    TG_CHAT_ID="" \
    BOT_CMD=run \
    BOT_LOG_LEVEL=INFO

ENTRYPOINT ["./docker_entrypoint.sh"]
CMD ["--cmd", "run"]
HEALTHCHECK --interval=5m --timeout=10s --retries=3 CMD ["./healthcheck.sh"]
