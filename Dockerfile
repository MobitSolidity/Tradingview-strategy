FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ETH_BTC_MTF_LIVE.py README.md ./
ENV BOT_INITIAL_EQUITY=1000 \
    TG_ENABLED=false \
    TG_BOT_TOKEN="" \
    TG_CHAT_ID=""

CMD ["python", "ETH_BTC_MTF_LIVE.py", "--once", "--disable-telegram"]
