#!/bin/sh
set -e
CMD_MODE=${BOT_CMD:-healthcheck}
exec python ETH_BTC_MTF_LIVE.py --cmd "$CMD_MODE"
