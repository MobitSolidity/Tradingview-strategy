#!/bin/sh
set -e
CMD_MODE=${BOT_CMD:-run}
EXTRA_ARGS=${BOT_EXTRA_ARGS:-}
DISABLE_TG=${BOT_DISABLE_TELEGRAM:-}
if [ "$DISABLE_TG" = "true" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --disable-telegram"
fi
exec python ETH_BTC_MTF_LIVE.py --cmd "$CMD_MODE" $EXTRA_ARGS "$@"
