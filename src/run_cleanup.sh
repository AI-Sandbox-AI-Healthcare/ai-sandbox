#!/usr/bin/env bash

# Default age threshold in minutes (change if needed)
AGE_MINUTES=${1:-30}
USER_NAME=$(whoami)

echo "🔍 Looking for GPU processes by user '$USER_NAME' older than $AGE_MINUTES minutes..."

# Get GPU-using PIDs via nvidia-smi and filter by user
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -n 1 ps -o pid=,etime=,user=,cmd= -p | grep "$USER_NAME" | awk -v min=$AGE_MINUTES '
function to_minutes(t,   a, h, m) {
  split(t, a, "-")
  if (length(a) == 2) { h = a[2]; d = a[1] } else { h = a[1]; d = 0 }
  split(h, a, ":")
  return d*1440 + a[1]*60 + a[2]
}
{
  if (to_minutes($2) > min) print $1
}')

if [ -z "$PIDS" ]; then
  echo "✅ No stale GPU processes found."
else
  echo "⚠️ Killing stale GPU processes:"
  echo "$PIDS"
  echo "$PIDS" | xargs -r kill -9
fi
