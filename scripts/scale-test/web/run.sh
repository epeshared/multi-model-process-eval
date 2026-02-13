#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO_ROOT"

usage() {
	cat <<'EOF'
Usage:
  ./scripts/scale-test/web/run.sh [--restart|--no-restart] [PORT]

Defaults:
  PORT=30200
  If PORT is already used by an existing scale-test web server, this script
  will restart it (terminate the old process and start a new one).

Tips:
  - If PORT is used by another process, and you didn't pass PORT explicitly,
    the script will auto-pick the next free port.
EOF
}

PORT=30200
RESTART=1
USER_PORT=0

while [[ $# -gt 0 ]]; do
	case "$1" in
		-h|--help)
			usage
			exit 0
			;;
		--restart)
			RESTART=1
			shift
			;;
		--no-restart)
			RESTART=0
			shift
			;;
		--*)
			echo "error: unknown option: $1" >&2
			usage >&2
			exit 2
			;;
		*)
			PORT="$1"
			USER_PORT=1
			shift
			;;
	esac
done

if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
	echo "error: PORT must be an integer (got: $PORT)" >&2
	exit 2
fi

get_listener_pid() {
	local p="$1"
	ss -H -ltnp "sport = :${p}" 2>/dev/null | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n 1
}

port_in_use() {
	local p="$1"
	ss -H -ltn "sport = :${p}" 2>/dev/null | grep -q .
}

restart_if_ours() {
	local p="$1"
	local pid
	pid="$(get_listener_pid "$p")"
	[[ -n "$pid" ]] || return 1

	local cmd
	cmd="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
	if [[ "$cmd" != *"scripts/scale-test/web/server.py"* ]]; then
		return 1
	fi

	echo "[run.sh] port ${p} is in use by scale-test web server (pid=${pid}); restarting..." >&2
	kill "${pid}" 2>/dev/null || true

	# Wait a bit for the socket to be released.
	for _ in {1..40}; do
		if ! port_in_use "$p"; then
			return 0
		fi
		sleep 0.1
	done

	echo "[run.sh] graceful stop timed out; sending SIGKILL to pid=${pid}" >&2
	kill -9 "${pid}" 2>/dev/null || true
	for _ in {1..20}; do
		if ! port_in_use "$p"; then
			return 0
		fi
		sleep 0.1
	done

	return 1
}

if port_in_use "$PORT"; then
	if [[ "$RESTART" == "1" ]] && restart_if_ours "$PORT"; then
		:
	elif [[ "$USER_PORT" == "0" ]]; then
		# Default port is taken by something else; pick the next free port.
		base="$PORT"
		for cand in $(seq "$base" $((base + 50))); do
			if ! port_in_use "$cand"; then
				echo "[run.sh] port ${base} is busy; using ${cand} instead" >&2
				PORT="$cand"
				break
			fi
		done
		if port_in_use "$PORT"; then
			echo "error: no free port found in range ${base}..$((base + 50))" >&2
			exit 1
		fi
	else
		pid="$(get_listener_pid "$PORT")"
		if [[ -n "$pid" ]]; then
			echo "error: port ${PORT} is already in use (pid=${pid})." >&2
			echo "  - To restart if it's our server: ./scripts/scale-test/web/run.sh --restart ${PORT}" >&2
			echo "  - Or pick another port:          ./scripts/scale-test/web/run.sh 30201" >&2
		else
			echo "error: port ${PORT} is already in use." >&2
		fi
		exit 1
	fi
fi

echo "[run.sh] serving on http://127.0.0.1:${PORT} (Ctrl+C to stop)" >&2
exec python3 scripts/scale-test/web/server.py --port "$PORT"
