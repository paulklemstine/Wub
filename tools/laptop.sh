#!/bin/bash
# Laptop (RTX 4050, WSL) ops via reverse tunnel. Password via env, never in repo:
#   export WUB_LAPTOP_PASS='...'
#   printf '%s' 'nvidia-smi' | tools/laptop.sh exec
#   tools/laptop.sh put <local> <remote>
#   tools/laptop.sh get <remote> <local>
# Tunnel prerequisite (run IN WSL on laptop):
#   ssh -f -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
#       -R 2223:localhost:22 raver1975@192.168.1.118
set -u
: "${WUB_LAPTOP_PASS:?set WUB_LAPTOP_PASS first}"

MODE="${1:-}"; SRC="${2:-}"; DST="${3:-}"
export MODE SRC DST

DATA_FILE=""
if [ "$MODE" = "exec" ]; then
  DATA_FILE="$(mktemp /tmp/opencode/laptop_data.XXXXXX)"
  export DATA_FILE
  cat > "$DATA_FILE"   # consume the piped remote script (python's stdin is the heredoc below)
fi

python3 <<'PY'
import pexpect, os, base64, sys
mode = os.environ['MODE']
if mode == 'exec':
    with open(os.environ['DATA_FILE'], 'rb') as f:
        raw = f.read()
    os.unlink(os.environ['DATA_FILE'])
    inner = 'echo ' + base64.b64encode(raw).decode() + ' | base64 -d | bash'
    c = pexpect.spawn('ssh', ['-p', '2223', '-o', 'StrictHostKeyChecking=no',
                      '-o', 'ConnectTimeout=15', 'raver1975@127.0.0.1', inner],
                      timeout=240, encoding='utf-8')
    while True:
        i = c.expect(['password:', 'Password:', pexpect.EOF, pexpect.TIMEOUT])
        if i in (0, 1):
            c.sendline(os.environ['WUB_LAPTOP_PASS'])
            continue
        break
    print(c.before or '')
elif mode in ('put', 'get'):
    src, dst = os.environ['SRC'], os.environ['DST']
    if mode == 'put':
        args = ['scp', '-P', '2223', '-o', 'StrictHostKeyChecking=no',
                '-o', 'ConnectTimeout=15', '-r', src, 'raver1975@127.0.0.1:' + dst]
    else:
        args = ['scp', '-P', '2223', '-o', 'StrictHostKeyChecking=no',
                '-o', 'ConnectTimeout=15', '-r', 'raver1975@127.0.0.1:' + src, dst]
    c = pexpect.spawn(args[0], args[1:], timeout=1200, encoding='utf-8')
    while True:
        i = c.expect(['password:', 'Password:', pexpect.EOF, pexpect.TIMEOUT])
        if i in (0, 1):
            c.sendline(os.environ['WUB_LAPTOP_PASS'])
            continue
        break
    print((c.before or '')[-300:])
else:
    print('usage: laptop.sh {exec|put|get} ...', file=sys.stderr)
    sys.exit(2)
PY
