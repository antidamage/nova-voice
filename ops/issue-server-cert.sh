#!/usr/bin/env bash
set -euo pipefail

# Issue one TLS *server* certificate from the existing household CA.
#
# The browser voice satellite needs the dashboard served over HTTPS/WSS from a
# certificate the household devices trust (browser mic capture requires a secure
# context). This issues a serverAuth certificate with the dashboard's hostnames
# / IPs as SANs. Run it on Iridium (where the CA lives) as a privileged operator.
# It never creates or replaces the CA.
#
# Install the resulting server.crt/server.key on the dashboard host and point
# NOVA_DASHBOARD_TLS_CERT / NOVA_DASHBOARD_TLS_KEY (and your reverse proxy) at
# them. Install ca.crt as a trusted root on each viewing device once.

usage() {
  cat >&2 <<'EOF'
Usage: issue-server-cert.sh OUTPUT_DIR SAN [SAN ...]

Each SAN is either "DNS:name" or "IP:addr" (a bare value is treated as DNS).
OUTPUT_DIR must not already exist; it receives ca.crt, server.crt, server.key.
The certificate CN is the first SAN's value.

Example:
  issue-server-cert.sh /tmp/nova-dashboard-tls DNS:nova.local DNS:dashboard.local IP:192.168.8.14

Override NOVA_VOICE_CA_DIR only when the existing CA is stored elsewhere.
EOF
  exit 2
}

output_dir="${1:-}"
shift || true
[[ -n "$output_dir" && "$#" -ge 1 ]] || usage
[[ ! -e "$output_dir" ]] || {
  echo "Refusing to overwrite existing output directory: $output_dir" >&2
  exit 2
}

# Normalise SANs (bare value => DNS:) and derive the CN from the first entry.
san_lines=()
cn=""
for entry in "$@"; do
  case "$entry" in
    DNS:* | IP:*) normalized="$entry" ;;
    *) normalized="DNS:$entry" ;;
  esac
  san_lines+=("$normalized")
  [[ -n "$cn" ]] || cn="${normalized#*:}"
done
san_csv="$(IFS=,; echo "${san_lines[*]}")"

ca_dir="${NOVA_VOICE_CA_DIR:-/etc/nova-voice/ca}"
ca_cert="$ca_dir/ca.crt"
ca_key="$ca_dir/ca.key"
[[ -r "$ca_cert" && -r "$ca_key" ]] || {
  echo "Existing household CA is not readable under $ca_dir" >&2
  exit 2
}

umask 077
install -d -m 0700 "$output_dir"
trap 'rm -f "$output_dir/server.csr" "$output_dir/server.ext"' EXIT

openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 \
  -out "$output_dir/server.key"
openssl req -new -sha256 -key "$output_dir/server.key" \
  -subj "/CN=$cn" -out "$output_dir/server.csr"
cat >"$output_dir/server.ext" <<EOF
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=$san_csv
EOF
openssl x509 -req -sha256 -days 825 \
  -in "$output_dir/server.csr" -CA "$ca_cert" -CAkey "$ca_key" \
  -set_serial "0x$(openssl rand -hex 16)" -extfile "$output_dir/server.ext" \
  -out "$output_dir/server.crt"
install -m 0644 "$ca_cert" "$output_dir/ca.crt"
chmod 0600 "$output_dir/server.key"
chmod 0644 "$output_dir/ca.crt" "$output_dir/server.crt"

echo "Issued server certificate ($san_csv) in $output_dir"
