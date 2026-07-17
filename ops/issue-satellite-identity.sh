#!/usr/bin/env bash
set -euo pipefail

# Issue one Nova Voice client identity from the existing household CA.
# Run this only on Iridium as a privileged operator.  It never creates or
# replaces the CA, and writes each private key to a newly-created output
# directory with restrictive permissions.

usage() {
  cat >&2 <<'EOF'
Usage: issue-satellite-identity.sh IDENTITY OUTPUT_DIR [P12_PASSWORD_FILE]

IDENTITY must be "nocturnium", "indium", "nova-dashboard", or
"browser-diagnostics". OUTPUT_DIR
must not already exist. The directory receives ca.crt, client.crt, and
client.key. Supplying a P12_PASSWORD_FILE additionally creates IDENTITY.p12 for
the macOS satellite or diagnostic browser; the password is read from that file
and never accepted as a command argument.

Override NOVA_VOICE_CA_DIR only when the existing CA is stored elsewhere.
EOF
  exit 2
}

identity="${1:-}"
output_dir="${2:-}"
p12_password_file="${3:-}"
[[ -n "$identity" && -n "$output_dir" ]] || usage
case "$identity" in
  nocturnium | indium | nova-dashboard | browser-diagnostics) ;;
  *) usage ;;
esac
[[ ! -e "$output_dir" ]] || {
  echo "Refusing to overwrite existing output directory: $output_dir" >&2
  exit 2
}
if [[ -n "$p12_password_file" && ! -r "$p12_password_file" ]]; then
  echo "P12 password file is not readable: $p12_password_file" >&2
  exit 2
fi

ca_dir="${NOVA_VOICE_CA_DIR:-/etc/nova-voice/ca}"
ca_cert="$ca_dir/ca.crt"
ca_key="$ca_dir/ca.key"
[[ -r "$ca_cert" && -r "$ca_key" ]] || {
  echo "Existing household CA is not readable under $ca_dir" >&2
  exit 2
}

umask 077
install -d -m 0700 "$output_dir"
trap 'rm -f "$output_dir/client.csr" "$output_dir/client.ext"' EXIT

openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 \
  -out "$output_dir/client.key"
openssl req -new -sha256 -key "$output_dir/client.key" \
  -subj "/CN=$identity" -out "$output_dir/client.csr"
cat >"$output_dir/client.ext" <<EOF
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyAgreement
extendedKeyUsage=clientAuth
subjectAltName=DNS:$identity
EOF
openssl x509 -req -sha256 -days 825 \
  -in "$output_dir/client.csr" -CA "$ca_cert" -CAkey "$ca_key" \
  -set_serial "0x$(openssl rand -hex 16)" -extfile "$output_dir/client.ext" \
  -out "$output_dir/client.crt"
install -m 0644 "$ca_cert" "$output_dir/ca.crt"
chmod 0600 "$output_dir/client.key"
chmod 0644 "$output_dir/ca.crt" "$output_dir/client.crt"

if [[ -n "$p12_password_file" ]]; then
  # macOS Security accepts the OpenSSL legacy/3DES + SHA-1 interchange
  # profile reliably. This bundle is short-lived transfer material; the
  # deployed identity remains an EC P-256 key protected by its Keychain.
  if openssl pkcs12 -help 2>&1 | grep -q -- '-legacy'; then
    openssl pkcs12 -export -legacy -descert -macalg sha1 \
      -inkey "$output_dir/client.key" -in "$output_dir/client.crt" \
      -certfile "$output_dir/ca.crt" -name "$identity" \
      -passout "file:$p12_password_file" -out "$output_dir/$identity.p12"
  else
    # LibreSSL's older exporter already uses the compatible 3DES/SHA-1
    # defaults but does not implement OpenSSL 3's ``-legacy`` switch.
    openssl pkcs12 -export -descert -macalg sha1 \
      -inkey "$output_dir/client.key" -in "$output_dir/client.crt" \
      -certfile "$output_dir/ca.crt" -name "$identity" \
      -passout "file:$p12_password_file" -out "$output_dir/$identity.p12"
  fi
  chmod 0600 "$output_dir/$identity.p12"
fi

echo "Issued $identity client identity in $output_dir"
