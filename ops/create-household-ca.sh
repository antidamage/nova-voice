#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${1:-./household-ca}"
mkdir -p "$OUTPUT"
umask 077

openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out "$OUTPUT/ca.key"
openssl req -x509 -new -sha256 -days 3650 \
  -key "$OUTPUT/ca.key" \
  -subj "/CN=Nova Voice Household CA" \
  -out "$OUTPUT/ca.crt"

issue_certificate() {
  local name="$1"
  local usage="$2"
  local san="$3"
  openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out "$OUTPUT/$name.key"
  openssl req -new -sha256 -key "$OUTPUT/$name.key" -subj "/CN=$name" -out "$OUTPUT/$name.csr"
  {
    echo "basicConstraints=critical,CA:FALSE"
    echo "keyUsage=critical,digitalSignature,keyAgreement"
    echo "extendedKeyUsage=$usage"
    echo "subjectAltName=$san"
  } >"$OUTPUT/$name.ext"
  openssl x509 -req -sha256 -days 825 \
    -in "$OUTPUT/$name.csr" \
    -CA "$OUTPUT/ca.crt" \
    -CAkey "$OUTPUT/ca.key" \
    -CAcreateserial \
    -extfile "$OUTPUT/$name.ext" \
    -out "$OUTPUT/$name.crt"
  rm -f "$OUTPUT/$name.csr" "$OUTPUT/$name.ext"
}

# The server SAN must match the deployed voice-server hostname/IP
# (see PRIVATEREF.md#1.1), e.g. NOVA_VOICE_SERVER_SAN='DNS:voice.local,IP:10.0.0.5'.
: "${NOVA_VOICE_SERVER_SAN:?Set NOVA_VOICE_SERVER_SAN to the voice server SubjectAltName (see PRIVATEREF.md#1.1)}"
issue_certificate iridium serverAuth "$NOVA_VOICE_SERVER_SAN"
issue_certificate nocturnium clientAuth "DNS:nocturnium"
issue_certificate indium clientAuth "DNS:indium"
chmod 600 "$OUTPUT"/*.key
chmod 644 "$OUTPUT"/*.crt
echo "Created one server and two client identities in $OUTPUT"
