#!/usr/bin/env bash
# download the official MELD release (MELD.Raw.tar.gz) with full tls verification
#
# web.eecs.umich.edu doesn't send its intermediate certificate, so default
# verification fails. instead of --insecure, we fetch that intermediate from the
# issuer's AIA url, check that it chains to a root the system already trusts, and
# pass a combined bundle to the download curl call only (--cacert). nothing is
# added to the system trust store.
#
# resumable: rerun the script and it continues from the partial file
#
# usage:
#   bash scripts/download_meld.sh
set -euo pipefail

URL="https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
EXPECTED_BYTES=10878146150
INTERMEDIATE_URL="http://crt.sectigo.com/InCommonRSAOVSSLCA3.crt"
SYSTEM_CA=/etc/ssl/certs/ca-certificates.crt

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CERT_DIR="$ROOT/data/certs"
OUT_DIR="$ROOT/data/raw"
OUT="$OUT_DIR/MELD.Raw.tar.gz"
mkdir -p "$CERT_DIR" "$OUT_DIR"

# build the cert bundle once
if [[ ! -f "$CERT_DIR/bundle.pem" ]]; then
  curl -sSf --max-time 30 -o "$CERT_DIR/InCommonRSAOVSSLCA3.crt" "$INTERMEDIATE_URL"
  openssl x509 -inform DER -in "$CERT_DIR/InCommonRSAOVSSLCA3.crt" -out "$CERT_DIR/InCommonRSAOVSSLCA3.pem"
  # AIA fetches are plain http by design, trust comes from the signature check below
  openssl verify -CAfile "$SYSTEM_CA" "$CERT_DIR/InCommonRSAOVSSLCA3.pem"
  openssl x509 -in "$CERT_DIR/InCommonRSAOVSSLCA3.pem" -noout -subject -issuer -dates -fingerprint -sha256
  cat "$SYSTEM_CA" "$CERT_DIR/InCommonRSAOVSSLCA3.pem" > "$CERT_DIR/bundle.pem"
fi

# a 10 GB download can drop partway, so keep resuming until the size matches
for attempt in $(seq 1 50); do
  have=0
  [[ -f "$OUT" ]] && have=$(stat -c %s "$OUT")
  [[ "$have" -ge "$EXPECTED_BYTES" ]] && break
  echo "[$(date '+%F %T')] attempt $attempt: have $have / $EXPECTED_BYTES bytes"
  # -C - continues from the current size of $OUT
  curl --fail --cacert "$CERT_DIR/bundle.pem" -C - -o "$OUT" "$URL" || sleep 10
done

have=$(stat -c %s "$OUT")
if [[ "$have" -ne "$EXPECTED_BYTES" ]]; then
  echo "size mismatch: got $have, expected $EXPECTED_BYTES" >&2
  exit 1
fi
echo "[$(date '+%F %T')] download complete: $have bytes"
sha256sum "$OUT" | tee "$OUT_DIR/MELD.Raw.tar.gz.sha256"
echo "DONE"
