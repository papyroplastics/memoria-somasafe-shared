#!/bin/sh
# This script can be used to generate an ECDSA key pair over a P-256 curve.
cd "$(dirname "$0")" || exit

private="gen/server-private-key.pem"
public="gen/server-public-key.pem"

# Generate private key
openssl ecparam -name prime256v1 -genkey -noout -out "$private"

# Extract public key
openssl ec -in "$private" -pubout -out "$public"

# Display the public key
openssl ec -in "$private" -pubout -text
