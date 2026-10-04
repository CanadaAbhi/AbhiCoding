# Compile all examples
gcc -o trustzone_basics trustzone_basics.c -lm
gcc -o trustzone_key_management trustzone_key_management.c -lm
gcc -o trustzone_attestation trustzone_attestation.c -lm
gcc -o trustzone_sealing trustzone_sealing.c -lm
gcc -o trustzone_secure_payment trustzone_secure_payment.c -lm

# Run examples
./trustzone_basics
./trustzone_key_management
./trustzone_attestation
./trustzone_sealing
./trustzone_secure_payment
