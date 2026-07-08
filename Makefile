
out_dir = gen
private = $(out_dir)/server-private-key.pem
public  = $(out_dir)/server-public-key.pem

setup: $(public) $(private)

$(public): $(private)
	openssl ec -in "$(private)" -pubout -out "$(public)"
	openssl ec -in "$(private)" -pubout -text

$(private):
	openssl ecparam -name prime256v1 -genkey -noout -out "$(private)"
