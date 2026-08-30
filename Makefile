
out_dir = gen
private = $(out_dir)/server-private-key.pem
public  = $(out_dir)/server-public-key.pem
code    = $(out_dir)/code
proto_py = $(code)/dataset_pb2.py

setup: $(public) $(private) $(proto_py)

$(public): $(private)
	openssl ec -in "$(private)" -pubout -out "$(public)"
	openssl ec -in "$(private)" -pubout -text

$(private):
	openssl ecparam -name prime256v1 -genkey -noout -out "$(private)"

# Python bindings for the capture schema, generated where both the backend and the
# firmware harness can import them as `shared.gen.code.dataset_pb2` (PEP 420
# namespace package, resolved through each module's `shared` symlink).
$(proto_py): dataset.proto
	mkdir -p $(code)
	protoc --proto_path=. --python_out=$(code) dataset.proto
