# Build the circt4coral Docker image
# Usage: make -f docker/Makefile build

IMAGE := circt4coral:0.1.0

build:
	docker build -f docker/Dockerfile.circt4coral -t $(IMAGE) .

run:
	docker run --rm -it $(IMAGE) bash

test: build
	docker run --rm $(IMAGE) bash -c "cd /workspace/tests && bash ci.sh"

push: build
	docker push $(IMAGE)

clean:
	docker rmi $(IMAGE) || true
