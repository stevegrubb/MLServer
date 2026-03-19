ARG BUILDER_BASE_IMAGE="registry.access.redhat.com/ubi9/ubi-minimal"
ARG RUNTIME_BASE_IMAGE="registry.access.redhat.com/ubi9/ubi-minimal"
ARG RUNTIMES="lightgbm onnx sklearn xgboost"

FROM ${BUILDER_BASE_IMAGE} AS wheel-builder

ARG RUNTIMES
ARG POETRY_VERSION="2.1.1"
ARG PYTHON_VERSION=3.12

WORKDIR /opt/mlserver

COPY ./hack/build-wheels.sh ./hack/build-wheels.sh
COPY ./mlserver ./mlserver
COPY ./runtimes ./runtimes
COPY \
    pyproject.toml \
    poetry.lock \
    README.md \
    ./

RUN microdnf update -y && \
    microdnf install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-pip && \
    microdnf clean all && \
    alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
    ln -sf /usr/bin/pip${PYTHON_VERSION} /usr/bin/pip3 && \
    ln -sf /usr/bin/pip${PYTHON_VERSION} /usr/bin/pip && \
    pip install --upgrade pip wheel setuptools

# Install Poetry, build wheels and export constraints.txt file
RUN pip install poetry==$POETRY_VERSION && \
    pip install poetry-plugin-export && \
    ./hack/build-wheels.sh /opt/mlserver/dist "$RUNTIMES" && \
    poetry export --with all-runtimes \
        --without-hashes \
        --format constraints.txt \
        -o /opt/mlserver/dist/constraints.txt

FROM ${RUNTIME_BASE_IMAGE}

ARG RUNTIMES
ARG PYTHON_VERSION=3.12

# Set a few default environment variables, including `LD_LIBRARY_PATH`
# (required to use GKE's injected CUDA libraries).
# NOTE: When updating between major Python versions, update the PYTHON_VERSION ARG above.
ENV MLSERVER_MODELS_DIR=/mnt/models \
    MLSERVER_ENV_TARBALL=/mnt/models/environment.tar.gz \
    MLSERVER_PATH=/opt/mlserver \
    HF_HOME=/opt/mlserver/.cache \
    NUMBA_CACHE_DIR=/opt/mlserver/.cache

# Install some base dependencies required for some libraries.
# Libomp is needed by the LightGBM runtime.
RUN microdnf update -y && \
    microdnf install -y \
        libgomp \
        shadow-utils \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-pip && \
    microdnf clean all

WORKDIR /opt/mlserver

# Create user and fix permissions
# NOTE: We need to make /opt/mlserver world-writable so that the image is
# compatible with random UIDs.
RUN mkdir -p $MLSERVER_PATH && \
    useradd -u 1000 -s /bin/bash mlserver -d $MLSERVER_PATH && \
    chown -R 1000:0 $MLSERVER_PATH && \
    chmod 1776 $MLSERVER_PATH

# Configure the new python as default
RUN --mount=type=bind,from=wheel-builder,src=/opt/mlserver/dist,target=./dist \
    alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
    ln -sf /usr/bin/pip${PYTHON_VERSION} /usr/bin/pip3 && \
    ln -sf /usr/bin/pip${PYTHON_VERSION} /usr/bin/pip &&\
    pip install --upgrade pip wheel setuptools && \
    for _runtime in $RUNTIMES; do \
        _wheel="./dist/mlserver_$_runtime-"*.whl; \
        echo "--> Installing $_wheel..."; \
        pip install $_wheel --constraint ./dist/constraints.txt; \
    done && \
    pip install $(ls "./dist/mlserver-"*.whl) --constraint ./dist/constraints.txt && \
    rm -rf /root/.cache/pip

COPY ./licenses/license.txt .
COPY ./licenses/license.txt /licenses/

USER 1000

# MLServer starts
CMD ["/bin/sh", "-c", "mlserver start $MLSERVER_MODELS_DIR"]
