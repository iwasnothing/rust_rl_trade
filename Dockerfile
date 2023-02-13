FROM rust:latest as build

WORKDIR /app
RUN apt update &&\
    rm -rf ~/.cache &&\
    apt clean all &&\
    apt install -y cmake &&\
    apt install -y clang


# install libtorch=1.9.0
# https://pytorch.org/get-started/locally/
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip -O libtorch.zip
RUN unzip -o libtorch.zip
ENV LIBTORCH /app/libtorch
ENV LD_LIBRARY_PATH /app/libtorch/lib:$LD_LIBRARY_PATH
COPY src src
COPY Cargo.toml Cargo.toml
RUN cargo build --release

FROM rust:latest

WORKDIR /app
RUN apt update &&\
    rm -rf ~/.cache &&\
    apt clean all &&\
    apt install -y cmake &&\
    apt install -y clang

COPY --from=build /app/libtorch.zip .
RUN unzip -o libtorch.zip
ENV LIBTORCH /app/libtorch
ENV LD_LIBRARY_PATH /app/libtorch/lib:$LD_LIBRARY_PATH

COPY --from=build /app/target/release/rust_rl .

CMD ["./rust_rl"]
