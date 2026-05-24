#!/bin/bash

echo "Building tera_renderer from source"

curl https://sh.rustup.rs -sSf | sh -s -- -y

tera_renderer_dir="/acados/interfaces/acados_template/tera_renderer/"
echo $tera_renderer_dir
cd $tera_renderer_dir

source "$HOME/.cargo/env"

cargo build --release --verbose
echo "finished building tera_renderer"

file_name="/acados/bin/t_renderer"
echo $file_name
if [-e "$file_name" ] || [-L "$file_name"]; then
  rm "$file_name"
fi

mv target/release/t_renderer /acados/bin/

echo "cleaning build"
cargo clean

echo "uninstalling rust compiler"
rustup self uninstall -y

exit 0