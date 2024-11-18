#!/bin/bash
# Installer Rust via rustup
curl https://sh.rustup.rs -sSf | sh -s -- -y
# Ajouter Rust au PATH
source $HOME/.cargo/env
