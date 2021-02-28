let
  pkgs = import <nixpkgs> {};
in
pkgs.mkShell {
  nativeBuildInputs = with pkgs; [ clang_11 cmake ];
  buildInputs = with pkgs; [ eigen ];
}
