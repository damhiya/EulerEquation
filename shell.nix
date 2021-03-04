let
  pkgs = import <nixpkgs> {};
  py = pkgs.python3.withPackages (ps: with ps; [ numpy matplotlib seaborn ]);
in
pkgs.mkShell {
  nativeBuildInputs = with pkgs; [ clang_11 cmake ];
  buildInputs = with pkgs; [ eigen py ];
}
