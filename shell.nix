with import <nixpkgs> { };
let py = pkgs.python3.withPackages (ps: with ps; [ numpy matplotlib seaborn ]);
in llvm_11.stdenv.mkDerivation {
  name = "EulerEquation";
  nativeBuildInputs = with pkgs; [ clang_11 cmake ];
  buildInputs = with pkgs; [ eigen py ];
}
