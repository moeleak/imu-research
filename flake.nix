{
  description = "Typst environment.";

  inputs = {
    nixpkgs.url = "github:cherrypiejam/nixpkgs?ref=typst";
    flake-utils.url = "github:numtide/flake-utils";
    self.submodules = true;
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        typstWithPkgs = pkgs.typst.withPackages (
          ps: with ps; [
            touying
            cetz
            fletcher
            numbly
            theorion
            fletcher
          ]
        );

        tools = [
          typstWithPkgs
          pkgs.ghostscript
        ];

        fonts = with pkgs; [
          noto-fonts
          noto-fonts-cjk-sans
          noto-fonts-cjk-serif
          source-han-sans
          source-han-mono
          source-han-serif
          dejavu_fonts
          liberation_ttf
          libertinus
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = tools;
          env.TYPST_FONT_PATHS = pkgs.lib.strings.concatStringsSep ":" fonts;
          shellHook = ''
            unset SOURCE_DATE_EPOCH
          '';
        };

        packages = {
          default = pkgs.stdenv.mkDerivation {
            pname = "typst-pdfs";
            version = "1.0.0";
            src = ./.;
            nativeBuildInputs = tools;
            env.TYPST_FONT_PATHS = pkgs.lib.strings.concatStringsSep ":" fonts;
            buildPhase = ''
              unset SOURCE_DATE_EPOCH
              typst compile main.typ main.pdf
              gs -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -dNOPAUSE -dQUIET -dBATCH -sOutputFile=compressed.pdf main.pdf
            '';
            installPhase = ''
              mkdir -p $out
              cp -v main.pdf compressed.pdf $out/
            '';
          };

          inherit (pkgs) buildTypstPackage typstPackages;
        };
      }
    );
}
