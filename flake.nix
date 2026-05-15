{
  description = "rinhabackend2026";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };

    devshell = {
      url = "github:numtide/devshell";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      imports = [
        inputs.devshell.flakeModule
        inputs.treefmt-nix.flakeModule
      ];

      perSystem = { system, lib, ... }:
        let
          pkgs = import inputs.nixpkgs { inherit system; };
        in
        {
          _module.args.pkgs = pkgs;

          treefmt = {
            projectRootFile = "flake.nix";
            programs = {
              clang-format.enable = true;
              nixpkgs-fmt.enable = true;
              buildifier.enable = true;
            };
          };

          devshells.default = {
            name = "rinhabackend2026";

            packages = with pkgs; [
              gcc14

              bazel_7
              bazel-buildtools

              pkg-config
              zlib

              # profiling
              perf
              flamegraph
              hotspot

              # bench + debug
              hyperfine
              wrk
              k6
              gdb

              # infra
              docker-compose
              git
              curl
              jq
            ] ++ lib.optionals pkgs.stdenv.isDarwin [
              pkgs.libiconv
            ];

            env = [
              { name = "CC"; value = "${pkgs.gcc14}/bin/gcc"; }
              { name = "CXX"; value = "${pkgs.gcc14}/bin/g++"; }
            ];

            commands = [
              {
                name = "build";
                command = "bazel build //src:rinha";
                help = "Build the rinha binary";
                category = "bazel";
              }
              {
                name = "build-profile";
                command = "bazel build //src:rinha //bench:bench_http --config=profile";
                help = "Build with profiling symbols (-O2 -g -fno-omit-frame-pointer)";
                category = "bazel";
              }
              {
                name = "build-release";
                command = "bazel build //src:rinha --config=release";
                help = "Build release (-O3 -march=haswell -flto)";
                category = "bazel";
              }
              {
                name = "bench";
                command = "bazel run //bench:smoke";
                help = "Smoke test: compose up + k6 smoke (5 reqs)";
                category = "bench";
              }
              {
                name = "bench-full";
                command = "bazel run //bench:full";
                help = "Full rinha test: compose up + k6 ramp 120s (needs bench/test-data.json)";
                category = "bench";
              }
              {
                name = "test";
                command = "bazel test //...";
                help = "Run all tests";
                category = "bazel";
              }
              {
                name = "fmt";
                command = "nix fmt";
                help = "Format all files";
                category = "dev";
              }
              {
                name = "clean";
                command = "bazel clean --expunge";
                help = "Clean bazel cache";
                category = "bazel";
              }
            ];
          };
        };
    };
}
