#!/usr/bin/env bash
# Build Nova's portable satellite AEC (WebRTC AEC3 behind include/nova_aec.h)
# and run its ERLE self-test.
#
# Deliberately assumes no package manager. webrtc-audio-processing is a
# meson/ninja C++ build and its abseil dependency arrives as a meson subproject,
# so the only host requirements are a C/C++ toolchain, python3, curl and tar --
# which is what makes this reproducible on a bare macOS box (Indium has no
# Homebrew) as well as on Linux. meson/ninja go into a local venv so nothing is
# added to the host environment.
#
#   ./build.sh              build the static lib and run the self-test
#   ./build.sh selftest     re-run the self-test only
#
# Outputs (git-ignored, under build/):
#   build/libnova_aec.a          our ABI + AEC3, ready for a binding to link
#   build/nova_aec_selftest      the proof obligation
set -euo pipefail

WAP_VERSION="1.3"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="${here}/build"
vendor="${build}/vendor"
wap_dir="${vendor}/webrtc-audio-processing-v${WAP_VERSION}"
wap_lib="${wap_dir}/build/webrtc/modules/audio_processing/libwebrtc-audio-processing-1.a"
venv="${build}/toolchain"

log() { printf '>>> %s\n' "$*"; }

ensure_toolchain() {
  if [[ -x "${venv}/bin/meson" && -x "${venv}/bin/ninja" ]]; then
    return
  fi
  log "bootstrapping meson + ninja into ${venv}"
  python3 -m venv "${venv}"
  "${venv}/bin/pip" -q install --upgrade pip
  "${venv}/bin/pip" -q install meson ninja
}

ensure_webrtc() {
  if [[ -f "${wap_lib}" ]]; then
    return
  fi
  mkdir -p "${vendor}"
  if [[ ! -d "${wap_dir}" ]]; then
    log "fetching webrtc-audio-processing v${WAP_VERSION}"
    curl -sSfL -o "${vendor}/wap.tar.gz" \
      "https://gitlab.freedesktop.org/pulseaudio/webrtc-audio-processing/-/archive/v${WAP_VERSION}/webrtc-audio-processing-v${WAP_VERSION}.tar.gz"
    tar xzf "${vendor}/wap.tar.gz" -C "${vendor}"
  fi
  log "building webrtc-audio-processing (AEC3) -- first run takes a few minutes"
  PATH="${venv}/bin:${PATH}" meson setup "${wap_dir}/build" "${wap_dir}" \
    --default-library=static --buildtype=release >/dev/null
  # ranlib warns about abseil translation units that are empty on this platform.
  # They are harmless and would otherwise bury real errors.
  PATH="${venv}/bin:${PATH}" ninja -C "${wap_dir}/build" 2>&1 \
    | grep -v "has no symbols" || true
  if [[ ! -f "${wap_lib}" ]]; then
    echo "webrtc-audio-processing build produced no library" >&2
    exit 1
  fi
}

abseil_dir() {
  find "${wap_dir}/subprojects" -maxdepth 1 -type d -name 'abseil-cpp-*' | head -1
}

build_lib() {
  log "compiling nova_aec and linking AEC3 into libnova_aec.a"
  mkdir -p "${build}/obj"
  c++ -std=c++17 -O2 -fPIC -c "${here}/src/nova_aec.cc" \
    -o "${build}/obj/nova_aec.o" \
    -I"${here}/include" \
    -I"${wap_dir}" \
    -I"${wap_dir}/webrtc" \
    -I"$(abseil_dir)"

  # Merge into one archive so a binding links a single artefact instead of
  # reproducing webrtc's internal library split. Object files are unpacked into
  # per-archive directories because names collide across them.
  rm -rf "${build}/obj/merge"
  mkdir -p "${build}/obj/merge"
  while read -r archive; do
    tag="$(basename "${archive}" .a)"
    mkdir -p "${build}/obj/merge/${tag}"
    ( cd "${build}/obj/merge/${tag}" && ar x "${archive}" )
  done < <(find "${wap_dir}/build" -name '*.a' -print)

  rm -f "${build}/libnova_aec.a"
  find "${build}/obj/merge" -name '*.o' -print0 \
    | xargs -0 ar crs "${build}/libnova_aec.a" "${build}/obj/nova_aec.o"
  log "libnova_aec.a: $(du -h "${build}/libnova_aec.a" | cut -f1)"
}

build_selftest() {
  log "building the ERLE self-test"
  # AEC3 is C++ and, on Apple platforms, pulls in Foundation for its
  # system_wrappers; the Linux link line is the fallback.
  if [[ "$(uname -s)" == "Darwin" ]]; then
    cc -std=c11 -O2 -I"${here}/include" \
      "${here}/selftest/nova_aec_selftest.c" "${build}/libnova_aec.a" \
      -lc++ -framework Foundation -framework CoreFoundation \
      -o "${build}/nova_aec_selftest"
  else
    cc -std=c11 -O2 -I"${here}/include" \
      "${here}/selftest/nova_aec_selftest.c" "${build}/libnova_aec.a" \
      -lstdc++ -lm -lpthread \
      -o "${build}/nova_aec_selftest"
  fi
}

case "${1:-all}" in
  selftest)
    "${build}/nova_aec_selftest"
    ;;
  all)
    ensure_toolchain
    ensure_webrtc
    build_lib
    build_selftest
    log "running the self-test"
    "${build}/nova_aec_selftest"
    ;;
  *)
    echo "usage: $0 [all|selftest]" >&2
    exit 2
    ;;
esac
