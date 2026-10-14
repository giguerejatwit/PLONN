#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
package="$(mktemp -d /tmp/plonn-referral-tests.XXXXXX)"
trap 'rm -rf "$package"' EXIT
mkdir -p "$package/Sources/ReferralCore" "$package/Tests/ReferralCoreTests"
cp "$root/PLONN/Referral.swift" "$package/Sources/ReferralCore/"
cp "$root/Tests/ReferralStateTests.swift" "$package/Tests/ReferralCoreTests/"
cat > "$package/Package.swift" <<'SWIFT'
// swift-tools-version: 5.9
import PackageDescription
let package = Package(name: "ReferralCore", targets: [
    .target(name: "ReferralCore"),
    .testTarget(name: "ReferralCoreTests", dependencies: ["ReferralCore"], swiftSettings: [.define("REFERRAL_STANDALONE_TESTS")])
])
SWIFT
swift test --package-path "$package" --scratch-path /tmp/plonn-referral-swift-build
