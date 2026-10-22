#!/usr/bin/env python3
"""Install a locally signed simulator app only after checking its Keychain identity."""
import argparse
import json
import plistlib
import struct
import subprocess
from pathlib import Path


def simulated_entitlements(binary):
    if len(binary) < 32 or struct.unpack_from('<I', binary)[0] != 0xFEEDFACF:
        raise ValueError('Expected a thin 64-bit simulator executable')
    offset = 32
    count = struct.unpack_from('<I', binary, 16)[0]
    for _ in range(count):
        if offset + 8 > len(binary):
            raise ValueError('Truncated Mach-O commands')
        command, size = struct.unpack_from('<II', binary, offset)
        if size < 8 or offset + size > len(binary):
            raise ValueError('Invalid Mach-O command')
        if command == 0x19 and size >= 72:  # LC_SEGMENT_64
            sections = struct.unpack_from('<I', binary, offset + 64)[0]
            if 72 + sections * 80 > size:
                raise ValueError('Invalid section table')
            for index in range(sections):
                section = offset + 72 + index * 80
                name = binary[section:section + 16].rstrip(b'\0')
                segment = binary[section + 16:section + 32].rstrip(b'\0')
                if (segment, name) == (b'__TEXT', b'__entitlements'):
                    length = struct.unpack_from('<Q', binary, section + 40)[0]
                    start = struct.unpack_from('<I', binary, section + 48)[0]
                    if not length or start + length > len(binary):
                        raise ValueError('Invalid entitlement section')
                    return plistlib.loads(binary[start:start + length].rstrip(b'\0'))
        offset += size
    raise ValueError('Missing simulated Keychain identity. Rebuild with CODE_SIGNING_ALLOWED=YES CODE_SIGN_IDENTITY=-')


def inspect_bundle(bundle, team):
    info = plistlib.loads((bundle / 'Info.plist').read_bytes())
    bundle_id = info['CFBundleIdentifier']
    if info.get('DTPlatformName') != 'iphonesimulator' or bundle_id not in {'com.plonn.preview', 'com.oakas.plonn'}:
        raise ValueError('Expected a PLONN iOS simulator app')
    executable = info['CFBundleExecutable']
    if Path(executable).name != executable:
        raise ValueError('Invalid executable name')
    entitlements = simulated_entitlements((bundle / executable).read_bytes())
    if entitlements.get('application-identifier') != team + '.' + bundle_id:
        raise ValueError('Simulator application identity does not match this app and team')
    subprocess.run(['codesign', '--verify', '--deep', '--strict', str(bundle)], check=True, capture_output=True)
    return bundle_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--device', required=True)
    parser.add_argument('--team', default='R9G3K6TK2Y')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    bundle_id = inspect_bundle(args.bundle, args.team)
    subprocess.run(['xcrun', 'simctl', 'install', args.device, str(args.bundle)], check=True)
    if args.launch:
        subprocess.run(['xcrun', 'simctl', 'launch', args.device, bundle_id], check=True)
    print(json.dumps({'installed': True, 'bundle': bundle_id, 'keychain_identity_verified': True}))


if __name__ == '__main__':
    main()
