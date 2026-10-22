#!/usr/bin/env python3
"""Inspect a locally built Release .app. Reads files only; never signs or uploads."""
import argparse
import hashlib
import json
import plistlib
import re
from pathlib import Path


ORIENTATIONS = {
    'UIInterfaceOrientationPortrait', 'UIInterfaceOrientationPortraitUpsideDown',
    'UIInterfaceOrientationLandscapeLeft', 'UIInterfaceOrientationLandscapeRight',
}

# Historical reports remain as development evidence until their inputs are licensed.
# Detect both the original filenames and identical payloads renamed inside a bundle.
UNCLEARED_REPORTS = {
    'FrozenMatchupReports-20261013.json': 'be7c932a32c4f8658b2933c777f740c7a4801370523a849e41c827d09057d321',
    'MatchupNarrativeReports-v4.json': '3db2a6d6752f3dc1414ea692fd2302c1ab9caad9f52bcb81dc35d10198bf79f8',
    'MatchupNarrativeReports-v5.json': '52c38edaaed4b44727385cff61f15dfc77ffb24f9994fe0075f2ea7ed14baf05',
}
UNCLEARED_PROOF_MARKERS = (
    b'public-023247816fbd8a0fd6d5', b'public-0013fe0dfb7806b1e3c0',
)


def orientations_configured(info):
    phone = info.get('UISupportedInterfaceOrientations')
    if not isinstance(phone, list) or not phone or any(not isinstance(value, str) or value not in ORIENTATIONS for value in phone):
        return False
    if 2 in info.get('UIDeviceFamily', []):
        ipad = info.get('UISupportedInterfaceOrientations~ipad', phone)
        if not isinstance(ipad, list) or any(not isinstance(value, str) or value not in ORIENTATIONS for value in ipad):
            return False
        return set(ipad) == ORIENTATIONS
    return True


def inspect_bundle(bundle, approved_services=None):
    bundle = Path(bundle)
    info = plistlib.loads((bundle / 'Info.plist').read_bytes())
    executable = bundle / info['CFBundleExecutable']
    binary = executable.read_bytes()
    diagnostic_copy = [
        'Local prototype · ',
        'Start the local odds server',
        'Imported local ledger, unaudited.',
        'Retained weights · Frozen historical feature snapshot',
        'Verified local summary',
        ' · Run ',
        'private model migration',
    ]
    private_keys = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|sb_secret_[A-Za-z0-9_-]{12,}')
    approval = approved_services or {}
    approved_chat = approval.get('chat_enabled') is True
    approved_pages = approval.get('public_pages_approved') is True
    legacy_service = (
        approval.get('schema_version') == 1 and
        approval.get('endpoint') == 'https://api.oakas.ai/plonn/v1/chat' and
        approval.get('model') == 'gemma4:26b' and
        approval.get('digest') == '08ae7ec1744bd7f451c4a530afb39d2673ad9d07a8369b8a33a3613b41212a68'
    )
    cloud_service = (
        approval.get('schema_version') == 2 and
        approval.get('endpoint') == 'https://api.oakas.ai/plonn/v1/chat/cloud' and
        approval.get('model') == 'gemma4:31b' and
        approval.get('provider') == 'ollama_cloud' and
        approval.get('consent_version') == 'v2_ollama_cloud' and
        'digest' not in approval and
        # Swift can inline short strings (including the model name) in machine
        # instructions. Only require contiguous bytes for the longer values.
        all(value.encode() in binary for value in [
            'https://api.oakas.ai/plonn/v1/chat/cloud', 'v2_ollama_cloud'])
    )
    approval_matches = not approval or (
        (legacy_service or cloud_service) and
        approval.get('bundle_id') == info.get('CFBundleIdentifier') and
        approval.get('version') == info.get('CFBundleShortVersionString') and
        approval.get('build') == info.get('CFBundleVersion') and
        approval.get('privacy_url') == 'https://oakas.ai/plonn/privacy' and
        approval.get('support_url') == 'https://oakas.ai/plonn/support' and
        approval.get('live_pages_verified') is True and
        bool(re.fullmatch(r'[a-f0-9]{40}', approval.get('publication_commit', ''))) and
        approved_chat and approved_pages
    )
    def flag_matches(key, expected):
        value = info.get(key, False)
        return type(value) is bool and value is expected
    private_key_found = report_found = proof_found = False
    for path in bundle.rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        payload = path.read_bytes()
        private_key_found |= bool(private_keys.search(payload))
        report_found |= path.name in UNCLEARED_REPORTS or hashlib.sha256(payload).hexdigest() in UNCLEARED_REPORTS.values()
        proof_found |= any(marker in payload for marker in UNCLEARED_PROOF_MARKERS)
    findings = {
        'privacy_manifest_present': (bundle / 'PrivacyInfo.xcprivacy').is_file(),
        'arbitrary_loads_disabled': not info.get('NSAppTransportSecurity', {}).get('NSAllowsArbitraryLoads', False),
        'interface_orientations_configured': orientations_configured(info),
        'development_ledger_excluded': not (bundle / 'history.json').exists(),
        'uncleared_report_archives_excluded': not report_found,
        'uncleared_inline_proofs_excluded': not proof_found,
        'development_ui_copy_excluded': not any(s.encode() in binary for s in diagnostic_copy),
        'purchases_disabled': not info.get('PLONNPurchasesEnabled', False),
        'service_approval_matches_artifact': approval_matches,
        'hosted_chat_configuration_authorized': flag_matches('PLONNChatEnabled', approved_chat),
        'public_links_configuration_authorized': flag_matches('PLONNPublicPagesApproved', approved_pages),
        'chat_requires_approved_pages': not info.get('PLONNChatEnabled', False) or info.get('PLONNPublicPagesApproved') is True,
        'unconfigured_social_providers_disabled': all(flag_matches(key, False) for key in [
            'PLONNAppleAuthEnabled', 'PLONNGoogleAuthEnabled', 'PLONNAppleAuthReady',
            'PLONNGoogleAuthReady', 'PLONNIdentityLinkingReady']),
        'private_key_patterns_absent': not private_key_found,
    }
    return {
        'bundle': str(bundle.resolve()),
        'bundle_id': info.get('CFBundleIdentifier'),
        'version': info.get('CFBundleShortVersionString'),
        'build': info.get('CFBundleVersion'),
        'checks': findings,
        'enabled_features': {'hosted_chat': info.get('PLONNChatEnabled') is True,
                             'public_pages': info.get('PLONNPublicPagesApproved') is True},
        'passed': all(findings.values()),
        'scope': 'Local packaging checks only; does not verify Apple membership, signing, privacy metadata, live services or data rights.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--approved-services', type=Path,
                        help='Explicit reviewed service configuration for this exact artifact; defaults require disabled services.')
    args = parser.parse_args()
    approved = json.loads(args.approved_services.read_text()) if args.approved_services else None
    report = inspect_bundle(args.bundle, approved)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
