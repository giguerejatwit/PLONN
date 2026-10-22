import importlib.util
import plistlib
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('release_check', Path(__file__).parents[1] / 'ops/check_plonn_release.py')
release_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_check)


class ReleaseBundleTests(unittest.TestCase):
    def bundle(self, root, **overrides):
        p = Path(root) / 'PLONN.app'; p.mkdir()
        info = dict(CFBundleExecutable='PLONN', CFBundleIdentifier='com.plonn.preview', PLONNPurchasesEnabled=False,
                    UIDeviceFamily=[1, 2], UISupportedInterfaceOrientations=['UIInterfaceOrientationPortrait'],
                    **{'UISupportedInterfaceOrientations~ipad': sorted(release_check.ORIENTATIONS)})
        info.update(overrides)
        (p / 'Info.plist').write_bytes(plistlib.dumps(info))
        (p / 'PrivacyInfo.xcprivacy').write_bytes(plistlib.dumps({}))
        (p / 'PLONN').write_bytes(b'Compiled application; sb_publishable_public_client_key')
        return p

    def testPublicClientKeyIsAllowedButBundledPrivateCredentialsFail(self):
        with tempfile.TemporaryDirectory() as root:
            p = self.bundle(root)
            self.assertTrue(release_check.inspect_bundle(p)['passed'])
            (p / 'server.env').write_text('KEY=sb_secret_abcdefghijklmno')
            report = release_check.inspect_bundle(p)
            self.assertFalse(report['passed'])
            self.assertFalse(report['checks']['private_key_patterns_absent'])
            self.assertNotIn('abcdefghijklmno', str(report))

    def testDevelopmentResourcesAndUnsafeTransportBlockPackagingCheck(self):
        with tempfile.TemporaryDirectory() as root:
            p = self.bundle(root, NSAppTransportSecurity={'NSAllowsArbitraryLoads': True})
            (p / 'history.json').write_text('[]')
            (p / 'PLONN').write_bytes('Local prototype · '.encode())
            report = release_check.inspect_bundle(p)
            self.assertFalse(report['passed'])
            for key in ['arbitrary_loads_disabled', 'development_ledger_excluded', 'development_ui_copy_excluded']:
                self.assertFalse(report['checks'][key])

    def testUnapprovedChatAndPublicPagesBlockPreparationPackaging(self):
        with tempfile.TemporaryDirectory() as root:
            p = self.bundle(root, PLONNChatEnabled=True, PLONNPublicPagesApproved=True)
            report = release_check.inspect_bundle(p)
            self.assertFalse(report['passed'])
            self.assertFalse(report['checks']['hosted_chat_configuration_authorized'])
            self.assertFalse(report['checks']['public_links_configuration_authorized'])

    def testUnclearedArchiveFailsEvenInsideANestedResourceBundle(self):
        with tempfile.TemporaryDirectory() as root:
            p = self.bundle(root)
            nested = p / 'OldReports.bundle'; nested.mkdir()
            (nested / 'FrozenMatchupReports-20261013.json').write_text('[]')
            report = release_check.inspect_bundle(p)
            self.assertFalse(report['passed'])
            self.assertFalse(report['checks']['uncleared_report_archives_excluded'])

    def testRenamingAnUnclearedPayloadDoesNotPassTheReleaseCheck(self):
        sources = Path(__file__).parents[1] / 'app/ios/PLONN'
        for name in release_check.UNCLEARED_REPORTS:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as root:
                p = self.bundle(root)
                (p / 'renamed.data').write_bytes((sources / name).read_bytes())
                self.assertFalse(release_check.inspect_bundle(p)['checks']['uncleared_report_archives_excluded'])

    def testEmbeddedHistoricalProofIsRejectedInAnAuxiliaryBinary(self):
        with tempfile.TemporaryDirectory() as root:
            p = self.bundle(root)
            (p / 'PLONN.debug.dylib').write_bytes(b'Compiled code public-023247816fbd8a0fd6d5 historical data')
            report = release_check.inspect_bundle(p)
            self.assertFalse(report['passed'])
            self.assertFalse(report['checks']['uncleared_inline_proofs_excluded'])

    def approved_services(self):
        return dict(schema_version=1, bundle_id='com.oakas.plonn', version='0.1.0', build='4',
                    chat_enabled=True, public_pages_approved=True, live_pages_verified=True,
                    privacy_url='https://oakas.ai/plonn/privacy', support_url='https://oakas.ai/plonn/support',
                    endpoint='https://api.oakas.ai/plonn/v1/chat', model='gemma4:26b',
                    digest='08ae7ec1744bd7f451c4a530afb39d2673ad9d07a8369b8a33a3613b41212a68',
                    publication_commit='0c3db926719fe868f97ec1fcadbe7917ab8bd10b')

    def testExactApprovedServiceArtifactPassesWhileDefaultStillFails(self):
        with tempfile.TemporaryDirectory() as root:
            p = self.bundle(root, CFBundleIdentifier='com.oakas.plonn', CFBundleShortVersionString='0.1.0',
                            CFBundleVersion='4', PLONNChatEnabled=True, PLONNPublicPagesApproved=True)
            self.assertTrue(release_check.inspect_bundle(p, self.approved_services())['passed'])
            self.assertFalse(release_check.inspect_bundle(p)['passed'])

    def testApprovalCannotEnableWrongArtifactEndpointModelOrUnpublishedPages(self):
        for key, value in [('build','3'), ('endpoint','https://other.example/chat'), ('model','gemma3'),
                           ('live_pages_verified',False), ('publication_commit',''), ('public_pages_approved',False)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                p = self.bundle(root, CFBundleIdentifier='com.oakas.plonn', CFBundleShortVersionString='0.1.0',
                                CFBundleVersion='4', PLONNChatEnabled=True, PLONNPublicPagesApproved=True)
                approval = self.approved_services(); approval[key] = value
                self.assertFalse(release_check.inspect_bundle(p, approval)['passed'])

    def testMalformedFlagsOrUnconfiguredSocialProvidersFail(self):
        for overrides in [dict(PLONNChatEnabled='false'), dict(PLONNPublicPagesApproved=1),
                          dict(PLONNAppleAuthEnabled=True), dict(PLONNGoogleAuthReady=True)]:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as root:
                self.assertFalse(release_check.inspect_bundle(self.bundle(root, **overrides))['passed'])

    def testCloudApprovalRequiresMatchingBinaryProviderAndFreshConsent(self):
        approval = self.approved_services()
        approval.update(schema_version=2, endpoint='https://api.oakas.ai/plonn/v1/chat/cloud',
                        model='gemma4:31b', provider='ollama_cloud', consent_version='v2_ollama_cloud')
        approval.pop('digest')
        with tempfile.TemporaryDirectory() as root:
            p = self.bundle(root, CFBundleIdentifier='com.oakas.plonn', CFBundleShortVersionString='0.1.0',
                            CFBundleVersion='4', PLONNChatEnabled=True, PLONNPublicPagesApproved=True)
            self.assertFalse(release_check.inspect_bundle(p, approval)['passed'])
            (p / 'PLONN').write_bytes(b'https://api.oakas.ai/plonn/v1/chat/cloud gemma4:31b v2_ollama_cloud')
            self.assertTrue(release_check.inspect_bundle(p, approval)['passed'])
            for key, value in [('provider', 'other'), ('consent_version', 'v1'),
                               ('digest', 'pretend-cloud-model-digest'), ('model', 'gemma4:26b')]:
                with self.subTest(key=key):
                    self.assertFalse(release_check.inspect_bundle(p, dict(approval, **{key: value}))['passed'])

    def testMissingOrIncompleteIPadOrientationsFailPackaging(self):
        for phone, ipad in [([], []), (['UIInterfaceOrientationPortrait'], ['UIInterfaceOrientationPortrait']),
                            (['UIInterfaceOrientationPortrait'], ['UnsupportedOrientation'])]:
            with self.subTest(phone=phone, ipad=ipad), tempfile.TemporaryDirectory() as root:
                p = self.bundle(root, UISupportedInterfaceOrientations=phone,
                                **{'UISupportedInterfaceOrientations~ipad': ipad})
                report = release_check.inspect_bundle(p)
                self.assertFalse(report['passed'])
                self.assertFalse(report['checks']['interface_orientations_configured'])

    def testIPadCanUseFullBaseOrientationsAndPhoneCanUsePortraitOnly(self):
        for family, phone in [([1, 2], sorted(release_check.ORIENTATIONS)), ([1], ['UIInterfaceOrientationPortrait'])]:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as root:
                p = self.bundle(root, UIDeviceFamily=family, UISupportedInterfaceOrientations=phone)
                info = plistlib.loads((p / 'Info.plist').read_bytes())
                info.pop('UISupportedInterfaceOrientations~ipad')
                (p / 'Info.plist').write_bytes(plistlib.dumps(info))
                self.assertTrue(release_check.inspect_bundle(p)['passed'])


if __name__ == '__main__':
    unittest.main()
