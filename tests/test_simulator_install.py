import importlib.util
import plistlib
import struct
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('sim_install', Path(__file__).parents[1] / 'ops/install_plonn_simulator.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SimulatorIdentityTests(unittest.TestCase):
    def test_unsigned_build_is_rejected(self):
        binary = bytearray(32)
        struct.pack_into('<I', binary, 0, 0xFEEDFACF)
        with self.assertRaisesRegex(ValueError, 'Missing simulated Keychain identity'):
            module.simulated_entitlements(binary)

    def test_reads_linked_simulator_identity(self):
        identity = {'application-identifier': 'TEAM.com.plonn.preview'}
        payload = plistlib.dumps(identity)
        binary = bytearray(32 + 152 + len(payload))
        struct.pack_into('<I', binary, 0, 0xFEEDFACF)
        struct.pack_into('<I', binary, 16, 1)
        struct.pack_into('<II', binary, 32, 0x19, 152)
        struct.pack_into('<I', binary, 96, 1)
        binary[104:120] = b'__entitlements'.ljust(16, b'\0')
        binary[120:136] = b'__TEXT'.ljust(16, b'\0')
        struct.pack_into('<Q', binary, 144, len(payload))
        struct.pack_into('<I', binary, 152, 184)
        binary[184:] = payload
        self.assertEqual(module.simulated_entitlements(binary), identity)

    def test_malformed_binary_is_rejected(self):
        for binary in [b'', b'not a Mach-O', struct.pack('<I', 0xFEEDFACF) + bytes(28) + b'junk']:
            with self.assertRaises(ValueError):
                module.simulated_entitlements(binary)


if __name__ == '__main__':
    unittest.main()
