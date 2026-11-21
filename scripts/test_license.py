"""Every license statement says SSPL v1.0 or a commercial license from Thomas Cherickal: python3 scripts/test_license.py"""
import glob
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPR = "SSPL-1.0 OR LicenseRef-Commercial"


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


class License(unittest.TestCase):
    def test_every_manifest_declares_the_dual_license(self):
        manifests = [m for m in glob.glob(f"{ROOT}/**/Cargo.toml", recursive=True) if "/target/" not in os.path.relpath(m, ROOT)]
        self.assertGreaterEqual(len(manifests), 2)
        for m in manifests:
            text = open(m, encoding="utf-8").read()
            found = re.findall(r'^license\s*=\s*"([^"]+)"', text, re.M)
            inherited = re.search(r"^license\.workspace\s*=\s*true", text, re.M)
            self.assertTrue(found == [EXPR] or (inherited and not found), f"{m}: {found}")

    def test_the_license_files_and_documents_agree(self):
        lic = read("LICENSE")
        self.assertIn("SSPL", lic)
        self.assertIn("Commercial License", lic)
        self.assertIn("Thomas Cherickal", lic)
        self.assertIn("Server Side Public License", read("LICENSE-SSPL.txt"))
        self.assertIn(EXPR, read("readme.md"))
        agents = read("AGENTS.md")
        self.assertIn("**SSPL v1.0**", agents)
        statements = {
            "readme.md": read("readme.md").split("## License")[-1],
            "AGENTS.md": agents.split("## 5. Licensing")[-1].split("## 6.")[0],
        }
        for name, text in statements.items():
            self.assertNotRegex(text, r"AGPL|\bMIT\b", name)
        self.assertNotIn("AGPL", lic)


if __name__ == "__main__":
    unittest.main()
