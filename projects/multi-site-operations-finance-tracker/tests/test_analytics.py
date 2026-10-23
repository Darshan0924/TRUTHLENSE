"""Tests use the synthetic sample only; no company records required."""

import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics import calculate, clean, read_sites


class OperationsFinanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_sites()

    def test_data_contains_sixteen_unique_sites(self):
        self.assertEqual(len(self.rows), 16)
        self.assertEqual(len({r["Site ID"] for r in self.rows}), 16)

    def test_expected_dashboard_totals(self):
        result = calculate(self.rows, date(2026, 10, 23))
        self.assertEqual(result["accounts_receivable"], Decimal("196550"))
        self.assertEqual(result["payment_status_flagged_missing_invoice"], Decimal("114000"))
        self.assertEqual(result["accounts_payable"], Decimal("121400"))

    def test_invoice_present_disables_aging(self):
        result = calculate(self.rows, date(2026, 10, 23))
        by_id = {r["Site ID"]: r for r in result["aging"]}
        self.assertEqual(by_id["OPS-002"]["Days Since Survey"], "Not Required")
        self.assertEqual(by_id["OPS-001"]["Days Since Survey"], 35)

    def test_totals_update_when_invoice_added(self):
        modified = [dict(r) for r in self.rows]
        original = calculate(modified, date(2026, 10, 23))
        modified[0]["Invoice Number"] = "INV-TEST"
        updated = calculate(modified, date(2026, 10, 23))
        self.assertEqual(original["accounts_receivable"] - updated["accounts_receivable"], Decimal("18500"))

    def test_payment_spelling_normalized(self):
        self.assertEqual(clean("  Payment   Received "), "payment received")


if __name__ == "__main__":
    unittest.main()
