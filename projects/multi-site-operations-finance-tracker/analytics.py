"""Analyze a synthetic multi-site facilities dataset using only Python's standard library."""

from __future__ import annotations

import csv
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE_DATA = ROOT / "data" / "sample_sites.csv"


def clean(value: str | None) -> str:
    """Normalize regular whitespace and capitalization for comparisons."""
    return " ".join((value or "").split()).casefold()


def money(value: str, column: str, site_id: str) -> Decimal:
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid {column} for {site_id}: {value!r}") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"Invalid nonnegative {column} for {site_id}: {value!r}")
    return amount


def read_sites(path: Path = SAMPLE_DATA) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("The site file has no records")
    required = {
        "Site ID", "Site Name", "Survey Assigned Date", "Completion Date",
        "Work Status", "Client Amount", "Invoice Number",
        "Client Payment Status", "Partner Amount", "Partner Payment Status",
    }
    if required - rows[0].keys():
        raise ValueError(f"Missing columns: {sorted(required - rows[0].keys())}")
    ids = [row["Site ID"].strip() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Site IDs must be unique")
    return rows


def calculate(rows: list[dict[str, str]], as_of: date | None = None) -> dict:
    """Calculate the exact dashboard conditions, plus invoice-aware site aging."""
    as_of = as_of or date.today()
    receivable = Decimal("0")
    status_flagged = Decimal("0")
    payable = Decimal("0")
    aging = []

    for row in rows:
        site_id = row["Site ID"].strip()
        client = money(row["Client Amount"], "Client Amount", site_id)
        partner = money(row["Partner Amount"], "Partner Amount", site_id)
        completed = clean(row["Work Status"]) == "completed"
        missing_invoice = not (row["Invoice Number"] or "").strip()
        payment = clean(row["Client Payment Status"])

        if completed and missing_invoice:
            receivable += client
            if payment in {"invoice raised", "payment received"}:
                status_flagged += client

        if completed and clean(row["Partner Payment Status"]) == "pending payment":
            payable += partner

        if not missing_invoice:
            days = "Not Required"
        elif not row["Survey Assigned Date"].strip():
            days = ""
        else:
            assigned = date.fromisoformat(row["Survey Assigned Date"].strip())
            days = (as_of - assigned).days
        aging.append({"Site ID": site_id, "Site Name": row["Site Name"].strip(), "Days Since Survey": days})

    return {
        "as_of": as_of,
        "total_sites": len(rows),
        "accounts_receivable": receivable,
        "payment_status_flagged_missing_invoice": status_flagged,
        "accounts_payable": payable,
        "aging": aging,
    }


def main() -> None:
    summary = calculate(read_sites())
    print("MULTI-SITE OPERATIONS & FINANCE TRACKER")
    print("Synthetic demonstration data — not real company records")
    print(f"As of: {summary['as_of']} | Sites: {summary['total_sites']}")
    print(f"Accounts receivable (completed, invoice missing): ₹{summary['accounts_receivable']:,.2f}")
    print(f"Invoice-missing rows flagged by payment status: ₹{summary['payment_status_flagged_missing_invoice']:,.2f}")
    print(f"Accounts payable (completed, partner pending): ₹{summary['accounts_payable']:,.2f}")
    print("\nSITE AGING")
    for site in summary["aging"]:
        print(f"  {site['Site ID']} | {site['Site Name']} | {site['Days Since Survey']}")


if __name__ == "__main__":
    main()
