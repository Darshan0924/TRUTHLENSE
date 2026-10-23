# Synthetic dataset: definitions and assumptions

These CSV records were invented for this public portfolio exercise. They are not taken from a company, a recruitment exercise, or a production dataset.

| Column | Meaning | Note |
| --- | --- | --- |
| Site ID | Unique fictional site identifier | Do not duplicate |
| Site Name | Human-readable site/location name | Names can contain messy spacing |
| Survey Assigned Date | Date a site survey was allocated | ISO 8601 YYYY-MM-DD |
| Completion Date | Date work was completed | Blank if not completed |
| Work Status | Completed / In Progress / Scheduled | Normalize spelling |
| Client Amount | Expected amount to bill a client | Numeric, nonnegative |
| Invoice Number | Recorded invoice identifier | Blank means invoice missing |
| Client Payment Status | Operational billing/payment status | Does not independently prove funds received |
| Partner Amount | Cost associated with service partner | Numeric, nonnegative |
| Partner Payment Status | Partner's payment stage | E.g. Pending Payment or Paid |

## Defined metrics

**Accounts receivable (simplified):** sum Client Amount for completed sites with no Invoice Number. This is an operational proxy rather than a full accounting ledger balance.

**Missing-invoice payment-status flag:** sum Client Amount for completed sites where the invoice is blank but Client Payment Status says Invoice Raised or Payment Received. Review the mismatch rather than treating the number as confirmed receipts.

**Accounts payable (simplified):** sum Partner Amount for completed sites with Partner Payment Status equal to Pending Payment.

**Days since survey:** number of calendar days since assignment for a site without an invoice; otherwise display Not Required.

## Quality and safety notes

- Prefer Site ID rather than free-text Site Name for production joins.
- Accounts data deliberately includes leading/trailing whitespace and lowercase names.
- Real-life imported CSV may require numeric conversion and whitespace normalization.
- Do not share actual client records publicly or assume a payment-status field is bank-confirmed.
