# Google Sheets setup and formulas

This guide uses the **fictional datasets in this repository**. It is not a copy of any employer's assessment.

## Import
1. Upload `data/sample_sites.csv` as a tab called **Sites**.
2. Upload `data/sample_accounts.csv` as a tab called **Accounts**.
3. Convert survey and completion date fields to real Google Sheets dates (Format > Number > Date).
4. Keep monetary amounts numeric.

**Sites columns:** A Site ID; B Site Name; C Survey Assigned Date; D Completion Date; E Work Status; F Client Amount; G Invoice Number; H Client Payment Status; I Partner Amount; J Partner Payment Status.

**Accounts columns:** A Site Name; B Amount.

## 1. Dynamic report
In a new **Report** tab, cell A1:

```excel
=QUERY(Sites!A:J,"select B,E,D,F,I where B is not null",1)
```

This shows Site Name, Work Status, Completion Date, Client Amount, and Partner Amount. Source changes appear automatically.

## 2. Accounts lookup
Put `Amount from Accounts` in Report!F1. In Report!F2:

```excel
=IF(A2="","",IFNA(VLOOKUP(TRIM(A2),{ARRAYFORMULA(TRIM(Accounts!$A$2:$A$1000)),Accounts!$B$2:$B$1000},2,FALSE),"Not Found"))
```

Copy down the rows you need. `TRIM` fixes ordinary extra whitespace in both lists; `2` selects the Amount; `FALSE` enforces exact matching; `$` prevents shifting references. Google Sheets VLOOKUP is normally case-insensitive. Use unique site IDs for more robust production matching.

## 3. Dashboard formulas

**Accounts receivable** — completed sites whose invoice number is blank:

```excel
=SUMIFS(Sites!F2:F,Sites!E2:E,"Completed",Sites!G2:G,"")
```

**Missing-invoice status anomaly** — completed site, no invoice number, but status says Invoice Raised or Payment Received:

```excel
=SUMIFS(Sites!F2:F,Sites!E2:E,"Completed",Sites!G2:G,"",Sites!H2:H,"Invoice Raised")+SUMIFS(Sites!F2:F,Sites!E2:E,"Completed",Sites!G2:G,"",Sites!H2:H,"Payment Received")
```

This is a record-quality/status flag, **not confirmed cash received**.

**Accounts payable** — completed work with partner payment pending:

```excel
=SUMIFS(Sites!I2:I,Sites!E2:E,"Completed",Sites!J2:J,"Pending Payment")
```

For the supplied fictional CSV, results are **₹196,550**, **₹114,000** and **₹121,400** respectively.

## 4. Days since survey assignment

In `Sites!K1` enter `Days Since Survey`. Ensure all cells below K1 are empty. In K2:

```excel
=ARRAYFORMULA(IF(B2:B="","",IF(LEN(TRIM(G2:G&""))>0,"Not Required",IF(C2:C="","",TODAY()-C2:C))))
```

Sites with invoice numbers display `Not Required`; uninvoiced sites show their age as of today. Survey dates must be stored as actual dates.

## 5. Optional stakeholder view

Create another Google spreadsheet and in A1 enter:

```excel
=IMPORTRANGE("PASTE_SOURCE_SPREADSHEET_URL","Report!A:E")
```

Approve access when prompted. Updates may be delayed slightly. For real financial data, only share with authorized stakeholders.

## Validation checklist
- Change a site status and verify the Report updates; restore it.
- Change an Accounts amount and verify the lookup updates; restore it.
- Lookup an unknown site and expect `Not Found`.
- Add a temporary invoice number and confirm receivables fall and survey aging becomes `Not Required`.
- Cross-check the totals using `python analytics.py`.

## Known limitations
- A copied-down lookup must be extended as new Report rows appear.
- The Accounts range currently stops at row 1000.
- `TRIM` may not clean nonbreaking spaces.
- Actual payments require bank reconciliation; a payment-status label is not proof of receipt.
