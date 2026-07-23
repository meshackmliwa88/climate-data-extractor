# Public Release Notes

This package was prepared for publication with the following changes:

- Removed all institutional logo and coat-of-arms image assets and references.
- Removed logo embedding from QR codes, Excel outputs and PDF outputs.
- Removed the following sidebar modules and their public routes/templates:
  - Export Logs
  - Data Cost Recovery
  - Customers
  - Proposed Cost Recovery
  - Station Registration
- Removed station-registration dependencies from user management.
- Removed customer selection from the public extraction workflow.
- Added `.gitignore`, `.env.example`, `SECURITY.md` and public setup instructions.
- Replaced fixed example deployment credentials with environment-based or randomly generated values.
- Removed Python caches, runtime databases, datasets, logs and generated outputs.

Large climate datasets and runtime SQLite data are intentionally not included.
