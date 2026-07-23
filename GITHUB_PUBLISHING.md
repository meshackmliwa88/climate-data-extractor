# Publish to GitHub

Review the project before publishing:

```bash
git status
git grep -nEi 'password|passwd|secret|api[_-]?key|token|PRIVATE KEY'
find . -type f -size +90M -print
```

Create an empty public GitHub repository, then run:

```bash
git init
git branch -M main
git add .
git status
git commit -m "Initial public release of Climate Data Extractor"
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Do not commit `.env`, `storage/`, databases, Zarr/NetCDF files, generated outputs, private keys or user records.
