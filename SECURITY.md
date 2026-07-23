# Security

Do not commit `.env`, SQLite databases, climate datasets, generated exports, private keys, access tokens, or user records.

Before publishing changes, run:

```bash
git grep -nEi 'password|passwd|secret|api[_-]?key|token|PRIVATE KEY'
```

Configure `CDE_SECRET_KEY` and the initial administrator credentials through environment variables. Change the initial administrator password immediately after first login.
