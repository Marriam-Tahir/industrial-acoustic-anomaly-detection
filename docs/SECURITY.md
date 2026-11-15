# Smart Listener - security

This is a single-site prototype that runs on one computer. This page lists what is protected, how it is checked,
and what would still be needed for a real deployment.

## In place

| Area | What is done | Checked by |
|---|---|---|
| Passwords | Stored only as PBKDF2-SHA256 hashes (600,000 rounds, random salt per password). At least 8 characters, and not the same as the username. | `tests/test_phase8_security.py` (no password in the database dump) |
| Sessions | Random 256-bit tokens; only their SHA-256 hash is stored. Sessions expire after 12 hours (`SL_SESSION_HOURS`). Changing a password signs out the account's other sessions. | `test_phase8_security.py`, `test_redesign_backend.py` |
| Access control | Every API route except `GET /health` and `POST /auth/login` needs a valid session. | A test walks **every** route without a token and with a fake token and expects 401/403 |
| Login attempts | After 5 failed logins for a user from one address within 5 minutes, logins are refused (HTTP 429) until the window passes, even with the right password. | `test_login_lockout_after_repeated_failures` |
| Network exposure | The API listens on 127.0.0.1 only, so other computers cannot reach it. | `scripts/run_api.py` default; `scripts/check_system.py` |
| Response headers | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, and `Cache-Control: no-store` on data responses. | `test_security_headers_on_responses` |
| Database queries | All SQL uses parameters, never string building. | A machine named `x'); DROP TABLE machines;--` is stored as plain text |
| Web page content | Names, notes and other text from the database are HTML-escaped before display. | `test_names_from_the_database_cannot_inject_html` |
| Uploads | The dashboard accepts .wav files; the API accepts only files that decode as audio. 60 MB and 600 s at most, silence and non-audio refused. Files go to a temporary file that is deleted after scoring; uploaded audio is never kept. | `tests/test_phase5_api.py` |
| E-mail password | Read only from `config/smtp.env` or environment variables. Never stored in the database, returned by the API or logged. `config/smtp.env` is in `.gitignore`. The unedited example file is refused. | `test_phase7_mail.py`, `test_unedited_example_mail_file_is_refused`; `check_system.py` checks `.gitignore` |
| Tests and e-mail | The test suite always runs in test mode, so it can never send real e-mail. | `test_config_file_and_environment` |
| Audit trail | Sign-ins, machine changes, acknowledgements, notes, reports, e-mail actions and settings changes are recorded with user and time. | Settings > Audit trail |

## Not in place (would be needed before real deployment)

- **HTTPS.** The API and dashboard use plain HTTP on the local computer. Serving them to other computers would need
  HTTPS (for example a reverse proxy) and a firewall.
- **User roles.** Every account is an administrator; there are no read-only operator accounts.
- **Self-service password reset.** An administrator resets passwords with `scripts/init_db.py`.
- **Encryption of the database file.** `data/app/smartlistener.db` is protected only by the computer's own file
  permissions. Its sensitive values (password hashes, token hashes) are hashed, not encrypted.
- **Dashboard sign-in.** The dashboard holds the session token in server memory for each browser tab. It is meant
  to be used on the same computer as the API.
- **Rate limiting** applies to sign-in only, not to other routes.
