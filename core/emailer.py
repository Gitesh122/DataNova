"""Send reports by e-mail over SMTP (STARTTLS or SSL)."""
from __future__ import annotations

import re
import smtplib
import ssl
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

PRESETS = {
    "Gmail": ("smtp.gmail.com", 587, "starttls"),
    "Outlook / Office 365": ("smtp.office365.com", 587, "starttls"),
    "Yahoo": ("smtp.mail.yahoo.com", 465, "ssl"),
    "Zoho": ("smtp.zoho.com", 465, "ssl"),
    "Custom": ("", 587, "starttls"),
}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def parse_recipients(text: str) -> tuple[list[str], list[str]]:
    items = [x.strip() for x in re.split(r"[,;\s]+", text or "") if x.strip()]
    good = [x for x in items if EMAIL_RE.match(x)]
    bad = [x for x in items if not EMAIL_RE.match(x)]
    return good, bad


def send_email(cfg: dict, recipients: list[str], subject: str, html_body: str,
               attachments: list[tuple[str, bytes, str]] | None = None, cc: list[str] | None = None,
               timeout: int = 30) -> tuple[bool, str]:
    """cfg keys: smtp_server, smtp_port, security ('starttls'|'ssl'|'none'), sender_email, sender_password, sender_name."""
    if not recipients:
        return False, "No valid recipient addresses."
    if not cfg.get("smtp_server") or not cfg.get("sender_email"):
        return False, "SMTP server and sender e-mail are required (Settings → E-mail)."
    msg = MIMEMultipart("mixed")
    sender = cfg["sender_email"]
    msg["From"] = f"{cfg.get('sender_name') or 'DataNova'} <{sender}>"
    msg["To"] = ", ".join(recipients)
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg["Subject"] = subject
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(re.sub(r"<[^>]+>", " ", html_body)[:5000], "plain", "utf-8"))
    alt.attach(MIMEText(html_body, "html", "utf-8"))
    msg.attach(alt)
    for name, data, mime in attachments or []:
        sub = mime.split("/")[-1]
        part = MIMEApplication(data, _subtype=sub)
        part.add_header("Content-Disposition", "attachment", filename=name)
        msg.attach(part)
    try:
        port = int(cfg.get("smtp_port") or 587)
        security = cfg.get("security", "starttls")
        ctx = ssl.create_default_context()
        if security == "ssl":
            server = smtplib.SMTP_SSL(cfg["smtp_server"], port, timeout=timeout, context=ctx)
        else:
            server = smtplib.SMTP(cfg["smtp_server"], port, timeout=timeout)
            server.ehlo()
            if security == "starttls":
                server.starttls(context=ctx)
                server.ehlo()
        if cfg.get("sender_password"):
            server.login(sender, cfg["sender_password"])
        server.sendmail(sender, recipients + (cc or []), msg.as_string())
        server.quit()
        return True, f"E-mail sent to {len(recipients) + len(cc or [])} recipient(s)."
    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed. For Gmail, enable 2-step verification and use an App Password."
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed to send e-mail: {exc}"


def test_connection(cfg: dict) -> tuple[bool, str]:
    try:
        port = int(cfg.get("smtp_port") or 587)
        if cfg.get("security") == "ssl":
            server = smtplib.SMTP_SSL(cfg["smtp_server"], port, timeout=15, context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(cfg["smtp_server"], port, timeout=15)
            if cfg.get("security") == "starttls":
                server.starttls(context=ssl.create_default_context())
        if cfg.get("sender_password"):
            server.login(cfg["sender_email"], cfg["sender_password"])
        server.quit()
        return True, "Connection and login succeeded ✅"
    except Exception as exc:  # noqa: BLE001
        return False, f"Connection failed: {exc}"
