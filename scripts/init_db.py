"""Create the application database, the first admin login and (optionally) the 8 demo machines.

    python scripts/init_db.py                                    # create tables only (safe to repeat)
    python scripts/init_db.py --create-admin admin               # asks you to type a password (it is not shown)
    python scripts/init_db.py --seed-demo --owner-name "Your Name" --owner-email you@example.com

The demo machines are the 8 trained machine IDs (fan and pump, IDs 00 02 04 06). Every one gets the owner you
give here, so alert e-mails in Phase 7 go to a real address of yours, never to an invented one.
Nothing is ever deleted: running this again only adds what is missing.
"""
import argparse
import getpass
import os
import sys

from smart_listener import db, profiles, repo
from smart_listener.config import load_config


def ask_password() -> str:
    env = os.environ.get("SL_ADMIN_PASSWORD")
    if env:
        return env
    while True:
        a = getpass.getpass("Choose an admin password (at least 8 characters, not shown): ")
        b = getpass.getpass("Type it again: ")
        if a == b:
            return a
        print("The two passwords were different. Try again.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Create the database, an admin and demo machines")
    ap.add_argument("--db", help="database file (default: data/app/smartlistener.db or the SL_DB_PATH variable)")
    ap.add_argument("--create-admin", metavar="USERNAME")
    ap.add_argument("--seed-demo", action="store_true", help="register the 8 trained machine IDs as demo machines")
    ap.add_argument("--owner-name")
    ap.add_argument("--owner-email")
    args = ap.parse_args()

    path = args.db or db.default_db_path()
    conn = db.connect(path)
    db.init_schema(conn)
    print(f"Database ready: {path}")

    if args.create_admin:
        try:
            repo.create_admin(conn, args.create_admin, ask_password())
            print(f"Admin '{args.create_admin}' created.")
        except ValueError as e:
            print(f"Admin not created: {e}")
            return 1

    if args.seed_demo:
        if not args.owner_name or not args.owner_email:
            print("--seed-demo needs --owner-name and --owner-email (the person who will receive alert e-mails).")
            return 1
        cfg = load_config()
        added = 0
        for g in profiles.model_groups(cfg):
            mtype, mid = g.split("_", 1)
            name = f"{mtype.capitalize()} {mid[-2:]}"
            try:
                repo.create_machine(conn, name=name, machine_type=mtype, model_group=g, owner_name=args.owner_name,
                                    owner_email=args.owner_email, location="Demo line (simulated)",
                                    notes=f"Simulated from the MIMII {mtype} {mid} recordings. Not a real factory machine.")
                added += 1
            except ValueError:
                pass  # already registered
        print(f"Demo machines added: {added} (already present: {8 - added}).")

    print(f"Admins: {repo.count_admins(conn)} | machines: {len(repo.list_machines(conn, include_inactive=True))}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
