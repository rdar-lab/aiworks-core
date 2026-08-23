from urllib.parse import urlparse, unquote

import psycopg2
from django.conf import settings
from django.core.management.base import BaseCommand


def _print_data(write, cols, rows):
    write("")
    write(" | ".join(cols))
    write("-" * 120)
    for row in rows:
        write(" | ".join(str(x) for x in row))


def _get_command_output(cur, cmd):
    cur.execute(cmd)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    return cols, rows


class Command(BaseCommand):
    help = "Show PgBouncer pool statistics"

    def handle(self, *args, **options):
        db_url = settings.DATABASE_URL
        parsed = urlparse(db_url)

        dsn = {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            "user": parsed.username or "",
            "password": unquote(parsed.password) if parsed.password else "",
            "dbname": "pgbouncer",
        }

        conn = None
        cur = None
        try:
            conn = psycopg2.connect(**dsn)
            conn.autocommit = True
            cur = conn.cursor()

            server_cols, server_rows = _get_command_output(cur, "SHOW SERVERS")
            client_cols, client_rows = _get_command_output(cur, "SHOW CLIENTS")
            pool_cols, pool_rows = _get_command_output(cur, "SHOW POOLS")
            lists_cols, lists_rows = _get_command_output(cur, "SHOW LISTS")

        finally:
            if cur:
                try:
                    cur.close()
                except Exception as _exp:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception as _exp:
                    pass

        pool_stats = {}
        for pr in pool_rows:
            db = pr[pool_cols.index("database")]
            user = pr[pool_cols.index("user")]
            pool_stats[(db, user)] = {
                "sv_active": int(pr[pool_cols.index("sv_active")]),
                "sv_idle": int(pr[pool_cols.index("sv_idle")]),
                "sv_used": int(pr[pool_cols.index("sv_used")]),
                "cl_waiting": int(pr[pool_cols.index("cl_waiting")]),
            }

        idle = sum([pool.get("sv_idle", 0) for pool in pool_stats.values()])
        borrowed = (sum([pool.get("sv_active", 0) for pool in pool_stats.values()]) +
                    sum([pool.get("sv_used", 0) for pool in pool_stats.values()]))
        total = idle + borrowed
        self.stdout.write(f"POOL: total={total}, idle={idle}, borrowed={borrowed}")

        db_total = len(server_rows)
        db_idle = sum(1 for r in server_rows if r[server_cols.index("state")] == "idle")
        self.stdout.write(f"DB: total={db_total}, idle={db_idle}")

        self.stdout.write('')
        self.stdout.write('=== SERVER CONNECTIONS ===')
        _print_data(self.stdout.write, server_cols, server_rows)
        self.stdout.write('')
        self.stdout.write('=== CLIENT CONNECTIONS ===')
        _print_data(self.stdout.write, client_cols, client_rows)
        self.stdout.write('')
        self.stdout.write('=== POOL STATS ===')
        _print_data(self.stdout.write, pool_cols, pool_rows)
        self.stdout.write('')
        self.stdout.write('=== LISTS STATS ===')
        _print_data(self.stdout.write, lists_cols, lists_rows)
