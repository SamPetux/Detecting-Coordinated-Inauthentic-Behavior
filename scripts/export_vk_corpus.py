"""Export the civic-monitor-nn PostgreSQL database to a pseudonymised corpus for coordination analysis.

Runs on the machine that holds the database. Account IDs never leave it in clear form: they are replaced by
sha256(CIB_SALT + id)[:16]. Keep CIB_SALT private and identical across exports so hashes stay comparable.

    export DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/civic
    export CIB_SALT=<long random string>
    python scripts/export_vk_corpus.py --since 2026-03-01 --out corpus_cib_2026-11.csv.gz
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

POSTS_SQL = """
SELECT g.vk_id AS group_vk_id, g.screen_name AS grp, p.vk_post_id,
       COALESCE(p.from_id, p.author_vk_id, p.owner_id) AS raw_author,
       p.date, p.text, p.district_id, p.likes_count, p.reposts_count, p.comments_count, p.is_pinned
FROM posts p JOIN groups g ON g.id = p.group_id
WHERE p.date >= :since
"""

COMMENTS_SQL = """
SELECT g.vk_id AS group_vk_id, g.screen_name AS grp, p.vk_post_id, c.vk_comment_id, c.parent_id,
       c.from_id AS raw_author, c.date, c.text, COALESCE(c.district_id, p.district_id) AS district_id,
       c.likes_count
FROM comments c JOIN posts p ON p.id = c.post_id JOIN groups g ON g.id = p.group_id
WHERE c.date >= :since
"""

COLUMNS = ["kind", "msg_key", "post_key", "parent_key", "group", "date_utc", "account", "author_is_community",
           "text", "district_id", "likes_count", "reposts_count", "comments_count", "is_pinned"]


def pseudonymise(raw: pd.Series, salt: str) -> pd.Series:
    def h(v) -> str | None:
        if pd.isna(v):
            return None
        return hashlib.sha256(f"{salt}{int(v)}".encode()).hexdigest()[:16]
    return raw.map(h)


def build(posts: pd.DataFrame, comments: pd.DataFrame, salt: str) -> pd.DataFrame:
    # Wall post IDs are only unique within a wall, so every key is prefixed by the group's VK id.
    p = pd.DataFrame({
        "kind": "post",
        "msg_key": posts.group_vk_id.astype(str) + "_p" + posts.vk_post_id.astype(str),
        "group": posts.grp,
        "date_utc": pd.to_datetime(posts.date, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "account": pseudonymise(posts.raw_author, salt),
        "author_is_community": posts.raw_author < 0,
        "text": posts.text,
        "district_id": posts.district_id,
        "likes_count": posts.likes_count,
        "reposts_count": posts.reposts_count,
        "comments_count": posts.comments_count,
        "is_pinned": posts.is_pinned,
    })
    p["post_key"] = p.msg_key
    p["parent_key"] = None

    prefix = comments.group_vk_id.astype(str) + "_c"
    c = pd.DataFrame({
        "kind": "comment",
        "msg_key": prefix + comments.vk_comment_id.astype(str),
        "post_key": comments.group_vk_id.astype(str) + "_p" + comments.vk_post_id.astype(str),
        "parent_key": (prefix + comments.parent_id.astype("Int64").astype(str)).where(comments.parent_id.notna()),
        "group": comments.grp,
        "date_utc": pd.to_datetime(comments.date, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "account": pseudonymise(comments.raw_author, salt),
        "author_is_community": comments.raw_author < 0,
        "text": comments.text,
        "district_id": comments.district_id,
        "likes_count": comments.likes_count,
    })
    out = pd.concat([p, c], ignore_index=True)[COLUMNS]
    return out.drop_duplicates("msg_key").sort_values("date_utc", kind="stable")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-03-01")
    ap.add_argument("--out", default="corpus_cib.csv.gz")
    args = ap.parse_args()

    url, salt = os.environ.get("DATABASE_URL"), os.environ.get("CIB_SALT")
    if not url or not salt or len(salt) < 16:
        print("Set DATABASE_URL and CIB_SALT (at least 16 characters).", file=sys.stderr)
        return 2

    engine = create_engine(url)
    with engine.connect() as conn:
        posts = pd.read_sql(text(POSTS_SQL), conn, params={"since": args.since})
        comments = pd.read_sql(text(COMMENTS_SQL), conn, params={"since": args.since})

    out = build(posts, comments, salt)
    out.to_csv(args.out, index=False, compression="gzip")

    n_posts, n_comments = (out.kind == "post").sum(), (out.kind == "comment").sum()
    print(f"posts={n_posts} comments={n_comments} accounts={out.account.nunique()} "
          f"missing_author={out.account.isna().sum()} replies={out.parent_key.notna().sum()} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
