import gzip
import json

import pandas as pd

from cib import io_botsim, io_fox8, schema


def test_botsim_kinds_labels_split():
    users = pd.DataFrame({"user_id": [f"h{i}" for i in range(io_botsim.N_HUMANS)] + ["b0"]})
    data = {
        "h0": {"posts": [{"submission_id": "s1", "created_utc": "2023-07-01 00:00:00", "subreddit": "news", "posts": "t"}],
               "comment_1": [], "comment_2": []},
        "b0": {"posts": [],
               "comment_1": [{"comment_id": "c1", "link_id": "t3_s1", "parent_id": "t3_s1", "subreddit": "news",
                              "created_utc": "2023-07-01 00:05:00", "comment_body": "x"}],
               "comment_2": [{"comment_id": "c2", "link_id": "t3_s1", "parent_id": "t1_c1", "subreddit": "news",
                              "created_utc": "2024-01-01 00:00:00", "comment_body": "y"}]},
    }
    df = io_botsim.from_raw(data, users, "salt", "2023-12-20T00:00:00Z")
    schema.validate(df)
    assert df.kind.tolist() == ["post", "comment", "reply"]
    assert df.target_id.tolist() == ["s1", "s1", "s1"]
    assert df.split.tolist() == ["explore", "explore", "confirm"]
    lab = schema.account_labels(df)
    assert lab[schema.hash_account("b0", "salt")] and not lab[schema.hash_account("h0", "salt")]


def test_fox8_kinds(tmp_path):
    tw = lambda i, **k: {"id_str": i, "created_at": "Mon Apr 24 10:00:00 +0000 2023", "text": "hi", **k}
    rec = {"user_id": 1, "label": "bot", "dataset": "fox8",
           "user_tweets": [tw("1"), tw("2", retweeted_status={"id_str": "9"}), tw("3", in_reply_to_status_id_str="1")]}
    p = tmp_path / "f.ndjson.gz"
    with gzip.open(p, "wt") as f:
        f.write(json.dumps(rec) + "\n")
    df = io_fox8.load(str(p), "salt")
    schema.validate(df)
    assert dict(zip(df.msg_id, df.kind)) == {"1": "post", "2": "repost", "3": "reply"}
    assert df.loc[df.msg_id == "2", "text"].item() == ""
    assert (df.campaign_id == "fox8").all()
