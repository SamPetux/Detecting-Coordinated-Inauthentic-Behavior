import pandas as pd

from cib import io_vk, schema

H = "0123456789abcdef"


def test_from_export_kinds_and_splits():
    raw = pd.DataFrame({
        "kind": ["post", "comment", "comment", "comment"],
        "msg_key": ["1_p1", "1_c1", "1_c2", "1_c3"],
        "post_key": ["1_p1"] * 4,
        "parent_key": [None, None, "1_c1", None],
        "group": ["g"] * 4,
        # 2026-05-31 21:00Z is already June 1 in Moscow; the last row is past the confirm window
        "date_utc": ["2026-05-31T20:59:59Z", "2026-05-31T21:00:00Z", "2026-06-02T00:00:00Z", "2026-11-16T12:00:00Z"],
        "account": [H] * 4,
        "author_is_community": ["True", "False", "False", "False"],
        "text": ["p", "c", None, "late"],
    })
    df = io_vk.from_export(raw, "2026-06-01T00:00:00+03:00", "2026-11-16T00:00:00+03:00")
    schema.validate(df)
    assert df.kind.tolist() == ["post", "comment", "reply"]
    assert df.split.tolist() == ["explore", "confirm", "confirm"]
    assert df.author_is_community.tolist() == [True, False, False]
    assert df.text.iloc[2] == ""
