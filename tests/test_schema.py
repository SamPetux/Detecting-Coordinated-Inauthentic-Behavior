import pandas as pd
import pytest

from cib import schema


def _rows(**over):
    base = dict(msg_id=["a", "b"], account_id=[schema.hash_account(1, "s"), schema.hash_account(2, "s")],
                ts=[1, 2], kind=["post", "comment"], target_id=["a", "a"], community=["g", "g"],
                source=["vk", "vk"], split=["explore", "explore"])
    base.update(over)
    return schema.conform(pd.DataFrame(base))


def test_conform_adds_columns_and_validates():
    df = _rows()
    assert list(df.columns) == list(schema.COLUMNS)
    schema.validate(df)


def test_hash_is_stable_and_salted():
    assert schema.hash_account(42, "x") == schema.hash_account("42", "x")
    assert schema.hash_account(42, "x") != schema.hash_account(42, "y")
    assert len(schema.hash_account(42, "x")) == schema.ACCOUNT_ID_LEN


@pytest.mark.parametrize("over", [dict(msg_id=["a", "a"]), dict(kind=["post", "like"]),
                                  dict(split=["explore", "test"]), dict(account_id=["short", "x" * 16])])
def test_validate_rejects(over):
    with pytest.raises(ValueError):
        schema.validate(_rows(**over))


def test_account_labels():
    df = _rows(campaign_id=[None, "c1"])
    lab = schema.account_labels(df)
    assert lab[df.account_id[1]] and not lab[df.account_id[0]]
