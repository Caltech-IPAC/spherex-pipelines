import sys
import spx_butler.script.make_repaired_l2b_chain as mod


class DummyRegistry:
    def __init__(self, collection_exists=True):
        self.collection_exists = collection_exists
        self.register_calls = []
        self.chain_calls = []

    def getCollectionType(self, name):
        if not self.collection_exists:
            raise RuntimeError("missing")
        return "CHAINED"

    def registerCollection(self, name, type):
        self.register_calls.append((name, type))

    def setCollectionChain(self, name, chain):
        self.chain_calls.append((name, chain))


class DummyButler:
    def __init__(self, registry):
        self.registry = registry


def test_main_registers_missing_chain(monkeypatch, capsys):
    reg = DummyRegistry(collection_exists=False)

    monkeypatch.setattr(mod, "Butler", lambda path, writeable: DummyButler(reg))
    monkeypatch.setattr(
        mod,
        "CollectionType",
        type("CollectionType", (), {"CHAINED": "CHAINED"}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "-b", "/repo",
            "--chain", "u/test/chain",
            "--retry-run", "u/test/retry",
            "--original-run", "u/test/original",
        ],
    )

    mod.main()

    assert reg.register_calls == [("u/test/chain", "CHAINED")]
    assert reg.chain_calls == [
        ("u/test/chain", ["u/test/retry", "u/test/original"])
    ]

    out = capsys.readouterr().out
    assert "Created/updated chain u/test/chain = [u/test/retry, u/test/original]" in out


def test_main_uses_existing_chain_without_register(monkeypatch):
    reg = DummyRegistry(collection_exists=True)

    monkeypatch.setattr(mod, "Butler", lambda path, writeable: DummyButler(reg))
    monkeypatch.setattr(
        mod,
        "CollectionType",
        type("CollectionType", (), {"CHAINED": "CHAINED"}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "-b", "/repo",
            "--chain", "u/test/chain",
            "--retry-run", "u/test/retry",
            "--original-run", "u/test/original",
        ],
    )

    mod.main()

    assert reg.register_calls == []
    assert reg.chain_calls == [
        ("u/test/chain", ["u/test/retry", "u/test/original"])
    ]
