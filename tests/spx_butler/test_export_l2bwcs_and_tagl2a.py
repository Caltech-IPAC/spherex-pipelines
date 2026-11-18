import importlib
import logging

from astropy.io.fits import Header

mod = importlib.import_module("spx_butler.script.export_l2bwcs_and_tagl2a")


class DummyDatasetType:
    def __init__(self, name, dimensions, storageClass, universe):
        self.name = name
        self.dimensions = dimensions
        self.storageClass = storageClass
        self.universe = universe


class DummyObservationId:
    def __init__(self, observation):
        self.observation = observation
        self.year = 2026
        self.week = 10
        self.plan_number = 2
        self.plan_version = 1
        self.large_slew_counter = 7
        self.small_slew_counter = observation % 100 if isinstance(observation, int) else 0
        self.slewgroup_id = 1234
        self.slewgroup_counter_id = 5678
        self.plan_name = "2026W10_2A"
        self.sps_id = 999
        self.plan_id = 202610201


class DummyDataId:
    def __init__(self, mapping):
        self.mapping = dict(mapping)

    def __getitem__(self, key):
        return self.mapping[key]


class DummyRef:
    def __init__(self, mapping):
        self.dataId = DummyDataId(mapping)


def dummy_header(finast=0, sps_flags=""):
    hdr = Header()
    hdr["FINAST"] = finast
    hdr["SPS_FLAGS"] = sps_flags
    return hdr


class DummyRegistry:
    def __init__(self):
        self.dataset_types = {}
        self.collections = {}
        self.associated = []

    def getDatasetType(self, name):
        if name not in self.dataset_types:
            raise KeyError(name)
        return self.dataset_types[name]

    def registerDatasetType(self, dt):
        self.dataset_types[dt.name] = dt

    def getCollectionType(self, name):
        if name not in self.collections:
            raise KeyError(name)
        return self.collections[name]

    def registerCollection(self, name, type):
        self.collections[name] = type

    def associate(self, tag, refs):
        self.associated.append((tag, list(refs)))


class DummyURI:
    def __init__(self, ospath):
        self.ospath = ospath


class DummyButler:
    def __init__(self, registry, image_by_ref):
        self.registry = registry
        self.dimensions = "DUMMY_UNIVERSE"
        self.image_by_ref = image_by_ref
        self.put_calls = []
        self.level2_refs = []
        self.level2a_lookup = {}
        self.query_calls = []

    def query_datasets(
        self,
        dataset_type,
        collections=None,
        find_first=False,
        data_id=None,
        where=None,
    ):
        self.query_calls.append(
            {
                "dataset_type": dataset_type,
                "collections": collections,
                "findFirst": find_first,
                "data_id": data_id,
                "where": where,
            }
        )

        if dataset_type == "level2":
            return list(self.level2_refs)

        if dataset_type == "level2a":
            key = tuple(sorted(dict(data_id).items()))
            out = self.level2a_lookup.get(key, [])
            if find_first and out:
                return [out[0]]
            return list(out)

        raise AssertionError(f"Unexpected dataset_type {dataset_type}")

    def get(self, ref):
        return self.image_by_ref[id(ref)]

    def getURI(self, ref):
        # Return a DummyURI with ospath set to the ref id (used as lookup key)
        return DummyURI(str(id(ref)))

    def put(self, payload, dataset_type, dataId, run):
        self.put_calls.append(
            {
                "payload": payload,
                "dataset_type": dataset_type,
                "dataId": dataId,
                "run": run,
            }
        )


def test_ensure_dataset_type_registers_when_missing(monkeypatch):
    reg = DummyRegistry()
    butler = DummyButler(reg, {})

    monkeypatch.setattr(mod, "DatasetType", DummyDatasetType)

    mod.ensure_dataset_type(butler, "l2bwcs")

    assert "l2bwcs" in reg.dataset_types
    dt = reg.dataset_types["l2bwcs"]
    assert dt.name == "l2bwcs"
    assert dt.dimensions == ("instrument", "observation")
    assert dt.storageClass == "StructuredDataDict"
    assert dt.universe == "DUMMY_UNIVERSE"


def test_ensure_dataset_type_noop_when_present(monkeypatch):
    reg = DummyRegistry()
    reg.dataset_types["l2bwcs"] = object()
    butler = DummyButler(reg, {})

    monkeypatch.setattr(mod, "DatasetType", DummyDatasetType)

    mod.ensure_dataset_type(butler, "l2bwcs")

    assert len(reg.dataset_types) == 1


def test_ensure_collection_registers_when_missing():
    reg = DummyRegistry()
    butler = DummyButler(reg, {})

    mod.ensure_collection(butler, "u/test/tag", "TAGGED")

    assert reg.collections["u/test/tag"] == "TAGGED"


def test_ensure_collection_noop_when_present():
    reg = DummyRegistry()
    reg.collections["u/test/tag"] = "TAGGED"
    butler = DummyButler(reg, {})

    mod.ensure_collection(butler, "u/test/tag", "TAGGED")

    assert reg.collections["u/test/tag"] == "TAGGED"


def test_make_slewgroup_key(monkeypatch):
    monkeypatch.setattr(mod, "ObservationId", DummyObservationId)

    key = mod._make_slewgroup_key(202610202022140)

    assert key == (2026, 10, 2, 1, 7)


def test_detector_payload_with_obs_meta(monkeypatch):
    monkeypatch.setattr(mod, "ObservationId", DummyObservationId)

    def fake_build_l2bwcs_payload(spximg, source_collection):
        return {"seed": "payload", "source_collection": source_collection}

    monkeypatch.setattr(mod, "build_l2bwcs_payload", fake_build_l2bwcs_payload)

    ref = DummyRef(
        {
            "instrument": "spx",
            "observation": 202610202022140,
            "detector": 2,
        }
    )
    spximg = dummy_header(finast=3)

    payload = mod._detector_payload_with_obs_meta(ref, spximg, "u/input/run")

    assert payload["seed"] == "payload"
    assert payload["source_collection"] == "u/input/run"
    assert payload["instrument"] == "spx"
    assert payload["observation"] == 202610202022140
    assert payload["detector"] == 2
    assert payload["finast"] == 3
    assert payload["small_slew_counter"] == 40
    assert payload["large_slew_counter"] == "7"
    assert payload["slewgroup_id"] == 1234
    assert payload["slewgroup_counter_id"] == 5678
    assert payload["plan_name"] == "2026W10_2A"
    assert payload["sps_id"] == 999


def test_main_happy_path(monkeypatch, caplog):
    monkeypatch.setattr(mod, "ObservationId", DummyObservationId)
    monkeypatch.setattr(mod, "DatasetType", DummyDatasetType)

    def fake_build_l2bwcs_payload(spximg, source_collection):
        return {"base_payload": True, "source_collection": source_collection}

    monkeypatch.setattr(mod, "build_l2bwcs_payload", fake_build_l2bwcs_payload)

    reg = DummyRegistry()

    ref_bad = DummyRef(
        {"instrument": "spx", "observation": 202610202022140, "detector": 2}
    )
    ref_good_same_obs = DummyRef(
        {"instrument": "spx", "observation": 202610202022140, "detector": 5}
    )
    ref_good_other_obs = DummyRef(
        {"instrument": "spx", "observation": 202610202022141, "detector": 2}
    )

    l2a_ref_for_bad = DummyRef(
        {"instrument": "spx", "observation": 202610202022140, "detector": 2}
    )

    image_by_ref = {
        id(ref_bad): dummy_header(finast=3),
        id(ref_good_same_obs): dummy_header(finast=0),
        id(ref_good_other_obs): dummy_header(finast=0),
    }

    butler = DummyButler(reg, image_by_ref)
    butler.level2_refs = [ref_bad, ref_good_same_obs, ref_good_other_obs]
    butler.level2a_lookup = {
        tuple(
            sorted(
                {
                    "instrument": "spx",
                    "observation": 202610202022140,
                    "detector": 2,
                }.items()
            )
        ): [l2a_ref_for_bad]
    }

    def fake_getheader(path, ext=None):
        return image_by_ref[int(path)]

    monkeypatch.setattr(mod.fits, "getheader", fake_getheader)
    monkeypatch.setattr(mod, "Butler", lambda *args, **kwargs: butler)

    with caplog.at_level(logging.INFO):
        mod.export_l2bwcs_and_tagl2a(
            repo="butler.yaml",
            collections=["u/l2b/run"],
            output_run="u/l2bwcs/run",
            output_tag="u/tag/retry",
            progress_every=0,
        )
    out = caplog.text

    assert "l2bwcs" in reg.dataset_types
    assert reg.collections["u/l2bwcs/run"] == mod.CollectionType.RUN
    assert reg.collections["u/tag/retry"] == mod.CollectionType.TAGGED

    # Only observation 202610202022140 has bad finast, so only 1 l2bwcs written
    assert len(butler.put_calls) == 1

    written_obs = sorted(call["dataId"]["observation"] for call in butler.put_calls)
    assert written_obs == [202610202022140]

    for call in butler.put_calls:
        assert call["dataset_type"] == "l2bwcs"
        assert call["run"] == "u/l2bwcs/run"

        payload = call["payload"]
        assert payload["instrument"] == "spx"
        assert payload["source_collection"] == ["u/l2b/run"]
        assert payload["seed_strategy_version"] == "same_detector_adjacent_small_slew_v1"
        # slewgroup still contains all observations for donor lookup
        assert sorted(payload["slewgroup"]["observations_in_group"]) == [
            202610202022140,
            202610202022141,
        ]
        assert len(payload["slewgroup_solutions"]) == 3

    assert len(reg.associated) == 1
    tag_name, refs = reg.associated[0]
    assert tag_name == "u/tag/retry"
    assert refs == [l2a_ref_for_bad]

    assert "total_level2=3" in out
    assert "good_finast=2" in out
    assert "bad_finast=1" in out
    assert "written_groups=1" in out
    assert "written_observation_level_l2bwcs=1" in out
    assert "tagged_level2a=1" in out
    assert "missing_level2a=0" in out


def test_main_warns_when_level2a_missing(monkeypatch, caplog):
    monkeypatch.setattr(mod, "ObservationId", DummyObservationId)
    monkeypatch.setattr(mod, "DatasetType", DummyDatasetType)

    def fake_build_l2bwcs_payload(spximg, source_collection):
        return {"base_payload": True, "source_collection": source_collection}

    monkeypatch.setattr(mod, "build_l2bwcs_payload", fake_build_l2bwcs_payload)

    reg = DummyRegistry()

    ref_bad = DummyRef(
        {"instrument": "spx", "observation": 202610202022140, "detector": 2}
    )

    image_by_ref = {
        id(ref_bad): dummy_header(finast=4),
    }

    butler = DummyButler(reg, image_by_ref)
    butler.level2_refs = [ref_bad]
    butler.level2a_lookup = {}

    def fake_getheader(path, ext=None):
        return image_by_ref[int(path)]

    monkeypatch.setattr(mod.fits, "getheader", fake_getheader)
    monkeypatch.setattr(mod, "Butler", lambda *args, **kwargs: butler)

    with caplog.at_level(logging.INFO):
        mod.export_l2bwcs_and_tagl2a(
            repo="butler.yaml",
            collections=["u/l2b/run"],
            output_run="u/l2bwcs/run",
            output_tag="u/tag/retry",
            progress_every=0,
        )
    out = caplog.text

    assert len(butler.put_calls) == 1
    assert len(reg.associated) == 0
    assert "no level2a found" in out
    assert "bad_finast=1" in out
    assert "tagged_level2a=0" in out
    assert "missing_level2a=1" in out


def test_main_skips_saa_level2(monkeypatch, caplog):
    monkeypatch.setattr(mod, "ObservationId", DummyObservationId)
    monkeypatch.setattr(mod, "DatasetType", DummyDatasetType)

    monkeypatch.setattr(
        mod,
        "build_l2bwcs_payload",
        lambda spximg, source_collection: {"source_collection": source_collection},
    )

    reg = DummyRegistry()

    ref_saa_bad = DummyRef(
        {"instrument": "spx", "observation": 202610202022140, "detector": 2}
    )

    image_by_ref = {
        id(ref_saa_bad): dummy_header(finast=4, sps_flags="saa"),
    }

    butler = DummyButler(reg, image_by_ref)
    butler.level2_refs = [ref_saa_bad]

    def fake_getheader(path, ext=None):
        return image_by_ref[int(path)]

    monkeypatch.setattr(mod.fits, "getheader", fake_getheader)
    monkeypatch.setattr(mod, "Butler", lambda *args, **kwargs: butler)

    with caplog.at_level(logging.INFO):
        mod.export_l2bwcs_and_tagl2a(
            repo="butler.yaml",
            collections=["u/l2b/run"],
            output_run="u/l2bwcs/run",
            output_tag="u/tag/retry",
            progress_every=0,
        )
    out = caplog.text

    assert len(butler.put_calls) == 0
    assert len(reg.associated) == 0
    assert "total_level2=1" in out
    assert "good_finast=0" in out
    assert "bad_finast=0" in out
    assert "written_groups=0" in out


def test_main_passes_sps_id_where_clause(monkeypatch):
    monkeypatch.setattr(mod, "ObservationId", DummyObservationId)
    monkeypatch.setattr(mod, "DatasetType", DummyDatasetType)
    monkeypatch.setattr(
        mod,
        "build_l2bwcs_payload",
        lambda spximg, source_collection: {"source_collection": source_collection},
    )

    reg = DummyRegistry()
    butler = DummyButler(reg, {})

    monkeypatch.setattr(mod, "Butler", lambda *args, **kwargs: butler)

    mod.export_l2bwcs_and_tagl2a(
        repo="butler.yaml",
        collections=["u/l2b/run"],
        output_run="u/l2bwcs/run",
        output_tag="u/tag/retry",
        sps_plan="2026W18_1B",
        instrument="spx",
        progress_every=0,
    )

    level2_query = next(
        call for call in butler.query_calls if call["dataset_type"] == "level2"
    )
    assert level2_query["where"] == "instrument='spx' AND spsplan=202610201"
