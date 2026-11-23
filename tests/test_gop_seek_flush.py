import pytest
import media.gop_seek_flush as mod

CASES = [
    ("keyframe_locate", (1,)),
    ("precise_seek", (1,)),
    ("flush_rebuild", (1,)),
    ("pts_reorder", (1,)),
    ("speed_align", (1,)),
    ("cache_key", (1,)),
    ("cache_invalidate", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("gop_scan", (1,)),
    ("metrics", (1,)),
    ("timeline_map", (1,)),
    ("b_frame", (1,)),
    ("determinism", (1,)),
]


@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
