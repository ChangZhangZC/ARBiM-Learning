import pytest
from experiments.geometry_flow.episode_selection import resolve_episode_ids

@pytest.mark.parametrize("selection,expected",[
    ([3,1],[3,1]), ("0:3",[0,1,2]), ("all",list(range(5)))
])
def test_selection(selection,expected):
    assert resolve_episode_ids(selection,5)==expected

@pytest.mark.parametrize("selection",[[],None,True,[True],[1,1],[-1],[5],
                                      "0:6","2:2","4:2","-1:2","0:1:2","ALL",3])
def test_invalid_selection(selection):
    with pytest.raises(ValueError):
        resolve_episode_ids(selection,5)
