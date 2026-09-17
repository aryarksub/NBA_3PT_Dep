import numpy as np
import pytest
from nba_geometry import is_three, is_three_vec
from counterfactual import candidate_grid

@pytest.mark.parametrize('x', [2.,5.25,10.,84.,88.75,92.])
def test_both_corner_lines_at_both_baskets(x):
    assert is_three(x,2.99) and is_three(x,47.01)
    assert not is_three(x,3.) and not is_three(x,47.)
    assert not is_three(x,3.01) and not is_three(x,46.99)

def test_arc_and_symmetries():
    assert not is_three(29.,25.)
    assert is_three(29.001,25.)
    assert not is_three(28.999,25.)
    pts=np.random.default_rng(1).uniform([0,0],[94,50],(1000,2))
    assert np.array_equal(is_three_vec(pts),is_three_vec([94,50]-pts))
    assert np.array_equal(is_three_vec(pts),is_three_vec(pts*[1,-1]+[0,50]))

@pytest.mark.parametrize('step',[1.,.5,.25,.1])
def test_grids_respect_physical_boundary(step):
    pts=candidate_grid({'shooter_x':5.25,'shooter_y':2.},cell_size=step,max_radius=8)
    assert len(pts)>0 and (pts[:,1]>=3-1e-9).all()
    assert not is_three_vec(pts).any()
    assert (np.linalg.norm(pts-[5.25,2],axis=1)<=8+1e-9).all()
