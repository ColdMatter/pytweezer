from pytweezer.GUI.viewers.slm_monitor import SlmMonitor


def test_a_header_with_resolution_but_no_offset_is_drawn_in_pixels():
    head = {"_imgresolution": [1e-6, 1e-6]}

    assert SlmMonitor._placement(head) == ("px", {})


def test_a_header_with_resolution_and_offset_is_drawn_in_metres():
    head = {"_imgresolution": [1e-6, 1e-6], "_offset": [0, 0]}

    assert SlmMonitor._placement(head) == ("m", head)
