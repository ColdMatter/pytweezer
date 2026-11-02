from pytweezer.GUI.tweezer_loading_plots import TweezerLoadingPlots
from tests.test_analysis import SITE_CENTRES, loading_window, patterned_frame


def test_plots_draw_a_result_and_report_its_loading(qapp):
    head, photons = loading_window(patterned_frame(shot) for shot in range(40))
    plots = TweezerLoadingPlots()

    plots.show_result(head, photons)

    assert f"loading {head['loading'] * 100:.1f} %" in plots.readout.text()
    assert len(plots.site_map.data) == len(SITE_CENTRES)


def test_plots_show_the_error_of_a_result_without_photons(qapp):
    plots = TweezerLoadingPlots()
    head, photons = loading_window(patterned_frame(shot) for shot in range(40))
    plots.show_result(head, photons)

    plots.show_result(
        {"shots": 3, "sites": 64, "error": "Found no 64 trap sites"}, None
    )

    assert plots.readout.text() == "Found no 64 trap sites"
    assert len(plots.site_map.data) == 0


def test_plots_warn_when_the_sites_were_fitted_to_another_array(qapp):
    head, photons = loading_window(patterned_frame(shot) for shot in range(40))
    plots = TweezerLoadingPlots()

    plots.show_result(head | {"sites_match": False}, photons)
    assert "fitted to another array" in plots.readout.text()

    plots.show_result(head | {"sites_match": True}, photons)
    assert "fitted to another array" not in plots.readout.text()
