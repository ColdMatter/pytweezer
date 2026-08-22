from pytweezer.experiment.motmaster_client import MotMasterClient
from pytweezer.drivers.imagemX2 import ImagEMX2Camera
from pytweezer.drivers.slm import SLM
from pytweezer.drivers.thorcam import ThorCamera
from pytweezer.experiment.experiment_parameter_manager import ExpParameterManager
import pytweezer.phasemask as pm
import pytweezer.analysis as an
import numpy as np
import matplotlib.pyplot as plt
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import Optimizer
from skopt.plots import plot_convergence

ExpParams = ExpParameterManager()

def DetectMOTPosition(motcam, exp, img_it=10, bg_it=2):
    exp.set_save_toggle(False)
    exp.set_run_until_stopped(False)
    exp.set_motmaster_experiment("RbTweezerBasic2026_2")
    motcam.set_trigger_source("bulb")

    exp.set_iterations(bg_it)
    motcam.setup_acquisition(nframes=bg_it)
    motcam.start_acquisition()
    exp.start_motmaster_experiment({"tMOTLoadRepVCO": ExpParams.get_parameter("tMOTLoadRepVCO"), "tMOTLoadCoolVCO": ExpParams.get_parameter("tMOTLoadCoolVCO"),
                                    "tMOTLoadRepVVA": ExpParams.get_parameter("tMOTLoadRepVVA"), "tMOTLoadCoolVVA": ExpParams.get_parameter("tMOTLoadCoolVVA"), 
                                    "tImgRepVCO1": ExpParams.get_parameter("rep_vco_resonance"), "tImgCoolVCO1": ExpParams.get_parameter("cool_vco_resonance"), 
                                    "tDelay1": 1, "tMolassesDuration": 1, "tImgCoolVVA1": 4.0, "tImgRepVVA1": 4.0, "tMOTccValue": 0.0, "tDelay2": 1, 
                                    "tTweezerExposure1":100, "tTweezerExposure2":100, "PatternLength": 50000})
    imgs_bg = motcam.acquire_n_frames(nframes=bg_it)

    exp.set_iterations(img_it)
    motcam.setup_acquisition(nframes=img_it)
    motcam.start_acquisition()
    exp.start_motmaster_experiment({"tMOTLoadRepVCO": ExpParams.get_parameter("tMOTLoadRepVCO"), "tMOTLoadCoolVCO": ExpParams.get_parameter("tMOTLoadCoolVCO"),
                                    "tMOTLoadRepVVA": ExpParams.get_parameter("tMOTLoadRepVVA"), "tMOTLoadCoolVVA": ExpParams.get_parameter("tMOTLoadCoolVVA"), 
                                    "tImgRepVCO1": ExpParams.get_parameter("rep_vco_resonance"), "tImgCoolVCO1": ExpParams.get_parameter("cool_vco_resonance"), 
                                    "tDelay1": 1, "tMolassesDuration": 1, "tImgCoolVVA1": 4.0, "tImgRepVVA1": 4.0, "tDelay2": 1, 
                                    "tTweezerExposure1":100, "tTweezerExposure2":100, "PatternLength": 25000})
    imgs = motcam.acquire_n_frames(nframes=img_it)
    img_mean = imgs.mean(axis=0) - imgs_bg.mean(axis=0)

    try:
        gaussian_params = an.fit_gaussian(img_mean)
        centre_x, centre_y = int(gaussian_params["x0"]), int(gaussian_params["y0"])
        print(f"MOT detected at (x, y): ({centre_x}, {centre_y})")
    except Exception as e:
        print(f"Could not detect MOT: {e}")

    return centre_x, centre_y, img_mean

def DetectResonances(motcam, exp, img_it=10, bg_it=2):
    exp.set_save_toggle(False)
    exp.set_run_until_stopped(False)
    exp.set_motmaster_experiment("RbTweezerBasic2026_2")
    motcam.set_trigger_source("bulb")

    centre_x, centre_y, _ = DetectMOTPosition(motcam, exp, img_it=img_it, bg_it=bg_it)

    vco_list = np.linspace(3.0, 5.0, 20)
    counts = []
    print(f"Scanning cooling VCO...")
    for it, vco in enumerate(vco_list):
        exp.set_iterations(bg_it)
        motcam.setup_acquisition(nframes=bg_it)
        motcam.start_acquisition()
        exp.start_motmaster_experiment({"tMOTLoadRepVCO": ExpParams.get_parameter("tMOTLoadRepVCO"), "tMOTLoadCoolVCO": ExpParams.get_parameter("tMOTLoadCoolVCO"), 
                                        "tMOTLoadRepVVA": ExpParams.get_parameter("tMOTLoadRepVVA"), "tMOTLoadCoolVVA": ExpParams.get_parameter("tMOTLoadCoolVVA"),
                                        "tImgRepVCO1": ExpParams.get_parameter("rep_vco_resonance"), "tImgCoolVCO1": vco, 
                                        "tDelay1": 1, "tMolassesDuration": 1, "tImgCoolVVA1": 4.0, "tImgRepVVA1": 4.0, "tMOTccValue": 0.0, "tDelay2": 1, 
                                        "tTweezerExposure1":100, "tTweezerExposure2":100, "PatternLength": 25000})
        imgs_bg = motcam.acquire_n_frames(nframes=bg_it)

        exp.set_iterations(img_it)
        motcam.setup_acquisition(nframes=img_it)
        motcam.start_acquisition()
        exp.start_motmaster_experiment({"tMOTLoadRepVCO": ExpParams.get_parameter("tMOTLoadRepVCO"), "tMOTLoadCoolVCO": ExpParams.get_parameter("tMOTLoadCoolVCO"),
                                        "tMOTLoadRepVVA": ExpParams.get_parameter("tMOTLoadRepVVA"), "tMOTLoadCoolVVA": ExpParams.get_parameter("tMOTLoadCoolVVA"), 
                                        "tImgRepVCO1": ExpParams.get_parameter("rep_vco_resonance"), "tImgCoolVCO1": vco, 
                                        "tDelay1": 1, "tMolassesDuration": 1, "tImgCoolVVA1": 4.0, "tImgRepVVA1": 4.0, "tDelay2": 1, 
                                        "tTweezerExposure1":100, "tTweezerExposure2":100, "PatternLength": 25000})
        imgs = motcam.acquire_n_frames(nframes=img_it)
        img_mean = imgs.mean(axis=0) - imgs_bg.mean(axis=0)
        total_counts = an.get_total_counts(img_mean, centre_x, centre_y, window_size=50)
        counts.append(total_counts)
        print(f"Iteration {it+1}/{len(vco_list)}  |  Cooling VCO = {vco:.3f} V, Total Counts = {total_counts/1e6:.2f} E 6")

    max_index = np.argmax(counts)
    cool_vco_resonance = vco_list[max_index]
    print(f"Cooling VCO resonance: {cool_vco_resonance}")
    ExpParams.set_parameter("cool_vco_resonance", cool_vco_resonance)

    vco_list = np.linspace(0.0, 5.0, 20)
    counts = []
    print(f"Scanning repump VCO...")
    for it, vco in enumerate(vco_list):
        exp.set_iterations(bg_it)
        motcam.setup_acquisition(nframes=bg_it)
        motcam.start_acquisition()
        exp.start_motmaster_experiment({"tMOTLoadRepVCO": ExpParams.get_parameter("tMOTLoadRepVCO"), "tMOTLoadCoolVCO": ExpParams.get_parameter("tMOTLoadCoolVCO"), 
                                        "tMOTLoadRepVVA": ExpParams.get_parameter("tMOTLoadRepVVA"), "tMOTLoadCoolVVA": ExpParams.get_parameter("tMOTLoadCoolVVA"),
                                        "tImgRepVCO1": vco, "tImgCoolVCO1": ExpParams.get_parameter("cool_vco_resonance"), 
                                        "tDelay1": 1, "tMolassesDuration": 1, "tImgCoolVVA1": 4.0, "tImgRepVVA1": 4.0, "tMOTccValue": 0.0, "tDelay2": 1, 
                                        "tTweezerExposure1":100, "tTweezerExposure2":100, "PatternLength": 25000})
        imgs_bg = motcam.acquire_n_frames(nframes=bg_it)

        exp.set_iterations(img_it)
        motcam.setup_acquisition(nframes=img_it)
        motcam.start_acquisition()
        exp.start_motmaster_experiment({"tMOTLoadRepVCO": ExpParams.get_parameter("tMOTLoadRepVCO"), "tMOTLoadCoolVCO": ExpParams.get_parameter("tMOTLoadCoolVCO"), 
                                        "tMOTLoadRepVVA": ExpParams.get_parameter("tMOTLoadRepVVA"), "tMOTLoadCoolVVA": ExpParams.get_parameter("tMOTLoadCoolVVA"),
                                        "tImgRepVCO1": vco, "tImgCoolVCO1": ExpParams.get_parameter("cool_vco_resonance"), 
                                        "tDelay1": 1, "tMolassesDuration": 1, "tImgCoolVVA1": 4.0, "tImgRepVVA1": 4.0, "tDelay2": 1, 
                                        "tTweezerExposure1":100, "tTweezerExposure2":100, "PatternLength": 25000})
        imgs = motcam.acquire_n_frames(nframes=img_it)
        img_mean = imgs.mean(axis=0) - imgs_bg.mean(axis=0)
        total_counts = an.get_total_counts(img_mean, centre_x, centre_y, window_size=50)
        counts.append(total_counts)
        print(f"Iteration {it+1}/{len(vco_list)}  |  Repump VCO = {vco:.3f} V, Total Counts = {total_counts/1e6:.2f} E 6")

    max_index = np.argmax(counts)
    rep_vco_resonance = vco_list[max_index]
    print(f"Repump VCO resonance: {rep_vco_resonance}")
    ExpParams.set_parameter("rep_vco_resonance", rep_vco_resonance)

    ExpParams.save_parameters()

def OptimiseMOTNumber(motcam, exp, img_it=10, bg_it=2):
    exp.set_save_toggle(False)
    exp.set_run_until_stopped(False)
    exp.set_motmaster_experiment("RbTweezerBasic2026_2")
    motcam.set_trigger_source("bulb")

    centre_x, centre_y, _ = DetectMOTPosition(motcam, exp, img_it=img_it, bg_it=bg_it)

    vco_list = np.linspace(3.0, 5.0, 20)
    counts = []
    print(f"Scanning MOT cooling VCO...")
    for it, vco in enumerate(vco_list):
        exp.set_iterations(bg_it)
        motcam.setup_acquisition(nframes=bg_it)
        motcam.start_acquisition()
        exp.start_motmaster_experiment({"tMOTLoadRepVCO": ExpParams.get_parameter("tMOTLoadRepVCO"), "tMOTLoadCoolVCO": vco, 
                                        "tMOTLoadRepVVA": ExpParams.get_parameter("tMOTLoadRepVVA"), "tMOTLoadCoolVVA": ExpParams.get_parameter("tMOTLoadCoolVVA"),
                                        "tImgRepVCO1": ExpParams.get_parameter("rep_vco_resonance"), "tImgCoolVCO1": ExpParams.get_parameter("cool_vco_resonance"), 
                                        "tDelay1": 1, "tMolassesDuration": 1, "tImgCoolVVA1": 4.0, "tImgRepVVA1": 4.0, "tMOTccValue": 0.0, "tDelay2": 1, 
                                        "tTweezerExposure1":100, "tTweezerExposure2":100, "PatternLength": 25000})
        imgs_bg = motcam.acquire_n_frames(nframes=bg_it)

        exp.set_iterations(img_it)
        motcam.setup_acquisition(nframes=img_it)
        motcam.start_acquisition()
        exp.start_motmaster_experiment({"tMOTLoadRepVCO": ExpParams.get_parameter("tMOTLoadRepVCO"), "tMOTLoadCoolVCO": vco, 
                                        "tMOTLoadRepVVA": ExpParams.get_parameter("tMOTLoadRepVVA"), "tMOTLoadCoolVVA": ExpParams.get_parameter("tMOTLoadCoolVVA"),
                                        "tImgRepVCO1": ExpParams.get_parameter("rep_vco_resonance"), "tImgCoolVCO1": ExpParams.get_parameter("cool_vco_resonance"), 
                                        "tDelay1": 1, "tMolassesDuration": 1, "tImgCoolVVA1": 4.0, "tImgRepVVA1": 4.0, "tDelay2": 1, 
                                        "tTweezerExposure1":100, "tTweezerExposure2":100, "PatternLength": 25000})
        imgs = motcam.acquire_n_frames(nframes=img_it)
        img_mean = imgs.mean(axis=0) - imgs_bg.mean(axis=0)
        total_counts = an.get_total_counts(img_mean, centre_x, centre_y, window_size=50)
        counts.append(total_counts)
        print(f"Iteration {it+1}/{len(vco_list)}  |  MOT Cooling VCO = {vco:.3f} V, Total Counts = {total_counts/1e6:.2f} E 6")

    max_index = np.argmax(counts)
    cool_vco_max = vco_list[max_index]
    print(f"MOT Cooling VCO resonance: {cool_vco_max}")
    ExpParams.set_parameter("tMOTLoadCoolVCO", cool_vco_max)

    vva_list = np.linspace(0.4, 1.5, 20)
    counts = []
    print(f"Scanning MOT cooling VVA...")
    for it, vva in enumerate(vva_list):
        exp.set_iterations(bg_it)
        motcam.setup_acquisition(nframes=bg_it)
        motcam.start_acquisition()
        exp.start_motmaster_experiment({"tMOTLoadRepVCO": ExpParams.get_parameter("tMOTLoadRepVCO"), "tMOTLoadCoolVCO": ExpParams.get_parameter("tMOTLoadCoolVCO"),
                                        "tMOTLoadRepVVA": ExpParams.get_parameter("tMOTLoadRepVVA"), "tMOTLoadCoolVVA": vva,
                                        "tImgRepVCO1": ExpParams.get_parameter("rep_vco_resonance"), "tImgCoolVCO1": ExpParams.get_parameter("cool_vco_resonance"), 
                                        "tDelay1": 1, "tMolassesDuration": 1, "tImgCoolVVA1": 4.0, "tImgRepVVA1": 4.0, "tMOTccValue": 0.0, "tDelay2": 1, 
                                        "tTweezerExposure1":100, "tTweezerExposure2":100, "PatternLength": 25000})
        imgs_bg = motcam.acquire_n_frames(nframes=bg_it)

        exp.set_iterations(img_it)
        motcam.setup_acquisition(nframes=img_it)
        motcam.start_acquisition()
        exp.start_motmaster_experiment({"tMOTLoadRepVCO": ExpParams.get_parameter("tMOTLoadRepVCO"), "tMOTLoadCoolVCO": ExpParams.get_parameter("tMOTLoadCoolVCO"),
                                        "tMOTLoadRepVVA": ExpParams.get_parameter("tMOTLoadRepVVA"), "tMOTLoadCoolVVA": vva,
                                        "tImgRepVCO1": ExpParams.get_parameter("rep_vco_resonance"), "tImgCoolVCO1": ExpParams.get_parameter("cool_vco_resonance"), 
                                        "tDelay1": 1, "tMolassesDuration": 1, "tImgCoolVVA1": 4.0, "tImgRepVVA1": 4.0, "tDelay2": 1, 
                                        "tTweezerExposure1":100, "tTweezerExposure2":100, "PatternLength": 25000})
        imgs = motcam.acquire_n_frames(nframes=img_it)
        img_mean = imgs.mean(axis=0) - imgs_bg.mean(axis=0)
        total_counts = an.get_total_counts(img_mean, centre_x, centre_y, window_size=50)
        counts.append(total_counts)
        print(f"Iteration {it+1}/{len(vva_list)}  |  MOT Cooling VVA = {vva:.3f} V, Total Counts = {total_counts/1e6:.2f} E 6")

    max_index = np.argmax(counts)
    cool_vva_max = vva_list[max_index]
    print(f"MOT Cooling VVA resonance: {cool_vva_max}")
    ExpParams.set_parameter("tMOTLoadCoolVVA", cool_vva_max)

    vco_list = np.linspace(0.0, 5.0, 20)
    counts = []
    print(f"Scanning MOT repump VCO...")
    for it, vco in enumerate(vco_list):
        exp.set_iterations(bg_it)
        motcam.setup_acquisition(nframes=bg_it)
        motcam.start_acquisition()
        exp.start_motmaster_experiment({"tMOTLoadRepVCO": vco, "tMOTLoadCoolVCO": ExpParams.get_parameter("tMOTLoadCoolVCO"),
                                        "tMOTLoadRepVVA": ExpParams.get_parameter("tMOTLoadRepVVA"), "tMOTLoadCoolVVA": ExpParams.get_parameter("tMOTLoadCoolVVA"), 
                                        "tImgRepVCO1": ExpParams.get_parameter("rep_vco_resonance"), "tImgCoolVCO1": ExpParams.get_parameter("cool_vco_resonance"), 
                                        "tDelay1": 1, "tMolassesDuration": 1, "tImgCoolVVA1": 4.0, "tImgRepVVA1": 4.0, "tMOTccValue": 0.0, "tDelay2": 1, 
                                        "tTweezerExposure1":100, "tTweezerExposure2":100, "PatternLength": 25000})
        imgs_bg = motcam.acquire_n_frames(nframes=bg_it)

        exp.set_iterations(img_it)
        motcam.setup_acquisition(nframes=img_it)
        motcam.start_acquisition()
        exp.start_motmaster_experiment({"tMOTLoadRepVCO": vco, "tMOTLoadCoolVCO": ExpParams.get_parameter("tMOTLoadCoolVCO"),
                                        "tMOTLoadRepVVA": ExpParams.get_parameter("tMOTLoadRepVVA"), "tMOTLoadCoolVVA": ExpParams.get_parameter("tMOTLoadCoolVVA"), 
                                        "tImgRepVCO1": ExpParams.get_parameter("rep_vco_resonance"), "tImgCoolVCO1": ExpParams.get_parameter("cool_vco_resonance"), 
                                        "tDelay1": 1, "tMolassesDuration": 1, "tImgCoolVVA1": 4.0, "tImgRepVVA1": 4.0, "tDelay2": 1, 
                                        "tTweezerExposure1":100, "tTweezerExposure2":100, "PatternLength": 25000})
        imgs = motcam.acquire_n_frames(nframes=img_it)
        img_mean = imgs.mean(axis=0) - imgs_bg.mean(axis=0)
        total_counts = an.get_total_counts(img_mean, centre_x, centre_y, window_size=50)
        counts.append(total_counts)
        print(f"Iteration {it+1}/{len(vco_list)}  |  MOT Repump VCO = {vco:.3f} V, Total Counts = {total_counts/1e6:.2f} E 6")

    max_index = np.argmax(counts)
    rep_vco_max = vco_list[max_index]
    print(f"MOT Repump VCO resonance: {rep_vco_max}")
    ExpParams.set_parameter("tMOTLoadRepVCO", rep_vco_max)

    vva_list = np.linspace(0.4, 1.0, 20)
    counts = []
    print(f"Scanning MOT repump VVA...")
    for it, vva in enumerate(vva_list):
        exp.set_iterations(bg_it)
        motcam.setup_acquisition(nframes=bg_it)
        motcam.start_acquisition()
        exp.start_motmaster_experiment({"tMOTLoadRepVCO": ExpParams.get_parameter("tMOTLoadRepVCO"), "tMOTLoadCoolVCO": ExpParams.get_parameter("tMOTLoadCoolVCO"),
                                        "tMOTLoadRepVVA": vva, "tMOTLoadCoolVVA": ExpParams.get_parameter("tMOTLoadCoolVVA"),
                                        "tImgRepVCO1": ExpParams.get_parameter("rep_vco_resonance"), "tImgCoolVCO1": ExpParams.get_parameter("cool_vco_resonance"), 
                                        "tDelay1": 1, "tMolassesDuration": 1, "tImgCoolVVA1": 4.0, "tImgRepVVA1": 4.0, "tMOTccValue": 0.0, "tDelay2": 1, 
                                        "tTweezerExposure1":100, "tTweezerExposure2":100, "PatternLength": 25000})
        imgs_bg = motcam.acquire_n_frames(nframes=bg_it)

        exp.set_iterations(img_it)
        motcam.setup_acquisition(nframes=img_it)
        motcam.start_acquisition()
        exp.start_motmaster_experiment({"tMOTLoadRepVCO": ExpParams.get_parameter("tMOTLoadRepVCO"), "tMOTLoadCoolVCO": ExpParams.get_parameter("tMOTLoadCoolVCO"),
                                        "tMOTLoadRepVVA": vva, "tMOTLoadCoolVVA": ExpParams.get_parameter("tMOTLoadCoolVVA"),
                                        "tImgRepVCO1": ExpParams.get_parameter("rep_vco_resonance"), "tImgCoolVCO1": ExpParams.get_parameter("cool_vco_resonance"), 
                                        "tDelay1": 1, "tMolassesDuration": 1, "tImgCoolVVA1": 4.0, "tImgRepVVA1": 4.0, "tDelay2": 1, 
                                        "tTweezerExposure1":100, "tTweezerExposure2":100, "PatternLength": 25000})
        imgs = motcam.acquire_n_frames(nframes=img_it)
        img_mean = imgs.mean(axis=0) - imgs_bg.mean(axis=0)
        total_counts = an.get_total_counts(img_mean, centre_x, centre_y, window_size=50)
        counts.append(total_counts)
        print(f"Iteration {it+1}/{len(vva_list)}  |  MOT Repump VVA = {vva:.3f} V, Total Counts = {total_counts/1e6:.2f} E 6")

    max_index = np.argmax(counts)
    rep_vva_max = vva_list[max_index]
    print(f"MOT Repump VVA resonance: {rep_vva_max}")
    ExpParams.set_parameter("tMOTLoadRepVVA", rep_vva_max)
    ExpParams.save_parameters()


def OptimiseMOTShimParameters(hamcam, exp, n_it_per_exp=20, n_calls=50, init_calls=10, xlim=(0.0, 10.0), ylim=(-5.0, 5.0), zlim=(-1.0, 1.0)):
    dimensions = [  
        Real(name='tMOTShimXccValue', low=xlim[0], high=xlim[1]),
        Real(name='tMOTShimYccValue', low=ylim[0], high=ylim[1]),
        Real(name='tMOTShimZccValue', low=zlim[0], high=zlim[1])
    ]

    # Experiment and camera setup
    n_iterations = n_it_per_exp
    exp.set_motmaster_experiment("RbTweezerBasic2026_2")
    exp.set_iterations(n_iterations)
    hamcam.setup_acquisition("snap", n_iterations * 2)

    @use_named_args(dimensions=dimensions)
    def tweezer_exp_objective(tMOTShimXccValue, tMOTShimYccValue, tMOTShimZccValue):
        hamcam.start_acquisition()
        exp.start_motmaster_experiment({
                "tMOTLoadRepVCO": ExpParams.get_parameter("tMOTLoadRepVCO"), "tMOTLoadCoolVCO": ExpParams.get_parameter("tMOTLoadCoolVCO"),
                "tMOTLoadRepVVA": ExpParams.get_parameter("tMOTLoadRepVVA"), "tMOTLoadCoolVVA": ExpParams.get_parameter("tMOTLoadCoolVVA"),
                "tMolCoolVCO1": ExpParams.get_parameter("tMolCoolVCO1"), "tMolRepVCO1": ExpParams.get_parameter("tMolRepVCO1"), 
                "tMolCoolVVA1": ExpParams.get_parameter("tMolCoolVVA1"), "tMolRepVVA1": ExpParams.get_parameter("tMolRepVVA1"),
                "tMolCoolVCO2": ExpParams.get_parameter("tMolCoolVCO2"), "tMolRepVCO2": ExpParams.get_parameter("tMolRepVCO2"), 
                "tMolCoolVVA2": ExpParams.get_parameter("tMolCoolVVA2"), "tMolRepVVA2": ExpParams.get_parameter("tMolRepVVA2"),
                "tImgCoolVCO1": ExpParams.get_parameter("tImgCoolVCO"), "tImgCoolVVA1": ExpParams.get_parameter("tImgCoolVVA"), 
                "tImgCoolVCO2": ExpParams.get_parameter("tImgCoolVCO"), "tImgCoolVVA2": ExpParams.get_parameter("tImgCoolVVA"),
                "tMOTShimXccValue": tMOTShimXccValue, "tMOTShimYccValue": tMOTShimYccValue, "tMOTShimZccValue": tMOTShimZccValue,
                "tMolassesShimXccValue": ExpParams.get_parameter("tMolassesShimXccValue"), "tMolassesShimYccValue": ExpParams.get_parameter("tMolassesShimYccValue"), "tMolassesShimZccValue": ExpParams.get_parameter("tMolassesShimZccValue"),
                "tMOTLoadingDuration": 10000, "PatternLength": 40000})

        imgs = hamcam.acquire_n_frames(n_iterations * 2)
        imgs1, imgs2 = imgs[::2], imgs[1::2]
        imgs_filtered_1 = [an.morphological_tophat_high_pass(img, feature_size=ExpParams.get_parameter("feature_size")) for img in imgs1]
        _, loading_probabilities, _, fidelity = an.get_array_loading_statistics(imgs_filtered_1, ExpParams.get_parameter("site_positions"), ExpParams.get_parameter("array_shape"), threshold_detection=False, threshold=1.6, window_size=ExpParams.get_parameter("sum_window"), binning=60, verbose=False, show_histogram=False)
        mean_loading_probability = np.mean(loading_probabilities)
        measured_loss = 1.0 - mean_loading_probability

        print(f"Parameters: tMOTShimXccValue = {tMOTShimXccValue:.2f}, tMOTShimYccValue = {tMOTShimYccValue:.2f}, tMOTShimZccValue = {tMOTShimZccValue:.2f}  |  Mean Loading Probability = {mean_loading_probability:.4f}")
        return measured_loss

    print("Starting Bayesian Optimization with dynamic exploration...")
    xi_start = 0.100
    xi_end = 0.001
    opt = Optimizer(
        dimensions=dimensions,
        n_initial_points=init_calls,
        acq_func='EI',
        random_state=50
    )

    for i in range(n_calls):
        current_xi = xi_start - (xi_start - xi_end) * (i / (n_calls - 1))
        opt.acq_func_kwargs = {'xi': current_xi}
        suggested_params = opt.ask()
        loss = tweezer_exp_objective(suggested_params)
        result = opt.tell(suggested_params, loss)
        print(f"Iteration {i+1}/{n_calls} (xi = {current_xi:.4f})")
        print(f"  Current score: {(1-loss):.4f}")
        print(f"  Best score: {(1-result.fun):.4f}\n")

    # --- 4. Analyze the Results ---
    print("\n" + "="*40)
    print("Optimization Complete!")
    print("="*40)
    high_score = 1 - result.fun
    best_params = result.x
    print(f"High score: {high_score:.4f}  |  Best parameters:")
    for dim, value in zip(dimensions, best_params):
        print(f"  {dim.name}: {value:.4f}")
        ExpParams.set_parameter(dim.name, value)
    ExpParams.save_parameters()

def OptimiseMolassesShimParameters(hamcam, exp, n_it_per_exp=20, n_calls=50, init_calls=10, xlim=(0.0, 10.0), ylim=(-5.0, 5.0), zlim=(-1.0, 1.0)):
    dimensions = [  
        Real(name='tMolassesShimXccValue', low=xlim[0], high=xlim[1]),
        Real(name='tMolassesShimYccValue', low=ylim[0], high=ylim[1]),
        Real(name='tMolassesShimZccValue', low=zlim[0], high=zlim[1])
    ]

    # Experiment and camera setup
    n_iterations = n_it_per_exp
    exp.set_motmaster_experiment("RbTweezerBasic2026_2")
    exp.set_iterations(n_iterations)
    hamcam.setup_acquisition("snap", n_iterations * 2)

    @use_named_args(dimensions=dimensions)
    def tweezer_exp_objective(tMolassesShimXccValue, tMolassesShimYccValue, tMolassesShimZccValue):
        # Check loading probability
        hamcam.start_acquisition()
        exp.start_motmaster_experiment({
            "tMOTLoadRepVCO": ExpParams.get_parameter("tMOTLoadRepVCO"), "tMOTLoadCoolVCO": ExpParams.get_parameter("tMOTLoadCoolVCO"),
            "tMOTLoadRepVVA": ExpParams.get_parameter("tMOTLoadRepVVA"), "tMOTLoadCoolVVA": ExpParams.get_parameter("tMOTLoadCoolVVA"),
            "tMolCoolVCO1": ExpParams.get_parameter("tMolCoolVCO1"), "tMolRepVCO1": ExpParams.get_parameter("tMolRepVCO1"),
            "tMolCoolVVA1": ExpParams.get_parameter("tMolCoolVVA1"), "tMolRepVVA1": ExpParams.get_parameter("tMolRepVVA1"),
            "tMolCoolVCO2": ExpParams.get_parameter("tMolCoolVCO2"), "tMolRepVCO2": ExpParams.get_parameter("tMolRepVCO2"),
            "tMolCoolVVA2": ExpParams.get_parameter("tMolCoolVVA2"), "tMolRepVVA2": ExpParams.get_parameter("tMolRepVVA2"),
            "tImgCoolVCO1": ExpParams.get_parameter("tImgCoolVCO"), "tImgCoolVVA1": ExpParams.get_parameter("tImgCoolVVA"),
            "tImgCoolVCO2": ExpParams.get_parameter("tImgCoolVCO"), "tImgCoolVVA2": ExpParams.get_parameter("tImgCoolVVA"),
            "tMOTShimXccValue": ExpParams.get_parameter("tMOTShimXccValue"), "tMOTShimYccValue": ExpParams.get_parameter("tMOTShimYccValue"), "tMOTShimZccValue": ExpParams.get_parameter("tMOTShimZccValue"),
            "tMolassesShimXccValue": tMolassesShimXccValue, "tMolassesShimYccValue": tMolassesShimYccValue, "tMolassesShimZccValue": tMolassesShimZccValue, 
            "BackgroundDrop": 1, "tDropDuration": int(4)})
        imgs = hamcam.acquire_n_frames(n_iterations * 2)
        imgs1, imgs2 = imgs[::2], imgs[1::2]
        imgs_filtered_1 = [an.morphological_tophat_high_pass(img, feature_size=ExpParams.get_parameter("feature_size")) for img in imgs1]
        imgs_filtered_2 = [an.morphological_tophat_high_pass(img, feature_size=ExpParams.get_parameter("feature_size")) for img in imgs2]
        _, loading_probabilities, threshold, _ = an.get_array_loading_statistics(imgs_filtered_1, ExpParams.get_parameter("site_positions"), ExpParams.get_parameter("array_shape"), threshold_detection=True, threshold=ExpParams.get_parameter("threshold"), binning=60, window_size=ExpParams.get_parameter("sum_window"), show_histogram=False, verbose=False)
        mean_loading_probability = loading_probabilities.mean()

        if mean_loading_probability < 0.5:
            measured_loss = 1 - mean_loading_probability
            survival_probability = np.nan
        else:
            survival_probability = an.extract_survival_probability(imgs_filtered_1, imgs_filtered_2, ExpParams.get_parameter("site_positions"), ExpParams.get_parameter("array_shape"), threshold=ExpParams.get_parameter("threshold"), window_size=ExpParams.get_parameter("sum_window"))
            measured_loss = 0.4 - survival_probability

        print(f"tMolassesShimXccValue = {tMolassesShimXccValue:.2f}, tMolassesShimYccValue = {tMolassesShimYccValue:.2f}, tMolassesShimZccValue = {tMolassesShimZccValue:.2f}  |  Loading Probability = {mean_loading_probability:.4f}  |  40 us Survival Probability = {survival_probability:.4f}")
        return measured_loss

    print("Starting Bayesian Optimization with dynamic exploration...")
    xi_start = 0.100
    xi_end = 0.001
    opt = Optimizer(
        dimensions=dimensions,
        n_initial_points=init_calls,
        acq_func='EI',
        random_state=50
    )

    for i in range(n_calls):
        current_xi = xi_start - (xi_start - xi_end) * (i / (n_calls - 1))
        opt.acq_func_kwargs = {'xi': current_xi}
        suggested_params = opt.ask()
        loss = tweezer_exp_objective(suggested_params)
        result = opt.tell(suggested_params, loss)
        print(f"Iteration {i+1}/{n_calls} (xi = {current_xi:.4f})")
        print(f"  Current score: {(1-loss):.4f}")
        print(f"  Best score: {(1-result.fun):.4f}\n")

    # --- 4. Analyze the Results ---
    print("\n" + "="*40)
    print("Optimization Complete!")
    print("="*40)
    high_score = 1 - result.fun
    best_params = result.x
    print(f"High score: {high_score:.4f}  |  Best parameters:")
    for dim, value in zip(dimensions, best_params):
        print(f"  {dim.name}: {value:.4f}")
        ExpParams.set_parameter(dim.name, value)
    ExpParams.save_parameters()
    

    


