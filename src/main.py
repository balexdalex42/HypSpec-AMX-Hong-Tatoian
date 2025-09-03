import sys, gc, logging
gc.enable()

from typing import Union, List
from config import * 

import tqdm
import pandas as pd

import hd_preprocess, hd_cluster, hd_cluster_amx
logger = logging.getLogger('HyperSpec')

import torch
import numpy as np

import cProfile
import pstats

# @profile
def main(args: Union[str, List[str]] = None) -> int:
    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : '
        '{message}', style='{'))
    root.addHandler(handler)
    
    # Disable dependency non-critical log messages.
    logging.getLogger('numpy').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('cupy').setLevel(logging.WARNING)
    logging.getLogger('joblib').setLevel(logging.WARNING)

    # Load the configuration.
    config.parse(args)
    logger.debug('input_filepath= %s', config.input_filepath)
    # logger.debug('work_dir = %s', config.work_dir)
    # logger.debug('overwrite = %s', config.overwrite)
    logger.debug('checkpoint = %s', config.checkpoint)
    logger.debug('representative_mgf = %s', config.representative_mgf)
    logger.debug('cpu_core_preprocess = %s', config.cpu_core_preprocess)
    logger.debug('cpu_core_cluster = %s', config.cpu_core_cluster)
    logger.debug('batch_size = %d', config.batch_size)
    logger.debug('use_gpu_cluster = %s', config.use_gpu_cluster)

    logger.debug('min_peaks = %d', config.min_peaks)
    logger.debug('min_mz_range = %.2f', config.min_mz_range)
    logger.debug('min_mz = %.2f', config.min_mz)
    logger.debug('max_mz = %.2f', config.max_mz)
    logger.debug('remove_precursor_tol = %.2f', config.remove_precursor_tol)
    logger.debug('min_intensity = %.2f', config.min_intensity)
    logger.debug('max_peaks_used = %d', config.max_peaks_used)
    logger.debug('scaling = %s', config.scaling)

    logger.debug('hd_dim = %d', config.hd_dim)
    logger.debug('hd_Q = %d', config.hd_Q)
    logger.debug('hd_id_flip_factor = %.1f', config.hd_id_flip_factor)
    logger.debug('cluster_charges = %s', config.cluster_charges)

    logger.debug('precursor_tol = %.2f %s', *config.precursor_tol)
    logger.debug('rt_tol = %s', config.rt_tol)
    logger.debug('cluster_alg = %s', config.cluster_alg)
    logger.debug('fragment_tol = %.2f', config.fragment_tol)
    logger.debug('eps = %.3f', config.eps)
    logger.debug('amx = %s', config.amx)

    #Setting which version of hd_cluster to use
    hd_cluster_lib = hd_cluster_amx if config.amx else hd_cluster

    #Setting up profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    
    
    # Restore checkpoints
    spectra_meta_df, spectra_hvs = None, None
    if config.checkpoint:
        spectra_meta_df, spectra_hvs = hd_preprocess.load_checkpoint(
            config=config, logger=logger)
    
    if (spectra_meta_df is None) or (spectra_hvs is None):
        ###################### 1. Load and parse spectra files
        spectra_meta_df, spectra_mz, spectra_intensity = hd_preprocess.load_process_spectra_parallel(config=config, logger=logger)
        logger.info("Preserve {} spectra for cluster charges: {}".format(len(spectra_meta_df), config.cluster_charges))

        ###################### 2 HD Encoding for spectra
        spectra_hvs = hd_cluster_lib.encode_spectra(
            spectra_mz=spectra_mz, spectra_intensity=spectra_intensity, config=config, logger=logger)

        # Save meta and encoding data
        if config.checkpoint:
            spectra_hvs_numpy = np.array([tensor.numpy() for tensor in spectra_hvs], dtype=np.float32)
            hd_preprocess.save_checkpoint(
                spectra_meta=spectra_meta_df, spectra_hvs=spectra_hvs_numpy, 
                config=config, logger=logger)


    ###################### 3. Cluster for each charge
    cluster_df = pd.DataFrame()
    for prec_charge_i in tqdm.tqdm(config.cluster_charges):
        # Select spectra with cluster charge
        idx = spectra_meta_df['precursor_charge']==prec_charge_i #we get a series
        spec_df_by_charge = spectra_meta_df.loc[idx]
        print(f"spec_df_by_charge: {spec_df_by_charge}")

        logger.info("Start clustering Charge {} with {} spectra".format(prec_charge_i, len(spec_df_by_charge)))
        # we need to make sure we get row hypervectors we need, good thing we have the indexes spec_df_by_charge
        idx_indices = spec_df_by_charge.index.tolist()
        print(f"idx_indices: {idx_indices}")
        selected_hvs = [spectra_hvs[i] for i in idx_indices]
        cluster_labels_per_charge, cluster_representatives_per_charge = hd_cluster_lib.cluster_spectra(
            spectra_by_charge_df=spec_df_by_charge, encoded_spectra_hv=selected_hvs,
            config=config, logger=logger)

        spec_df_by_charge = spec_df_by_charge.assign(
            cluster=list(cluster_labels_per_charge), 
            is_representative=list(cluster_representatives_per_charge))
        
        cluster_df = pd.concat([cluster_df, spec_df_by_charge])

    profiler.disable()
    
    # Print profiling results
    stats = pstats.Stats(profiler)
    print("=== PROFILING RESULTS ===")
    stats.sort_stats('cumulative').print_stats(20)
    hd_preprocess.export_cluster_results(
        spectra_df=cluster_df, config=config, logger=logger)


if __name__ == "__main__":
    main()

