#from utils import load_mgf_file 
from typing import Dict, IO, Iterator, List, Tuple, Union, Optional
import tqdm
#from hd_preprocess import fast_mgf_parse, preprocess_read_spectra_list
from spectrum_utils.spectrum import MsmsSpectrum
from pyteomics import mgf, mzml, mzxml, parser
import logging
from lxml.etree import LxmlError
from config import Config
import numba as nb
import numpy as np
import os
from collections import defaultdict
import copy



#logger = logging.getLogger()

def mzml_load:
# read_spectra_list.append([
#                         -1, charge, pepmass, 
#                         filename, scans, rtinsecs, 
#                         vector_to_array(mz, peak_i),
#                         vector_to_array(intensity, peak_i)
#                         ])
    query_filename = "./13_b3t1_Pbt_Fe.mzML"
    spectra_list = []
    for spectrum in read_mzml(query_filename):

        spectra_list.append([
                            -1, spectrum.precursor_charge, spectrum.precursor_mz,
                            query_filename, spectrum.identifier, spectrum.mz,
                            spectrum.intensity])
    
    print(spectra_list)





def convert_mzml_m():
    query_filename = "./13_b3t1_Pbt_Fe.mzML"
    print("works")
    #load_process_single("b1906_293T_proteinID_01A_QE3_122212.mgf")
    # fp = open("test.txt", "w")
    fp = open("./test.txt", "w")


    for spectrum in read_mzml(query_filename):

        # fp.write("BEGIN IONS\n")
        # fp.write("TITLE=temp")  
        # fp.write("SCANS=%s", spectrum.identifier)
        # fp.write("") 

        # print("BEGIN IONS")
        # print("TITLE=not needed")
        # print("SCANS=%s" % spectrum.identifier)
        # print("PEPMAS=%s" % spectrum.precursor_mz)
        # print("RTINSECONDS=%s" % (float(spectrum.retention_time) * 1000))
        # print("CHARGE=%s+" % spectrum.precursor_charge)


        # for i in range(len(spectrum.mz)):
        #     print("%s %s" % (spectrum.mz[i], spectrum.intensity[i]))
        # print("END IONS")


        fp.write("BEGIN IONS\n")
        fp.write("TITLE=not needed\n")
        fp.write("SCANS=%s\n" % spectrum.identifier)
        fp.write("PEPMAS=%s\n" % spectrum.precursor_mz)
        rtn_seconds = float(spectrum.retention_time) * 1000
        fp.write("RTINSECONDS=%f\n" % rtn_seconds)
        fp.write(("CHARGE=%s+\n" % spectrum.precursor_charge))

        for i in range(len(spectrum.mz)):
            fp.write("%s %s\n" % (spectrum.mz[i], spectrum.intensity[i]))

        fp.write("END IONS\n")

    # read_spectra_list = []
    # for i, spectrum in enumerate(mzml.read(query_filename)):
    #     print('\n Scan List: ', spectrum['scanList'])
                
    #     #print(spectrum['m/z array'])

    #     #print(spectrum['intensity arrayu]'])

    #     #print(spectrum['MS1 spectrum'])

    #     print('\n Title:', spectrum['spectrum title'])

    #     #print('\n MS1 spectrum: ', spectrum['MS1 spectrum'])

    #     print('\n Id: ', spectrum['id'])

    #     print('\n Base Peak Intensity: ', spectrum['base peak intensity'])

    #     print('\n Max mz: ', spectrum['highest observed m/z'])

    #     print('\n Min mz: ', spectrum['lowest observed m/z'])

    #     # read_spectra_list.append([-1, ])
    #     return
    fp.close()
    print("done")

    





    





def load_process_single(
    file: str,
    if_preprocess: bool = True,
    min_peaks: int = 5, min_mz_range: float = 250.0,
    mz_interval: int = 1,
    mz_min: Optional[float] = 101.0,
    mz_max: Optional[float] = 1500.,
    remove_precursor_tolerance: Optional[float] = 1.50,
    min_intensity: Optional[float] = 0.01,
    max_peaks_used: Optional[int] = 50,
    scaling: Optional[str] = 'off'
):
    spec_list = fast_mgf_parse(file)
    
    if if_preprocess:
        spec_list = preprocess_read_spectra_list(
            spectra_list = spec_list,
            min_peaks = min_peaks, min_mz_range = min_mz_range,
            mz_interval = mz_interval,
            mz_min = mz_min, mz_max = mz_max,
            remove_precursor_tolerance = remove_precursor_tolerance,
            min_intensity = min_intensity,
            max_peaks_used = max_peaks_used,
            scaling = scaling)

    return spec_list



def read_mzml(source: Union[IO, str]) -> Iterator[MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given mzML file.

    Parameters
    ----------
    source : Union[IO, str]
        The mzML source (file name or open file object) from which the spectra
        are read.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the requested spectra in the given file.
    """
    print("0")
    with mzml.MzML(source) as f_in:
        try:
            for i, spectrum in enumerate(f_in):
                if int(spectrum.get('ms level', -1)) == 2:
                    try:
                        parsed_spectrum = _parse_spectrum_mzml(spectrum)
                        parsed_spectrum.index = i
                        parsed_spectrum.is_processed = False
                        yield parsed_spectrum
                    except ValueError as e:
                        logger.warning(f'Failed to read spectrum %s: %s',
                                       spectrum['id'], e)
        except LxmlError as e:
            logger.warning('Failed to read file %s: %s', source, e)



def _parse_spectrum_mzml(spectrum_dict: Dict) -> MsmsSpectrum:
    """
    Parse the Pyteomics spectrum dict.

    Parameters
    ----------
    spectrum_dict : Dict
        The Pyteomics spectrum dict to be parsed.

    Returns
    -------
    MsmsSpectrum
        The parsed spectrum.

    Raises
    ------
    ValueError: The spectrum can't be parsed correctly:
        - Unknown scan number.
        - Not an MS/MS spectrum.
        - Unknown precursor charge.
    """
    spectrum_id = spectrum_dict['id']

    if 'scan=' in spectrum_id:
        scan_nr = int(spectrum_id[spectrum_id.find('scan=') + len('scan='):])
    elif 'index=' in spectrum_id:
        scan_nr = int(spectrum_id[spectrum_id.find('index=') + len('index='):])
    else:
        raise ValueError(f'Failed to parse scan/index number')

    if int(spectrum_dict.get('ms level', -1)) != 2:
        raise ValueError(f'Unsupported MS level {spectrum_dict["ms level"]}')


    mz_array = spectrum_dict['m/z array']
    intensity_array = spectrum_dict['intensity array']
    retention_time = spectrum_dict['scanList']['scan'][0]['scan start time']

    precursor = spectrum_dict['precursorList']['precursor'][0]
    precursor_ion = precursor['selectedIonList']['selectedIon'][0]
    precursor_mz = precursor_ion['selected ion m/z']
    if 'charge state' in precursor_ion:
        precursor_charge = int(precursor_ion['charge state'])
    elif 'possible charge state' in precursor_ion:
        precursor_charge = int(precursor_ion['possible charge state'])
    else:
        precursor_charge = None

    spectrum = MsmsSpectrum(str(scan_nr), precursor_mz, precursor_charge,
                            mz_array, intensity_array, retention_time)

    return spectrum

def process_spectrum(spectrum: MsmsSpectrum, is_library: bool) -> MsmsSpectrum:
    """
    Process the peaks of the MS/MS spectrum according to the config.

    Parameters
    ----------
    spectrum : MsmsSpectrum
        The spectrum that will be processed.
    is_library : bool
        Flag specifying whether the spectrum is a query spectrum or a library
        spectrum.

    Returns
    -------
    MsmsSpectrum
        The processed spectrum. The spectrum is also changed in-place.
    """
    if_preprocess = True
    min_peaks = 5 
    min_mz_range = 250.0
    mz_interval= 1
    mz_min = 101.0
    mz_max  = 1500.0
    remove_precursor_tolerance = 1.50
    min_intensity = 0.01
    max_peaks_used = 50



    if spectrum.is_processed:
        return spectrum

    spectrum = spectrum.set_mz_range(mz_min, mz_max)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum
    '''
    if config.resolution is not None:
        spectrum = spectrum.round(config.resolution, 'sum')
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            spectrum.is_valid = False
            spectrum.is_processed = True
            return spectrum
            '''
    

    spectrum = spectrum.remove_precursor_peak(
        remove_precursor_tolerance, 'Da', 2)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum
    spectrum = spectrum.filter_intensity(
        min_intensity, (max_peaks_used))
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum
    scaling = 'off'
    if scaling == 'sqrt':
        scaling = 'root'
    if scaling is not None:
        spectrum = spectrum.scale_intensity(
            scaling, max_rank=(max_peaks_used))

    spectrum._intensity = _norm_intensity(spectrum.intensity)

    # Set a flag to indicate that the spectrum has been processed to avoid
    # reprocessing of library spectra for multiple queries.
    spectrum.is_valid = True
    spectrum.is_processed = True

    return spectrum


@nb.njit
def _check_spectrum_valid(spectrum_mz: np.ndarray, min_peaks: int,
                          min_mz_range: float) -> bool:
    """
    Check whether a spectrum is of high enough quality to be used for matching.

    Parameters
    ----------
    spectrum_mz : np.ndarray
        M/z peaks of the spectrum whose quality is checked.
    min_peaks : int
        The minimum number of peaks for a spectrum to be valid.
    min_mz_range : float
        The minimum mass range (m/z difference between the highest and lowest
        peak) for a spectrum to be valid.

    Returns
    -------
    bool
        True if the spectrum has enough peaks covering a wide enough mass
        range, False otherwise.
    """
    return (len(spectrum_mz) >= min_peaks and
            spectrum_mz[-1] - spectrum_mz[0] >= min_mz_range)

@nb.njit
def _norm_intensity(spectrum_intensity: np.ndarray) -> np.ndarray:
    """
    Normalize spectrum peak intensities.

    Parameters
    ----------
    spectrum_intensity : np.ndarray
        The spectrum peak intensities to be normalized.

    Returns
    -------
    np.ndarray
        The normalized peak intensities.
    """
    return spectrum_intensity / np.linalg.norm(spectrum_intensity)

def read_query_file(filename: str) -> Iterator[MsmsSpectrum]:
    """
    Read all spectra from the given mgf, mzml, or mzxml file.

    Parameters
    ----------
    filename: str
        The peak file name from which to read the spectra.

    Returns
    -------
    Iterator[Spectrum]
        An iterator of spectra in the given mgf file.
    """
    verify_extension(['.mgf', '.mzml', '.mzxml'],
                     filename)

    _, ext = os.path.splitext(os.path.basename(filename))


    if ext == '.mzml':
        return read_mzml(filename)

def verify_extension(supported_extensions: List[str], filename: str) -> None:
    """
    Check that the given file name has a supported extension.

    Parameters
    ----------
    supported_extensions : List[str]
        A list of supported file extensions.
    filename : str
        The file name to be checked.

    Raises
    ------
    FileNotFoundError
        If the file name does not have one of the supported extensions.
    """
    _, ext = os.path.splitext(os.path.basename(filename))
    if ext.lower() not in supported_extensions:
        logging.error('Unrecognized file format: %s', filename)
        raise FileNotFoundError(f'Unrecognized file format (supported file '
                                f'formats: {", ".join(supported_extensions)})')
    elif not os.path.isfile(filename):
        logging.error('File not found: %s', filename)
        raise FileNotFoundError(f'File {filename} does not exist')



