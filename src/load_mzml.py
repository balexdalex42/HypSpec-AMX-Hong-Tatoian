#from utils import load_mgf_file 
from typing import Dict, IO, Iterator, List, Tuple, Union, Optional
import tqdm
#import hd_preprocess
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
import sys
import cProfile
import pstats
import time
# from multiprocessing import Pool, TimeoutError

temp_charge = 2
temp_mz = 150.0


#logger = logging.getLogger()

def mzxml_load(filename):
    query_filename = filename
    spectra_list = []
    for spectrum in read_mzxml(query_filename):

        spectra_list.append([
                            -1, spectrum.precursor_charge, spectrum.precursor_mz,
                            query_filename, spectrum.identifier, float(spectrum.retention_time * 1000), spectrum.mz,
                            spectrum.intensity])
    
    return spectra_list

    
    # start = time.time()
    # spectra_list = []
    # with mzxml.MzXML(filename) as f_in:
    #     try:
    #         map_start = time.time()

    #         # with Pool(processes = 8) as pool:

    #         #     parsed_spectrum = pool.imap_unordered(_parse_spectrum_mzxml, f_in)
    #         #     spectra_list =  pool.imap_unordered(appending, parsed_spectrum)

    #         parsed_spectrum = map(_parse_spectrum_mzxml, f_in)
    #         spectra_list = map(appending, parsed_spectrum, filename)
    #         map_runtime = time.time() - map_start
    #         print(spectra_list)
    #         # for spectrum in parsed_spectrum:
    #         #     spectra_list.append([
    #         #                 -1, spectrum.precursor_charge, spectrum.precursor_mz,
    #         #                 filename, spectrum.identifier, spectrum.mz,
    #         #                 spectrum.intensity])
    #     except LxmlError as e:
    #         logger.warning('Failed to read file %s: %s', source, e)
    # print("new runtime: %s seconds" % (time.time() - start))
    # print("map runtime: %s seconds" % map_runtime)
    # return spectra_list

def appending(spectrum, filename):
    return [-1, spectrum.precursor_charge, spectrum.precursor_mz,
                            filename, spectrum.identifier, spectrum.mz,
                            spectrum.intensity]

def mzml_load(filename):
    spectra_list = []
    # start = time.time()

    # with mzml.MzML(filename) as f_in:
    #     try:
    #         map_start = time.time()

    #         # with Pool(processes = 8) as pool:

    #         #     parsed_spectrum = pool.imap_unordered(_parse_spectrum_mzxml, f_in)
    #         #     spectra_list =  pool.imap_unordered(appending, parsed_spectrum)

    #         spectra_list = map(_parse_spectrum_mzml, f_in)
    #         map_runtime = time.time() - map_start
    #         for i in spectra_list:
    #             print(i)
    #         # for spectrum in parsed_spectrum:
    #         #     spectra_list.append([
    #         #                 -1, spectrum.precursor_charge, spectrum.precursor_mz,
    #         #                 filename, spectrum.identifier, spectrum.mz,
    #         #                 spectrum.intensity])
    #     except LxmlError as e:
    #         logger.warning('Failed to read file %s: %s', source, e)
    # print("new runtime: %s seconds" % (time.time() - start))
    # print("map runtime: %s seconds" % map_runtime)

    with mzml.MzML(filename) as f_in:
        try:
            spectra_list = list(filter(lambda item: item is not None, map(_parse_spectrum_mzml, f_in, filename)))

        except LxmlError as e:
            logger.warning('Failed to read file %s: %s', source, e)
    # for spectrum in read_mzml(filename):

    #     spectra_list.append(spectrum)
    return spectra_list

def convert_mzxml_mgf(filename):
    query_filename = filename
    print("works")
    #load_process_single("b1906_293T_proteinID_01A_QE3_122212.mgf")
    # fp = open("test.txt", "w")
    fp = open("./test.txt", "w")


    for spectrum in read_mzxml(query_filename):

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




def convert_mzml_mgf(filename):
    query_filename = filename
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
    print("start")

    #TODO 
    #found in documentation need to experiement
    #mzml.MzML.map()
    
    with mzml.MzML(source) as f_in:
        try:
            for i, spectrum in enumerate(f_in):
                try:
                    with cProfile.Profile() as profile:
                        parsed_spectrum = _parse_spectrum_mzml(spectrum)
                            #parsed_spectrum.index = i
                        yield parsed_spectrum
                        results = pstats.Stats(profile)
                        results.sort_stats(pstats.SortKey.TIME)
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
    global temp_charge
    global temp_mz
    if 'scan=' in spectrum_id:
        scan_nr = int(spectrum_id[spectrum_id.find('scan=') + len('scan='):])
    elif 'index=' in spectrum_id:
        scan_nr = int(spectrum_id[spectrum_id.find('index=') + len('index='):])
    else:
        raise ValueError(f'Failed to parse scan/index number')

    
    mz_array = spectrum_dict['m/z array']
    intensity_array = spectrum_dict['intensity array']
    retention_time = spectrum_dict['scanList']['scan'][0]['scan start time']
    try:
        precursor = spectrum_dict['precursorList']['precursor'][0]
        precursor_ion = precursor['selectedIonList']['selectedIon'][0]
        precursor_mz = precursor_ion['selected ion m/z']
        if 'charge state' in precursor_ion:
            precursor_charge = int(precursor_ion['charge state'])
        elif 'possible charge state' in precursor_ion:
            precursor_charge = int(precursor_ion['possible charge state'])
        temp_charge = precursor_charge
        temp_mz = precursor_mz
    except:
        precursor_charge = temp_charge
        precursor_mz = temp_mz


    spectrum = MsmsSpectrum(str(scan_nr), precursor_mz, precursor_charge,
                            mz_array, intensity_array, retention_time)
    return spectrum

def read_mzxml(source: Union[IO, str]) -> Iterator[MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given mzXML file.

    Parameters
    ----------
    source : Union[IO, str]
        The mzXML source (file name or open file object) from which the spectra
        are read.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the requested spectra in the given file.
    """

    with mzxml.MzXML(source) as f_in:
        try:
            parsed_spectrum = map(_parse_spectrum_mzxml, f_in)
            yield parsed_spectrum

        #     for i, spectrum in enumerate(f_in):
        #         if int(spectrum.get('msLevel', -1)) == 2:
        #             try:
        #                 print(i)
        #                 #with cProfile.Profile() as profile:
        #                 parsed_spectrum = _parse_spectrum_mzxml(spectrum)
        #                 parsed_spectrum.index = i
        #                 parsed_spectrum.is_processed = False
        #                 yield parsed_spectrum
        #                 # results = pstats.Stats(profile)
        #                 # results.sort_stats(pstats.SortKey.TIME)
        #                 # results.print_stats()
        #             except ValueError as e:
        #                 logger.warning(f'Failed to read spectrum %s: %s',
        #                                spectrum['id'], e)
        except LxmlError as e:
            logger.warning('Failed to read file %s: %s', source, e)

    '''
    start = time.time()
    with mzxml.MzXML(source) as f_in:
        print("creating dict: %s seconds " % (time.time() - start))
        try:
            for i, spectrum in enumerate(f_in):
                if int(spectrum.get('msLevel', -1)) == 2:
                    try:
                        print(i)
                        #with cProfile.Profile() as profile:
                        parsed_spectrum = _parse_spectrum_mzxml(spectrum)
                        parsed_spectrum.index = i
                        parsed_spectrum.is_processed = False
                        yield parsed_spectrum
                        # results = pstats.Stats(profile)
                        # results.sort_stats(pstats.SortKey.TIME)
                        # results.print_stats()
                    except ValueError as e:
                        logger.warning(f'Failed to read spectrum %s: %s',
                                       spectrum['id'], e)
        except LxmlError as e:
            logger.warning('Failed to read file %s: %s', source, e)
    print("read_mzxml: %s seconds" % (time.time() - start))
    '''

def _parse_spectrum_mzxml(spectrum_dict: Dict) -> MsmsSpectrum:
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
        - Not an MS/MS spectrum.
        - Unknown precursor charge.
    """
    start = time.time()
    scan_nr = int(spectrum_dict['id'])

    if int(spectrum_dict.get('msLevel', -1)) != 2:
        raise ValueError(f'Unsupported MS level {spectrum_dict["msLevel"]}')

    mz_array = spectrum_dict['m/z array']
    intensity_array = spectrum_dict['intensity array']
    retention_time = spectrum_dict['retentionTime']

    precursor_mz = spectrum_dict['precursorMz'][0]['precursorMz']
    if 'precursorCharge' in spectrum_dict['precursorMz'][0]:
        precursor_charge = spectrum_dict['precursorMz'][0]['precursorCharge']
    else:
        precursor_charge = None

    spectrum = MsmsSpectrum(str(scan_nr), precursor_mz, precursor_charge,
                            mz_array, intensity_array, retention_time)


    #print("parsing dict: %s seconds" % (time.time() - start))
    return spectrum




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

if __name__ == "__main__":
    if str(sys.argv[1])[-3] == "X":
        print("mzxml")
        with cProfile.Profile() as profile:
            print(mzxml_load(sys.argv[1]))
        results = pstats.Stats(profile)
        results.sort_stats(pstats.SortKey.TIME)
        #results.print_stats()
    elif str(sys.argv[1])[-3] == "z":
        print("mzml")
        
        spectra = mzml_load(sys.argv[1])

        print(spectra)  
    # with cProfile.Profile() as profile:
    #     mzml_load(sys.argv[1])

    # results = pstats.Stats(profile)
    # results.sort_stats(pstats.SortKey.TIME)
    # results.print_stats()


