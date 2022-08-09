import json
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
from mne.preprocessing import ICA
from plotly.offline import iplot_mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plotly.tools import mpl_to_plotly
from plotly.offline import iplot_mpl
from mne.time_frequency import psd_multitaper
import chart_studio.plotly as py
from plotly import tools
from plotly.graph_objs import Layout, Scatter, Annotation, Annotations, Data, Figure, Marker, Font
from plotly.graph_objs.layout import YAxis,XAxis
import random
from plotly import graph_objects as go  # for data visualisation
from plotly import io as pio
import plotly.express as px
import time
from datetime import datetime
import base64
import gzip
from werkzeug.utils import secure_filename
import jsonpickle
import numpy
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, send_file,redirect,render_template
from joblib import Parallel, delayed
import multiprocessing
import io
import os
import sys
import matplotlib.pyplot as plt
import compute_descriptive_statistics_MDA as cds
import global_variables as gv
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
try:
    import get_data_from_server
except ImportError:
    pass
from collections import namedtuple

from templates.model.model_nifti_to_mesh import *
from templates.model.model_bullseye_parcellation import *
import os
import SimpleITK as sitk
import logging
import pickle
import mne

start_time = time.time()

app = Flask(__name__)
app.config['EEG_UPLOADS']='/home/ntakmil/IntegratedDualAnalysisAproach_MDA-sarah/static/EEG'
id_data_type__categorical = "string"
id_data_type__numerical = "number"
id_data_type__date = "date"

# merged_all = get_data_from_server.get_dataframe_from_server()

# csv_file_name_missing_values = 'clinical_data_imputed.csv'
# merged_all.to_csv(csv_file_name_missing_values, index=False)

# missingness = merged_all.isnull().astype(int).sum()
# print(missingness)
# header = merged_all.head()
# abc = merged_all.head()
# abc['missing count'] = missingness
# csv_file_name_missing_values = 'clinical_data_missingness.csv'
# missingness.to_csv(csv_file_name_missing_values, index=False)


# FALK
# merged_all = pd.read_csv("resources/Repro_FastSurfer_run-01_cleaned.csv", keep_default_na=False, na_values=[""])

# synthetic
print(os.getcwd())
merged_all = pd.read_csv(os.path.join("resources", "test.csv"), keep_default_na=False, na_values=[""]) #synthetic_dates_missingness2_small.csv

#merged_all = pd.read_csv("resources/clinical_data_imputed.csv", keep_default_na=False, na_values=[""])


merged_all = merged_all.loc[:, ~merged_all.columns.duplicated()]  # remove duplicate rows

gv.initial_length_of_data_rows = len(merged_all)

# get all data types
dataTypeSeries = merged_all.dtypes

# remove_randomized_values = False
#
# def remove_randomized_values():
#
#     for col in merged_all.columns:
#
#         rand_percentage = numpy.random.randint(0, 20)
#         rand_percentage_data_rows = int(rand_percentage * 0.01 * len(merged_all))
#         randomized_indexes = numpy.random.randint(0, len(merged_all), size=rand_percentage_data_rows)
#
#         for rand_index in randomized_indexes:
#             merged_all[col][rand_index] = None
#
#
# if remove_randomized_values:
#     remove_randomized_values()

latest_labelmap  = os.path.join("resources", "output", "tmp")
parcellation_colortable_value = 8

def is_number(s):
    try:
        complex(s)  # for int, long, float and complex
    except ValueError:
        return False
    return True


class ColumnElementsClass(object):
    def __init__(self, header, column_id, data_type, col_values, descriptive_statistics):
        self.header = header
        self.id = column_id
        self.data_type = data_type
        self.column_values = col_values
        self.key_datatype_change = False
        self.key_removed_during_data_formatting = []
        self.descriptive_statistics = descriptive_statistics


# extra chars are not valid for json strings
def get_column_label(value):
    value = value.replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace("ß", "ss")

    return value


class DescriptiveStatisticsClass(object):
    def __init__(self, currentcol_descriptive, data_type, column_id):

        current_col_without_nan = [current_val for current_val in currentcol_descriptive if str(current_val) != 'nan']

        column_used = current_col_without_nan
        if gv.include_missing_values:
            column_used = currentcol_descriptive

        stdev = 0
        varnc = 0

        if len(current_col_without_nan) > 2:
            [stdev, varnc] = cds.compute_stdev(current_col_without_nan, data_type)

        self.normalized_values = currentcol_descriptive  # .tolist()
        self.coefficient_of_unalikeability = cds.compute_coefficient_of_unalikeability(column_used,
                                                                                       data_type, column_id)
        self.stDev = stdev
        self.varNC = varnc
        self.number_of_modes = cds.get_number_of_modes(column_used, data_type, column_id)
        self.missing_values_percentage = len(
            [x for x in currentcol_descriptive if (str(x) == 'nan' or str(x) == "None")]) / len(
            currentcol_descriptive)
        self.coefficient_of_unalikeability_deviation = 0
        self.stDev_deviation = 0
        self.varNC_deviation = 0
        self.number_of_modes_deviation = 0
        self.missing_values_percentage_deviation = 0
        self.categories = []
        self.overall_deviation = 0

        if data_type == id_data_type__categorical:
            self.categories = cds.get_categories(currentcol_descriptive)


# this creates the json object for more complex structures
def transform(my_object):
    jsonpickle.enable_fallthrough(False)
    jsonpickle.set_preferred_backend('simplejson')
    jsonpickle.set_encoder_options('simplejson', sort_keys=True, ignore_nan=True)
    return jsonpickle.encode(my_object, unpicklable=False)


class JsonTransformer(object):
    pass


initial_descriptive_statistics = []


# replace extra strings with _
def get_column_id(value):
    value = value.replace(" ", "_").replace(")", "_").replace("(", "_").replace(':', '_') \
        .replace("/", "_").replace("-", "_").replace("[", "_").replace("]", "_") \
        .replace(".", "_").replace("?", "_").replace("!", "_").replace("@", "_").replace("*", "_") \
        .replace("ä", "ae").replace("ü", "ue").replace("ö", "oe").replace("ß", "ss")

    value = "id_" + value

    return value


def normalize_values(current_col_for_normalization, current_data_type, col_id):
    current_col_normalized_in_function = current_col_for_normalization

    if current_data_type == id_data_type__numerical or current_data_type == id_data_type__date:

        min_val = numpy.amin(current_col_for_normalization)
        max_val = numpy.amax(current_col_for_normalization)

        normalized_values = list()

        for index_normalization in range(len(current_col_for_normalization)):
            row = current_col_for_normalization[index_normalization]

            if numpy.isnan(min_val):
                normalized_values.append(row)
            elif min_val == max_val:
                normalized_values.append(row)
            else:
                normalized_values.append((row - min_val) / (max_val - min_val))

        current_col_normalized_in_function = pd.Series(normalized_values)
        current_col_normalized_in_function = current_col_normalized_in_function.rename(col_id)

    return current_col_normalized_in_function


UNIXTIME = datetime.fromtimestamp(0)

def get_timestamp_windows(date, format='%Y-%m-%d'):
    """ Calculate timestamp using start of Gregorian calender as epoch.

        The date parameter can be either a string or a datetime.datetime
        object. Strings will be parsed using the '%Y-%m-%d' format by default
        unless a different one is specfied via the optional format parameter.
    """
    try:
        date = datetime.strptime(date, format)
    except TypeError:
        pass
    return (date - UNIXTIME).total_seconds() + -3600.0  # The timedelta in seconds.


def get_data_initially_formatted(index):
    this_data_type_parallel = id_data_type__numerical
    current_col_parallel = merged_all[index]

    if current_col_parallel.dtype == object:
        test_current_col_numeric_parallel = pd.to_numeric(current_col_parallel, errors='coerce')

        this_data_type_parallel = id_data_type__categorical

        if ~numpy.isnan(test_current_col_numeric_parallel.mean()):
            current_col_parallel = test_current_col_numeric_parallel
            this_data_type_parallel = id_data_type__numerical

        datatype_before = this_data_type_parallel
        for i in range(len(current_col_parallel)):
            number = current_col_parallel[i]

            if str(number) != 'nan' and number is not None and str(number).count('.') == 2:
                date_in_milisec = current_col_parallel[i]

                try:
                    #print(number)
                    #date_in_milisec = datetime.strptime(str(number), "%d.%m.%Y").timestamp() * 1000
                    date_in_milisec = get_timestamp_windows(str(number), "%d.%m.%Y") * 1000
                    this_data_type_parallel = id_data_type__date

                except (ValueError, TypeError):
                    this_data_type_parallel = datatype_before

                current_col_parallel.at[i] = date_in_milisec

            if number is None:
                current_col_parallel.at[i] = numpy.NaN

    if this_data_type_parallel == id_data_type__date:

        current_col_name = current_col_parallel.name

        for date_index in range(len(current_col_parallel)):
            date = current_col_parallel.at[date_index]
            if is_number(date):
                current_col_parallel.at[date_index] = current_col_parallel.at[date_index]
            else:
                current_col_parallel.at[date_index] = numpy.NaN

        current_col_parallel.astype('float64')

    current_col_normalized = list(normalize_values(current_col_parallel, this_data_type_parallel,
                                                   get_column_id(current_col_parallel.name)))

    col_descriptive_statistics = DescriptiveStatisticsClass(current_col_normalized, this_data_type_parallel,
                                                            get_column_id(current_col_parallel.name))
    col_description = ColumnElementsClass(get_column_label(current_col_parallel.name),
                                          get_column_id(current_col_parallel.name),
                                          this_data_type_parallel, current_col_parallel.tolist(),
                                          col_descriptive_statistics)

    return col_description


#[sqrt(i ** 2) for i in range(10)]
#Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
#gv.data_initially_formatted = [get_data_initially_formatted(i) for i in merged_all.columns]
num_cores = multiprocessing.cpu_count()

gv.data_initially_formatted = Parallel(n_jobs=num_cores)(delayed(get_data_initially_formatted)(i) for i in merged_all.columns)

gv.include_missing_values = False
gv.data_initially_formatted_no_missing_values = Parallel(n_jobs=num_cores)(delayed(get_data_initially_formatted)(i) for i in merged_all.columns)

#gv.data_initially_formatted_no_missing_values = [get_data_initially_formatted(i) for i in merged_all.columns]

gv.include_missing_values = True

print("--- %s seconds ---" % (time.time() - start_time))

@app.route('/load_csv/', methods=["POST"])
def main_interface():

    gv.request_data_list = []

    return transform([gv.data_initially_formatted, gv.data_initially_formatted_no_missing_values])


def customStudentDecoder(studentDict):
    return namedtuple('X', studentDict.keys())(*studentDict.values())


@app.route('/update_thresholds/', methods=["POST"])
def update_thresholds():

    request_thresholds = request.get_json()

    gv.coefficient_of_unalikeability_threshold = request_thresholds[0]
    gv.modes_threshold = request_thresholds[1]

    filtered_values = list(request_thresholds[2])

    gv.data_initially_formatted = [cds.update_coeff_unalikeability_modes(i) for i in gv.data_initially_formatted]
    gv.data_initially_formatted_no_missing_values = [cds.update_coeff_unalikeability_modes(i) for i in
                                                     gv.data_initially_formatted_no_missing_values]

    filtered_values = [cds.update_coeff_unalikeability_modes_dict(i) for i in filtered_values]

    filtered_values = [cds.update_coeff_unalikealibiity_modes_deviations(i) for i in filtered_values]

    return transform([gv.data_initially_formatted, gv.data_initially_formatted_no_missing_values, filtered_values])


@app.route('/toggle_include_missing_values/', methods=["POST"])
def toggle_include_missing_values():

    request_data_list = request.get_json()

    gv.include_missing_values = not gv.include_missing_values  # toggle_missing_json.include_missing_values_bool

    return comp_deviations(request_data_list)


@app.route('/compute_deviations_and_get_current_values/', methods=["POST"])
def compute_deviations_and_get_current_values():

    request_data_list = request.get_json()

    return comp_deviations(request_data_list)

def comp_deviation_in_loop (data_initial, request_data_list, data_to_use):
    new_values = list([data_initial.column_values[item_index] for
                       item_index in range(len(data_initial.column_values)) if item_index in request_data_list])

    new_values_normalized = list([data_initial.descriptive_statistics.normalized_values[
                                      item_index] for
                                  item_index in range(len(data_initial.column_values)) if item_index in request_data_list])

    col_descriptive_statistics_new = DescriptiveStatisticsClass(new_values_normalized, data_initial.data_type,
                                                                data_initial.id)
    col_descriptive_statistics_new = cds.get_descriptive_statistics_deviations(col_descriptive_statistics_new,
                                                                               [x for x in
                                                                                data_to_use if
                                                                                x.id == data_initial.id][
                                                                                   0].descriptive_statistics)

    col_description_new = ColumnElementsClass(data_initial.header, data_initial.id,
                                              data_initial.data_type, new_values, col_descriptive_statistics_new)

    return col_description_new


def comp_deviations(request_data_list):
    start_time_deviations = time.time()

    data_initially_formatted_new = []

    data_to_use = gv.data_initially_formatted

    if not gv.include_missing_values:
        data_to_use = gv.data_initially_formatted_no_missing_values

    data_initially_formatted_new = Parallel(n_jobs=num_cores)(delayed(comp_deviation_in_loop)(data_initial, request_data_list, data_to_use) for data_initial in data_to_use)

    print("--- %s seconds ---" % (time.time() - start_time_deviations))

    return jsonify(transform([data_initially_formatted_new, gv.columns_not_contributing]))
from plotly.offline import plot
from plotly.graph_objs import Scatter
from flask import Markup


@app.route('/', methods=["GET", "POST"])
def hello():
    return "Hello World!"


@app.after_request
def add_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response

def preprocess_brain_mesh():
    global flairImage
    global flairArray
    global defaultDir
    defaultDir = os.path.join("resources", "input", "default")
    if not os.path.exists(os.path.join(defaultDir, 'CerebrA_brain_Scaled.nii.gz')):
        print("default brain does not exist")
        return jsonify([])

    # load NIFTI volume data
    logging.debug("Process Volume Data")
    flairImage = sitk.ReadImage(os.path.join(defaultDir, 'CerebrA_brain_Scaled.nii.gz'))
    flairArray = sitk.GetArrayFromImage(flairImage)
    if not os.path.exists(os.path.join(defaultDir, 'brain.obj')):
        create_obj_brain(sitk.GetArrayFromImage(flairImage), defaultDir, 0.5)

def get_parcellation_meshes():
    filenames = []
    if not len([x for x in os.listdir(defaultDir) if x.endswith(".obj") and x.startswith("parcellation")]) == 36:
        image = sitk.ReadImage(os.path.join("resources", "input", "bullseye", "bullseye_wmparc.nii.gz"))
        arr = sitk.GetArrayFromImage(image)
        filenames.extend(createParcellationMeshes(arr))
    else:
        filenames.extend([x for x in os.listdir(defaultDir) if x.endswith(".obj") and x.startswith("parcellation")])
    return filenames

@app.route('/get_static_meshes/', methods=["POST"])
def get_static_meshes():
    return jsonify(get_parcellation_meshes()+["brain.obj"])


@app.route('/create_meshes_of_patient/<string:patientname>', methods=["POST"])
def create_meshes_of_patient(patientname):
    inputDir = os.path.join('resources', 'input', 'patients', patientname)
    outputDir = os.path.join('resources', 'output', patientname)
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    volume_size = flairArray.shape
    filenames = []

    # Load NIFTI labelmap
    logging.debug("Process Labelmap Data")
    if not os.path.exists(os.path.join(outputDir, 'combinedcolortable.txt')):
        wmh_filename = get_lesion_mask_of_patient(patientname, "wmh")
        if wmh_filename:
            wmhImage = sitk.ReadImage(wmh_filename)
            wmh_mat = sitk.GetArrayFromImage(wmhImage)
            wmh_mat = (wmh_mat > 0) * 1
            filenames.extend(create_obj_lesions(wmh_mat, outputDir, "wmh"))
        else:
            wmh_mat = np.zeros(volume_size)

        cmb_filename = get_lesion_mask_of_patient(patientname, "cmb")
        if cmb_filename:
            cmbImage = sitk.ReadImage(cmb_filename)
            cmb_mat = sitk.GetArrayFromImage(cmbImage)
            cmb_mat = (cmb_mat > 0) * 1
            filenames.extend(create_obj_lesions(cmb_mat, outputDir, "cmb"))
        else:
            cmb_mat = np.zeros(volume_size)

        epvs_filename = get_lesion_mask_of_patient(patientname, "epvs")
        if epvs_filename:
            epvsImage = sitk.ReadImage(epvs_filename)
            epvs_mat = sitk.GetArrayFromImage(epvsImage)
            epvs_mat = (epvs_mat > 0) * 1
            filenames.extend(create_obj_lesions(epvs_mat, outputDir, "epvs")) # "write_spheres_file" or "create_obj_lesions()"
        else:
            epvs_mat = np.zeros(volume_size)

        combined_labelmap, _, colortable = combine_labelmaps(wmh_mat, cmb_mat, epvs_mat, flairImage, outputDir)
        filenames.extend([colortable])
    else:
        wmh_filename = get_lesion_mask_of_patient(patientname, "wmh")
        if wmh_filename:
            wmhImage = sitk.ReadImage(wmh_filename)
            wmh_mat = sitk.GetArrayFromImage(wmhImage)
            wmh_mat = (wmh_mat > 0) * 1
        else:
            wmh_mat = np.zeros(volume_size)

        cmb_filename = get_lesion_mask_of_patient(patientname, "cmb")
        if cmb_filename:
            cmbImage = sitk.ReadImage(cmb_filename)
            cmb_mat = sitk.GetArrayFromImage(cmbImage)
            cmb_mat = (cmb_mat > 0) * 1
        else:
            cmb_mat = np.zeros(volume_size)

        epvs_filename = get_lesion_mask_of_patient(patientname, "epvs")
        if epvs_filename:
            epvsImage = sitk.ReadImage(epvs_filename)
            epvs_mat = sitk.GetArrayFromImage(epvsImage)
            epvs_mat = (epvs_mat > 0) * 1
        else:
            epvs_mat = np.zeros(volume_size)

        combined_labelmap, _, colortable = combine_labelmaps(wmh_mat, cmb_mat, epvs_mat, flairImage, outputDir)
        #
        filenames.extend(["combinedcolortable.txt"])
        filenames.extend([x for x in os.listdir(outputDir) if x.startswith("multiple") and x.endswith(".obj") and not "epvs" in x])
        filenames.extend([x for x in os.listdir(outputDir) if x.endswith(".spheres")])
        if os.path.exists(os.path.join(outputDir,"multiple_wmh_lesiondata.csv")):
            filenames.extend(["multiple_wmh_lesiondata.csv"])
        if os.path.exists(os.path.join(outputDir,"multiple_cmb_lesiondata.csv")):
            filenames.extend(["multiple_cmb_lesiondata.csv"])
        if os.path.exists(os.path.join(outputDir,"multiple_epvs_lesiondata.csv")):
            filenames.extend(["multiple_epvs_lesiondata.csv"])

    """wmhImageAdd = sitk.ReadImage(os.path.join('output', 'test.nii.gz'))
    sub_wmh(sitk.GetArrayFromImage(wmhImage1), sitk.GetArrayFromImage(wmhImageAdd), wmhImage1, outputDir)"""

    return jsonify(filenames)

@app.route('/get_mesh_file/<string:patientname>/<string:filename>', methods=["POST"])
def get_mesh_file(patientname,filename):
    logging.debug("get mesh file",filename)
    if filename.startswith("parcellation") or filename == "brain.obj":
        return send_from_directory(os.path.join('resources', 'input', 'default'), filename)
    else:
        return send_from_directory(os.path.join('resources', 'output', patientname), filename)

@app.route('/get_volume_of_patient/<string:patientname>', methods=["POST"])
def get_volume_of_patient(patientname):
    #return send_from_directory(os.path.join("resources", "input", "default"), "mni_icbm152_t1_tal_nlin_asym_09c_Scaled.nii.gz")
    logging.debug("get volume file", patientname)
    for lesiontype in ["wmh", "cmb", "epvs"]:
        if os.path.exists(os.path.join("resources", "input", "patients", patientname, lesiontype)):
            #file_candidates = [file for file in os.listdir(os.path.join("resources", "input", "patients", patientname, lesiontype)) if file.endswith("Warped_Scaled.nii.gz")]
            file_candidates = [file for file in os.listdir(os.path.join("resources", "input", "patients", patientname, lesiontype)) if file.endswith(".nii.gz")]
            if len(file_candidates) > 0:
                return send_from_directory(os.path.join("resources", "input", "patients", patientname, lesiontype), file_candidates[0])
    return send_from_directory(os.path.join('resources', 'input', 'patients', patientname), 'FLAIR.nii.gz')
@app.route('/get_maxeegsignal_of_patient/<string:patientname>', methods=["POST","GET"])
def get_maxeegsignal_of_patient(patientname):
    l=0
    logging.debug("get maxeeg file", patientname)
    if os.path.exists(os.path.join("resources", "input", "patients", patientname, 'EEG')):
        file_candidates = [file for file in os.listdir(os.path.join("resources", "input", "patients", patientname, 'EEG')) if file.endswith(".set")]
        raw = mne.io.read_raw_eeglab(os.path.join("resources", "input", "patients", patientname, 'EEG',file_candidates[0]))
        l=str(len(raw.times)/raw.info['sfreq'])
    return l
@app.route('/get_maxeegfreq_of_patient/<string:patientname>', methods=["POST","GET"])
def get_maxeegfreq_of_patient(patientname):
    l=0
    logging.debug("get maxeegfreq file", patientname)
    if os.path.exists(os.path.join("resources", "input", "patients", patientname, 'EEG')):
        file_candidates = [file for file in os.listdir(os.path.join("resources", "input", "patients", patientname, 'EEG')) if file.endswith(".set")]
        raw = mne.io.read_raw_eeglab(os.path.join("resources", "input", "patients", patientname, 'EEG',file_candidates[0]))
        l=str(raw.info['lowpass'])
    return l
@app.route('/get_eegsignal_of_patient/<string:patientname>/<int:value1>/<int:value2>', methods=["POST","GET"])
def get_eegsignal_of_patient(patientname,value1,value2):
    print(value1,value2)
    logging.debug("get eeg file", patientname)
    if os.path.exists(os.path.join("resources", "input", "patients", patientname, 'EEG')):
        file_candidates = [file for file in os.listdir(os.path.join("resources", "input", "patients", patientname, 'EEG')) if file.endswith(".set")]
        raw = mne.io.read_raw_eeglab(os.path.join("resources", "input", "patients", patientname, 'EEG',file_candidates[0]))
        #raw = mne.io.read_raw_brainvision(os.path.join("resources", "input", "patients", patientname, 'EEG',file_candidates[0]), preload=True, verbose=False)
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        #start, stop = raw.time_as_index([0, len(raw.times)])
        start= value1*raw.info['sfreq']
        stop = value2*raw.info['sfreq']
        n_channels = len(raw.info['ch_names'])
        n_channels=15
        data, times = raw[picks[:n_channels], start:stop]
        ch_names = [raw.info['ch_names'][p] for p in picks[:n_channels]]
        step = 1. /n_channels # 5 faghat baraze halati ke 10 taste
        kwargs = dict(domain = [1 - step, 1],showticklabels=False, zeroline=True, showgrid=False)

        # create objects for layout and traces
        layout = Layout(yaxis=YAxis(kwargs),showlegend=True,title='EEG signal from '+str(value1)+' to '+ str(value2)+' s ')
        layout.update(xaxis=XAxis(
            dict(showgrid=True, showticklabels=True, tickmode='array', tickvals=[i for i in range(1, len(times), 10)]
                 , ticktext=[str(i) for i in range(1, len(times), 10)])))
        traces = [Scatter(x=times, y=data.T[:, 0],name=ch_names[0])]
        # loop over the channels
        for ii in range(1, n_channels ):
            kwargs.update(domain=[1 - (ii + 1) * step, 1 - ii * step])
            layout.update({'yaxis%d' % (ii + 1): YAxis(kwargs), 'showlegend': True})
            traces.append(Scatter(x=times, y=data.T[:, ii], yaxis='y%d' % (ii + 1),name=ch_names[ii]))

        # add channel names using Annotations
        annotations = Annotations([Annotation(x=-0.06, y=0, xref='paper', yref='y%d' % (ii + 1),
                                              text=ch_name, font=Font(size=9), showarrow=False)
                                   for ii, ch_name in enumerate(ch_names)])
        layout.update(annotations=annotations)
        # set the size of the figure and plot it
        layout.update(autosize=False, width=1600, height=2000)
        #layout.update(hovermode="x")
        config = dict({'scrollZoom': True,'displayModeBar': True})
        #fig = Figure({'data':traces, 'layout':layout})
        fig=dict({
            'data':traces,
            'layout': layout,
            'labels' : {channel for channel in raw.info['ch_names']},

        })
       # fig.update_layout(title_text= < VALUE >)
        html = plotly.offline.plot(
            {"data": traces,
            "layout": layout,
            },

            output_type="div",
            show_link=False,
            config=config)
        #return plotly.io.to_html(fig, config=None, auto_play=True, include_plotlyjs=True, include_mathjax=False,
                          #post_script=None, full_html=True, animation_opts=None, default_width='100%',
                          #default_height='100%', validate=True, div_id='chart2')
        #graphJSON = json.dumps(html, cls=plotly.utils.PlotlyJSONEncoder)
        plot_fig(fig)
        return html
        #return jsonify({"meshfilenames": "Hi"})


def plot_fig(fig):
    #return render_template('notdash.html', graphJSON=data)
    #fig.show()
    pio.show(fig)

@app.route('/get_eegpsd_of_patient/<string:patientname>/<int:value1>/<int:value2>', methods=["POST","GET"])
def get_eegpsd_of_patient(patientname,value1,value2):
    logging.debug("get eeg file", patientname)
    if os.path.exists(os.path.join("resources", "input", "patients", patientname, 'EEG')):
        file_candidates = [file for file in os.listdir(os.path.join("resources", "input", "patients", patientname, 'EEG')) if file.endswith(".set")]
        raw = mne.io.read_raw_eeglab(
            os.path.join("resources", "input", "patients", patientname, 'EEG', file_candidates[0]))    #if request.method == 'POST':
        raw_data = raw.get_data()
        tmin, tmax = 0, raw_data.shape[1]
        fmin, fmax = value1, value2
        picks = mne.pick_types(raw.info, eeg=True, eog=False,
                               stim=False, exclude='bads')
        psds, freqs = psd_multitaper(raw, low_bias=True, tmin=tmin, tmax=tmax,
                                     fmin=fmin, fmax=fmax, proj=True, picks=picks,
                                     n_jobs=1)
        fig=go.Figure()
        n_channels = len(raw.info['ch_names'])
        for i in range(1,n_channels):
            fig.add_trace(go.Scatter(x=freqs, y=psds[:,i]))
        #return graphJSON
        html = plotly.offline.plot(
            fig,
            output_type="div",
            show_link=False)
        plot_fig(fig)
        return html

@app.route('/get_topoplot_of_patient/<string:patientname>/<int:value1>', methods=["POST","GET"])
def get_topoplot_of_patient(patientname,value1):
    logging.debug("get eeg file", patientname)
    if os.path.exists(os.path.join("resources", "input", "patients", patientname, 'EEG')):
        file_candidates = [file for file in os.listdir(os.path.join("resources", "input", "patients", patientname, 'EEG')) if file.endswith(".set")]
        raw = mne.io.read_raw_eeglab(
            os.path.join("resources", "input", "patients", patientname, 'EEG', file_candidates[0]))    #if request.method == 'POST':

    data=raw.get_data()
    data_evoked = mne.EvokedArray(data, raw.info)
    data_evoked.set_montage(raw.get_montage())
    montage_head = data_evoked.get_montage()
    ch_pos = montage_head.get_positions()['ch_pos']
    pos = np.stack([ch_pos[ch] for ch in raw.ch_names])
    radius = np.abs(pos[[2, 3], 0]).mean()
    mean_data=np.mean(data_evoked.data, axis=1)
    n=value1
    ica = ICA(n_components=n, max_iter='auto', random_state=97)
    ica.fit(raw)
    ica_fig = ica.plot_components(show=False)
    if n>20 :
        fig = make_subplots(cols=2,subplot_titles='Components from 1 to :'+str(value1))
    #mne.viz.plot_topomap(mean_data, data_evoked.info,axes=ax,show=False)
    #canvas = FigureCanvasTkAgg(fig)
    #fig=data_evoked.plot_topomap(ch_type='eeg')

    #ica_fig[0].savefig('ica.png')
    #img = cv2.imread(ica_fig[0])
    #img = cv2.imread('ica.png')
        img1 = np.frombuffer(ica_fig[0].canvas.tostring_rgb(), dtype=np.uint8)
        img2 = np.frombuffer(ica_fig[1].canvas.tostring_rgb(), dtype=np.uint8)
        img1= img1.reshape(ica_fig[0].canvas.get_width_height()[::-1] + (3,))
        img2 = img2.reshape(ica_fig[1].canvas.get_width_height()[::-1] + (3,))
        fig.add_trace(go.Image(z=img1),1,1)
        #fig,ax= plt.subplots(1,2)
        fig.add_trace(go.Image(z=img2),1,2)
    #plotly_fig = mpl_to_plotly(ica_fig[0].canvas)
    #canvas = FigureCanvas(fig)
    #canvas.print_figure('test')
    #plotly_fig = mpl_to_plotly(fig)
    else:
        img = np.frombuffer(ica_fig[0].canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(ica_fig[0].canvas.get_width_height()[::-1] + (3,))
        fig = px.imshow(img)

    html = plotly.offline.plot(
        fig,
        output_type="div",
        show_link=False)
    plot_fig(fig)
    return html
    #py.iplot_mpl(plt.gcf())
    #canvas = FigureCanvas(fig)
    #plotly_fig = tls.mpl_to_plotly(fig)
    #return iplot_mpl(plotly_fig)


@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    epochs = mne.io.read_epochs_eeglab('sampleEEGdata.mat').crop(-0.2,1)
    values = epochs.to_data_frame()
    values = values.groupby("time").mean()
    values.head()
    fig = go.Figure(layout=dict(xaxis=dict(title='time'), yaxis=dict(title='voltage')))
    for ch in epochs.info['ch_names']:
        fig.add_scatter(x=epochs.times, y=values[ch], name=ch)
    return fig




@app.route('/127.0.0.1:5000/upload_eeg_of_patient/', methods=["POST","GET"])
def upload_eeg():
    #return send_from_directory(os.path.join("resources", "input", "default"), "mni_icbm152_t1_tal_nlin_asym_09c_Scaled.nii.gz")
    #logging.debug("upload eeg file", patientname)
    if request.method == 'POST':
        if request.files:
            EEGfile=request.files['EEGfile']
            if EEGfile.filename=='':
                print('filename is invalid')
                return redirect(request.url)
            print(EEGfile)
            return redirect(request.url)
            EEGfilename=secure_filename(EEGfile.filename)
            basedir=os.path.abspath(os.path.dirname(__file__))
            EEG.save(os.path.join(basedir,app.config['EEG_UPLOADS'],EEGfilename))
            return render_template('templates/index.html')

    #for datatype in ["EEG"]:
        #if os.path.exists(os.path.join("resources", "input", "patients", patientname, datatype)):
            #file_candidates = [file for file in os.listdir(os.path.join("resources", "input", "patients", patientname, datatype)) if file.endswith("output_egg_results.vhdr")]

            #if len(file_candidates) > 0:
                #return send_from_directory(os.path.join("resources", "input", "patients", patientname, datatype), file_candidates[0])
    #return send_from_directory(os.path.join('resources', 'input', 'patients', patientname), 'FLAIR.nii.gz')

@app.route('/get_labelmap_of_patient/<string:patientname>/<string:lesiontype>', methods=["POST"])
def get_labelmap_of_patient(patientname, lesiontype):
    global latest_labelmap
    logging.debug("get labelmap file", patientname, lesiontype)
    if patientname == "tmp":
        latest_labelmap  = os.path.join('resources', 'output', patientname)
        return send_from_directory(os.path.join('resources', 'output', patientname), f'{lesiontype}.nii.gz')
    if lesiontype.startswith("combined"):
        latest_labelmap = os.path.join('resources', 'output', patientname)
        return send_from_directory(latest_labelmap, f'{lesiontype}.nii.gz')
    latest_labelmap = os.path.join('resources', 'input', patientname)
    return send_from_directory(os.path.join('resources', 'input', patientname), f'{lesiontype}.nii.gz')

@app.route('/get_latest_parcellation/<string:lesiontype>', methods=["POST"])
def get_latest_parcellation(lesiontype):
    global latest_labelmap
    logging.debug("get parcellation file", lesiontype)
    print(latest_labelmap)
    return send_from_directory(latest_labelmap, f'{lesiontype}.nii.gz')

def preprocess_bullseyes():
    patients = [name for name in os.listdir(os.path.join("resources", "input", "patients")) if os.path.isdir(os.path.join("resources", "input", "patients", name))]
    for patient in patients:
        for lesiontype, absolute in zip(["wmh", "cmb", "epvs"], [False, True, True]):
            if not os.path.exists(os.path.join("resources", "output", patient, "bullseyedata_"+lesiontype+".txt")):
                filename = get_lesion_mask_of_patient(patient, lesiontype)
                if filename and os.path.exists(os.path.join("resources", "input", "bullseye", "bullseye_wmparc.nii.gz")):
                    print("preprocessed bullseyes of patient ", patient, " of lesiontype ", lesiontype)
                    image = sitk.ReadImage(os.path.join("resources", "input", "bullseye", "bullseye_wmparc.nii.gz"))
                    arr_bullseye = sitk.GetArrayFromImage(image)
                    image_lesion = sitk.ReadImage(filename)
                    arr_lesion = sitk.GetArrayFromImage(image_lesion)
                    bullseye_data = mapToBullseye(arr_bullseye, arr_lesion, absolute)
                    if not os.path.exists(os.path.join("resources", "output", patient)):
                        os.makedirs(os.path.join("resources", "output", patient))
                    with open(os.path.join("resources", "output", patient, "bullseyedata_"+lesiontype+".txt"), "wb") as f:
                        pickle.dump(bullseye_data, f)

def get_lesion_mask_of_patient(patientname, lesiontype):
    if os.path.exists(os.path.join("resources", "input", "patients", patientname, lesiontype)):
        file_candidates = [file for file in os.listdir(os.path.join("resources", "input", "patients", patientname, lesiontype)) if file.endswith("mask_Scaled.nii.gz")]
        if len(file_candidates) > 0:
            return os.path.join("resources", "input", "patients", patientname, lesiontype, file_candidates[0])
    return None

@app.route('/get_patients/', methods=["POST"])
def get_patients():
    patients = [name for name in os.listdir(os.path.join("resources", "input", "patients")) if os.path.isdir(os.path.join("resources", "input", "patients", name))]
    logging.debug(patients)
    return jsonify(patients)

@app.route('/add_patient_labelmaps/', methods=["POST"])
def add_patient_labelmaps():
    wmhImages = []
    cmbImages = []
    epvsImages = []
    missing_wmh = set()
    missing_cmb = set()
    missing_epvs = set()

    originalImage = None

    patients = request.get_json()["message"]
    logging.debug(patients)
    if len(patients) == 0:
        return
    if not os.path.exists(os.path.join('resources', 'output', 'tmp')):
        os.mkdir(os.path.join('resources', 'output', 'tmp'))
    print(patients)
    for patient in patients:
        patient = str(patient)

        wmh_filename = get_lesion_mask_of_patient(patient, "wmh")
        if wmh_filename:
            patientImageWMH = sitk.ReadImage(wmh_filename)
            wmhImages.append(sitk.GetArrayFromImage(patientImageWMH))
            originalImage = patientImageWMH
        else:
            missing_wmh.add(patient)

        cmb_filename = get_lesion_mask_of_patient(patient, "cmb")
        if cmb_filename:
            patientImageCMB = sitk.ReadImage(cmb_filename)
            cmbImages.append(sitk.GetArrayFromImage(patientImageCMB))
            originalImage = patientImageCMB
        else:
            missing_cmb.add(patient)

        epvs_filename = get_lesion_mask_of_patient(patient, "epvs")
        if epvs_filename:
            patientImageEPVS = sitk.ReadImage(epvs_filename)
            epvsImages.append(sitk.GetArrayFromImage(patientImageEPVS))
            originalImage = patientImageEPVS
        else:
            missing_epvs.add(patient)

    imageShape = sitk.GetArrayFromImage(originalImage).shape
    allMeshes = []
    if len(wmhImages) != 0:
        [_, wmh_mat, mesh_files] = add_wmh(wmhImages, patientImageWMH, os.path.join('resources', 'output', 'tmp'), "wmh")
        allMeshes.extend(mesh_files)
    else:
        wmh_mat = np.zeros(imageShape)
    if len(cmbImages) != 0:
        [_, cmb_mat, mesh_files] = add_wmh(cmbImages, patientImageCMB, os.path.join('resources', 'output', 'tmp'), "cmb")
        allMeshes.extend(mesh_files)
    else:
        cmb_mat = np.zeros(imageShape)
    if len(epvsImages) != 0:
        [_, epvs_mat, mesh_files] = add_wmh(epvsImages, patientImageEPVS, os.path.join('resources', 'output', 'tmp'), "epvs")
        allMeshes.extend(mesh_files)
    else:
        epvs_mat = np.zeros(imageShape)

    if originalImage is not None:
        combined_labelmap,_,colortable = combine_labelmaps(wmh_mat,cmb_mat,epvs_mat,originalImage,os.path.join('resources', 'output','tmp'))
        allMeshes.append(colortable)

    return jsonify({"meshfilenames": allMeshes, "missing_wmh": list(missing_wmh), "missing_cmb": list(missing_cmb), "missing_epvs": list(missing_epvs)})

@app.route('/sub_patient_labelmaps/', methods=["POST"])
def subPatientLabelmaps():
    wmhImages = []
    cmbImages = []
    epvsImages = []
    missing_wmh = set()
    missing_cmb = set()
    missing_epvs = set()

    originalImage = None

    patients1 = request.get_json()["IDs1"]
    patients2 = request.get_json()["IDs2"]

    if len(patients1) == 0 and len(patients2) == 0:
        return
    if not os.path.exists(os.path.join('resources', 'output', 'tmp')):
        os.mkdir(os.path.join('resources', 'output', 'tmp'))
    for factor, patients in zip([-1, 1], [patients1, patients2]):
        for patient in patients:
            patient = str(patient)

            wmh_filename = get_lesion_mask_of_patient(patient, "wmh")
            if wmh_filename:
                patientImageWMH = sitk.ReadImage(wmh_filename)
                wmhImages.append(sitk.GetArrayFromImage(patientImageWMH)*factor)
                originalImage = patientImageWMH
            else:
                missing_wmh.add(patient)

            cmb_filename = get_lesion_mask_of_patient(patient, "cmb")
            if cmb_filename:
                patientImageCMB = sitk.ReadImage(cmb_filename)
                cmbImages.append(sitk.GetArrayFromImage(patientImageCMB)*factor)
                originalImage = patientImageCMB
            else:
                missing_cmb.add(patient)

            epvs_filename = get_lesion_mask_of_patient(patient, "epvs")
            if epvs_filename:
                patientImageEPVS = sitk.ReadImage(epvs_filename)
                epvsImages.append(sitk.GetArrayFromImage(patientImageEPVS)*factor)
                originalImage = patientImageEPVS
            else:
                missing_epvs.add(patient)

    imageShape = sitk.GetArrayFromImage(originalImage).shape
    allMeshes = []
    if len(wmhImages) != 0:
        [_, wmh_mat, mesh_files] = add_wmh(wmhImages,patientImageWMH, os.path.join('resources', 'output', 'tmp'), "wmh")
        allMeshes.extend(mesh_files)
    else:
        wmh_mat = np.zeros(imageShape)
    if len(cmbImages) != 0:
        [_, cmb_mat, mesh_files] = add_wmh(cmbImages,patientImageCMB, os.path.join('resources', 'output', 'tmp'), "cmb")
        allMeshes.extend(mesh_files)
    else:
        cmb_mat = np.zeros(imageShape)
    if len(epvsImages) != 0:
        [_, epvs_mat, mesh_files] = add_wmh(epvsImages,patientImageEPVS, os.path.join('resources', 'output', 'tmp'), "epvs")
        allMeshes.extend(mesh_files)
    else:
        epvs_mat = np.zeros(imageShape)

    if originalImage is not None:
        combined_labelmap,_,colortable = combine_labelmaps(wmh_mat,cmb_mat,epvs_mat,originalImage,os.path.join('resources', 'output', 'tmp'), colormapType="diverging")
        allMeshes.append(colortable)

    return jsonify({"meshfilenames": allMeshes, "missing_wmh": list(missing_wmh), "missing_cmb": list(missing_cmb), "missing_epvs": list(missing_epvs)})

@app.route('/get_bullseye/<string:patientname>', methods=["POST"])
def getBullseye(patientname):
    bullseye_data = dict()
    for filetype in ["wmh", "cmb", "epvs"]:
        if os.path.exists(os.path.join("resources", "output", patientname, "bullseyedata_"+filetype+".txt")):
            with open(os.path.join("resources", "output", patientname, "bullseyedata_"+filetype+".txt"), "rb") as f:
                bullseye_data_filetype = pickle.load(f)
                bullseye_data[filetype] = bullseye_data_filetype
        else:
            bullseye_data[filetype] = []
    return jsonify(bullseye_data)

@app.route('/add_bullseye/', methods=["POST"])
def addBullseye():
    results = dict()
    patients = request.get_json()["message"]
    missing_wmh = set()
    missing_cmb = set()
    missing_epvs = set()

    for filetype in ["wmh", "cmb", "epvs"]:
        bullseye_data = []
        for patientname in patients:
            patientname = str(patientname)
            if os.path.exists(os.path.join("resources", "output", patientname, "bullseyedata_"+filetype+".txt")):
                with open(os.path.join("resources", "output", patientname, "bullseyedata_"+filetype+".txt"), "rb") as f:
                    bullseye_data_filetype = pickle.load(f)
                    bullseye_data.append(bullseye_data_filetype)
            else:
                if filetype == "wmh":
                    missing_wmh.add(patientname)
                if filetype == "cmb":
                    missing_cmb.add(patientname)
                if filetype == "epvs":
                    missing_epvs.add(patientname)

        if len(bullseye_data) > 0:
            results[filetype] = addBullseyeData(bullseye_data)
        else:
            results[filetype] = []
    print(results)
    return jsonify({"bullseyedata": results, "missing_wmh": list(missing_wmh), "missing_cmb": list(missing_cmb), "missing_epvs": list(missing_epvs)})

@app.route('/sub_bullseye/', methods=["POST"])
def subBullseye():
    results = dict()
    patients1 = request.get_json()["IDs1"]
    patients2 = request.get_json()["IDs2"]
    missing_wmh = set()
    missing_cmb = set()
    missing_epvs = set()

    for filetype in ["wmh", "cmb", "epvs"]:
        bullseye_data_patients1 = []
        for patientname in patients1:
            patientname = str(patientname)
            if os.path.exists(os.path.join("resources", "output", patientname, "bullseyedata_"+filetype+".txt")):
                with open(os.path.join("resources", "output", patientname, "bullseyedata_"+filetype+".txt"), "rb") as f:
                    bullseye_data_filetype = pickle.load(f)
                    bullseye_data_patients1.append(bullseye_data_filetype)
            else:
                if filetype == "wmh":
                    missing_wmh.add(patientname)
                if filetype == "cmb":
                    missing_cmb.add(patientname)
                if filetype == "epvs":
                    missing_epvs.add(patientname)
        bullseye_data_patients2 = []
        for patientname in patients2:
            patientname = str(patientname)
            if os.path.exists(os.path.join("resources", "output", patientname, "bullseyedata_"+filetype+".txt")):
                with open(os.path.join("resources", "output", patientname, "bullseyedata_"+filetype+".txt"), "rb") as f:
                    bullseye_data_filetype = pickle.load(f)
                    bullseye_data_patients2.append(bullseye_data_filetype)
            else:
                if filetype == "wmh":
                    missing_wmh.add(patientname)
                if filetype == "cmb":
                    missing_cmb.add(patientname)
                if filetype == "epvs":
                    missing_epvs.add(patientname)
        if len(bullseye_data_patients1) >= 1:
            add_patients1 = addBullseyeData(bullseye_data_patients1)
        else:
            add_patients1 = defaultBullseyeData()
        if len(bullseye_data_patients2) >= 1:
            add_patients2 = addBullseyeData(bullseye_data_patients2)
        else:
            add_patients2 = defaultBullseyeData()

        filetype_results = []
        filetype_results.append(add_patients1)
        filetype_results.append(add_patients2)
        filetype_results.append(subBullseyeData(add_patients2[0], add_patients1[0]))
        results[filetype] = filetype_results

    print(results)
    return jsonify({"bullseyedata": results, "missing_wmh": list(missing_wmh), "missing_cmb": list(missing_cmb), "missing_epvs": list(missing_epvs)})

@app.route('/show_2Dparcellation/', methods=["POST"])
def show2Dparcellation():
    global latest_labelmap
    selected_parcels = request.get_json()["selected_parcels"]
    image_masks = []
    for parcel in selected_parcels:
        if os.path.exists(os.path.join("resources", "input", "default", "layer_mask_" + parcel + ".npz")):
            image_masks.append(np.load(os.path.join("resources", "input", "default", "layer_mask_" + parcel + ".npz"))["image"])
    if len(image_masks) > 0:
        combined_labelmap = sitk.ReadImage(os.path.join(latest_labelmap, "combined.nii.gz"))
        combined_array = sitk.GetArrayFromImage(combined_labelmap)
        print("2D: ", latest_labelmap)
        return jsonify([createParcellationSlices(combined_array, image_masks, combined_labelmap, latest_labelmap)])
    print("2D: ", latest_labelmap)
    return jsonify(["combined.nii.gz"])

if __name__ == '__main__':
    preprocess_brain_mesh()
    preprocess_bullseyes()
    app.run(debug=True)
