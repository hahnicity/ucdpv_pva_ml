from glob import glob
from os.path import dirname, join
import re

import numpy as np
import pandas as pd
from scipy.integrate import simps

from ventmap.breath_meta import get_file_experimental_breath_meta
from ventmap.raw_utils import extract_raw
from ventmap.clear_null_bytes import clear_descriptor_null_bytes
from ventmap.cut_breath_section import cut_breath_section

FA_COHORT_DIR = ["cohort_fa"]
DERIVATION_COHORT_FILES = ["cohort_derivation"]
DERIVATION_VALIDATION_COHORT_DIRS = ["cohort_derivation", "cohort_validation"]
GOLD_STANDARD_LOCATION = "gold_stnd_files"
DERIVED_COLS = [
    'I:E ratio',
    'eTime',
    'inst_RR',
    'tve',
    'tve:tvi ratio',
    'minF',
    'maxP',
    'Maw',
    'ipAUC',
    'epAUC',
    'mean_flow_from_pef',
    'dyn_compliance',
]
DT = 0.02
NON_NOISE = [
    'I:E ratio',
    'eTime',
    'inst_RR',
    'tve:tvi ratio',
]
RAW_TIME_REGEX = re.compile(r"__(\d\d:\d\d:\d\d)")
SETTINGS_COLS = ['iTime', 'tvi', 'maxF', 'PEEP']
VALIDATION_COHORT_FILES = ["cohort_validation"]


def get_y_observations(cohort_dirs):
    def do_file(file):
        patient = file.split("/")[-2]
        df = pd.read_csv(file)
        df['patient'] = [patient] * len(df)
        return df

    files = []
    for dir_ in cohort_dirs:
        files += glob(join(dirname(__file__), dir_, GOLD_STANDARD_LOCATION, "*", "*.csv"))
    df = do_file(files[0])
    for file in files[1:]:
        df = df.append(do_file(file))
    df.index = range(len(df))
    return df


def iter_baskets_contiguous(items, maxbaskets):
    '''
    generates balanced baskets from iterable, contiguous contents
    provide item_count if providing a iterator that doesn't support len()
    '''
    item_count = len(items)
    baskets = min(item_count, maxbaskets)
    items = iter(items)
    floor = item_count // baskets
    ceiling = floor + 1
    stepdown = item_count % baskets
    for x_i in xrange(baskets):
        length = ceiling if x_i < stepdown else floor
        yield [items.next() for _ in xrange(length)]


def _get_binary_target_async_data(cohort_dirs, type_):
    observations = get_y_observations(cohort_dirs)
    files = []
    for dir_ in cohort_dirs:
        files += glob(join(dirname(__file__), dir_, "0*RPI*", "*.csv"))
    target_vector = observations[type_]
    return observations, files, target_vector


def _get_multi_target_async_data(cohort_dirs, types):
    observations = get_y_observations(cohort_dirs)
    files = []
    for dir_ in cohort_dirs:
        files += glob(join(dirname(__file__), dir_, "0*RPI*", "*.csv"))
    target_vector = observations[types[0][0]] * types[0][1]
    for pva_type, class_label in types[1:]:
        target_vector = target_vector + (observations[pva_type] * class_label)
        if (pva_type == "bs") and (types[0][0] == "dbl"):
            dbl_and_bs = target_vector[target_vector == class_label + types[0][1]]
            target_vector.loc[dbl_and_bs.index] = types[0][1]
    return observations, files, target_vector


def _get_multi_target_retrospective_data(cohort_dir, types, alternating_as):
    observations, files, target_vector = _get_multi_target_async_data(cohort_dir, types)
    return _perform_dbl_alternating(observations, files, target_vector, alternating_as)


def _perform_dbl_alternating(observations, files, target_vector, alternating_as):
    dbl_obs = target_vector[target_vector == 1]
    # Basically the first breath of a double trigger will no
    # longer be marked as a double trigger but rather a label of our choosing
    #
    # usually this label is 0 but sometimes it can be something else
    alternating = True
    for idx in dbl_obs.index:
        if alternating:
            target_vector.loc[idx] = alternating_as
            alternating = False
        else:
            alternating = True
    return observations, files, target_vector


def get_bs_dbl_retrospective_first_dbl_as_bs_deriv_valid():
    types = [("dbl", 1), ("bs", 2)]
    observations, files, target_vector = _get_multi_target_async_data(DERIVATION_VALIDATION_COHORT_DIRS, types)
    target_vector[target_vector == 3] = 1
    return _perform_dbl_alternating(observations, files, target_vector, 2)


def get_bs_dbl_retrospective_first_dbl_as_norm_deriv_valid():
    types = [("dbl", 1), ("bs", 2)]
    observations, files, target_vector = _get_multi_target_async_data(DERIVATION_VALIDATION_COHORT_DIRS, types)
    target_vector[target_vector == 3] = 1
    return _perform_dbl_alternating(observations, files, target_vector, 0)


def get_bs_dbl_non_retrospective_data():
    types = [("dbl", 1), ("bs", 2)]
    return _get_multi_target_async_data(DERIVATION_COHORT_FILES, types)


def get_bs_dbl_retrospective_data():
    types = [("dbl", 1), ("bs", 2)]
    return _get_multi_target_retrospective_data(DERIVATION_COHORT_FILES, types, 0)


def get_bs_dbl_retrospective_data():
    types = [("dbl", 1), ("bs", 2)]
    return _get_multi_target_retrospective_data(DERIVATION_COHORT_FILES, types, 0)


def get_bs_dbl_retrospective_data_first_dbl_as_bs():
    types = [("dbl", 1), ("bs", 2)]
    return _get_multi_target_retrospective_data(DERIVATION_COHORT_FILES, types, 2)


def get_multi_retrospective_data_fdb_deriv_val_cohort():
    types = [("dbl", 1), ("bs", 2), ("co", 3), ("su", 4), ("mt", 5)]
    return _get_multi_target_retrospective_data(DERIVATION_VALIDATION_COHORT_DIRS, types, 2)


def get_bs_dbl_retrospective_data_first_dbl_as_bs_val_cohort():
    types = [("dbl", 1), ("bs", 2), ("co", 3), ("su", 4), ("mt", 4)]
    observations, files, target_vector = _get_multi_target_async_data(VALIDATION_COHORT_FILES, types)
    target_vector[target_vector == 3] = 1
    return _perform_dbl_alternating(observations, files, target_vector, 2)


def get_multi_target_derivation_cohort_data():
    types = [("dbl", 1), ("bs", 2), ("co", 3), ("su", 4), ("mt", 4)]
    return _get_multi_target_async_data(DERIVATION_COHORT_FILES, types)

def get_multi_target_retrospective_data():
    types = [("dbl", 1), ("bs", 2), ("co", 3), ("su", 4), ("mt", 4)]
    return _get_multi_target_retrospective_data(DERIVATION_COHORT_FILES, types, 0)


def get_multi_class_binary_label_retrospective_data():
    types = [("dbl", 1), ("bs", 1), ("co", 1), ("su", 1)]
    return _get_multi_target_retrospective_data(DERIVATION_COHORT_FILES, types, 0)


def get_multi_target_retrospective_data_first_dbl_as_bs():
    types = [("dbl", 1), ("bs", 2), ("co", 3), ("su", 4), ("mt", 4)]
    return _get_multi_target_retrospective_data(DERIVATION_COHORT_FILES, types, 2)


def get_binary_target_bs_patient_data():
    return _get_binary_target_async_data(DERIVATION_COHORT_FILES, 'bs')


def get_binary_target_deriv_valid_bs_patient_data():
    return _get_binary_target_async_data(DERIVATION_VALIDATION_COHORT_DIRS, 'bs')


def get_binary_target_fa_patient_data():
    return _get_binary_target_async_data(FA_COHORT_DIR, 'fa')


def get_binary_target_basic_dbl_trigger_data():
    return _get_binary_target_async_data(DERIVATION_COHORT_FILES, 'dbl')


def get_binary_target_deriv_valid_dbl_trigger_data():
    return _get_binary_target_async_data(DERIVATION_VALIDATION_COHORT_DIRS, 'dbl')


def get_binary_target_retrospective_dbl_trigger_data():
    observations, files, target_vector = get_binary_target_deriv_valid_dbl_trigger_data()
    return _perform_dbl_alternating(observations, files, target_vector, 0)


def get_binary_target_cough_data():
    return _get_binary_target_async_data(DERIVATION_COHORT_FILES, 'co')


def get_binary_target_suction_data():
    return _get_binary_target_async_data(DERIVATION_COHORT_FILES, 'su')


def do_basket_function(n_regions, feature_func, gold_stnd_func):
    observations, files, target_vector = gold_stnd_func()
    rows = []
    for file in files:
        patient = file.split("/")[-2]
        pt_observations = observations[observations['patient'] == patient]
        f = clear_descriptor_null_bytes(open(file))
        selection = cut_breath_section(
            f, pt_observations['BN'].min(), pt_observations['BN'].max()
        )
        all_metadata = get_file_experimental_breath_meta(selection)
        all_metadata = pd.DataFrame(all_metadata[1:], columns=all_metadata[0])
        # implement rounding
        all_metadata = all_metadata.round(2)
        all_metadata = all_metadata.round({"tvi": 1, "tve": 1})
        selection.seek(0)
        prev_prev_metadata = None
        prev_metadata = None
        pt_obs = 0
        for idx, breath in enumerate(extract_raw(selection, True)):
            metadata = all_metadata.loc[idx]
            # 28 is the index for x0. Overall this is a little hacky and
            # we should consider either adding a relative x0 time or an x01 index
            # to the metadata
            x01_index = metadata[28]
            i_pressure = breath['pressure'][:x01_index - 1] if x01_index != 0 else breath['pressure']
            bs_time = metadata[2]
            if len(i_pressure) < n_regions:
                # XXX this branch is unused and broken
                row = [0] * df.shape[1]
            else:
                row, colnames = feature_func(iter_baskets_contiguous(i_pressure, n_regions), i_pressure, metadata, prev_metadata, prev_prev_metadata)
                row = row + [bs_time, patient, breath["vent_bn"], breath['rel_bn'], file]
            rows.append(row)
            if not isinstance(prev_metadata, type(None)):
                prev_prev_metadata = prev_metadata.copy()
            prev_metadata = metadata.copy()
            pt_obs += 1
    df = pd.DataFrame(
        rows,
        columns=colnames + ['bs_time', "patient", "vent_bn", 'rel_bn', "filename"]
    )

    # Final processing on the DF
    pre_num_cols = df.shape[1]
    df.index = observations.index
    df['y'] = target_vector
    new_index = []
    for idx, i in enumerate(df.index):
        # should be <idx>-<patient id>-<bs time>
        new_index.append("{}-{}-{}".format(i, observations.iloc[idx]['patient'], df.iloc[idx]["bs_time"]))
    df.index = new_index
    del df['bs_time']
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def _suction_heuristic(_, __, metadata, *args):
    cols = ['su', 'su.2', 'mt.su', 'sumt']
    vals = [0] * len(cols)
    frame_dur = metadata['iTime'] + metadata['eTime']
    if frame_dur < 0.5 and metadata['tvi'] < 250:
        vals[0] = 1
    if metadata['eTime'] <= 0.3:
        vals[1] = 1
        vals[2] = 1
        vals[3] = 1
    return vals, cols


def _co_heuristic(_, __, metadata, *args):
    cols = ['co.orig', 'co.2thresh', 'co.notvi', 'co.sudo']
    vals = [0, 0, 0, 0]
    tve = metadata['tve']
    pif = metadata['maxF']
    itime = metadata['iTime']
    tvi = metadata['tvi']
    ip_auc = metadata['ipAUC']
    ideal_itime = 0.8
    ideal_ip_auc = metadata['maxP'] * ideal_itime
    # Needs to have an expiratory flow
    if tve > 0:
        if itime < 0.5 and tvi < 150 and ip_auc < 5 and pif > 20:
            vals[0] = 1
        if itime <= 0.2 and tvi < 150 and ip_auc < 5 and pif > 20:
            vals[1] = 1
        if itime <= 0.2 and ip_auc < 5 and pif > 20:
            vals[2] = 1
        if ip_auc < 0.25 * ideal_ip_auc and pif > 20:
            vals[3] = 1
    return vals, cols


def _bs_heuristic(_, __, metadata, *args):
    # XXX Figure out if rounding rules for tv_ratio are correct
    cols = ["bs.1", "bs.2"]
    vals = [0, 0]
    tv_ratio = metadata['tve:tvi ratio']
    e_time = metadata['eTime']
    if e_time > 0.3 and (.66 <= tv_ratio < .9):
        vals[0] = 1
    elif e_time > 0.3 and (.33 <= tv_ratio < .66):
        vals[0] = 2
    elif e_time > 0.3 and (0 <= tv_ratio < .33):
        vals[0] = 3

    if e_time <= 0.3 and metadata['tve'] > 100 and (.25 <= tv_ratio < .9):
        if .66 <= tv_ratio < .9:
            vals[1] = 1
        elif .33 <= tv_ratio < .66:
            vals[1] = 2
        elif 0 <= tv_ratio < .33:
            vals[1] = 3

    return vals, cols


def _selective_retro(cols, metadata, prev_metadata, prev_prev_metadata):
    if isinstance(prev_metadata, type(None)):
        prev_metadata = metadata
    if isinstance(prev_prev_metadata, type(None)):
        prev_prev_metadata = prev_metadata
    fused_cols = []
    prev_cols = []
    prev_prev_cols = []
    regular_cols = []
    cur_prev_cols = []
    for col in cols:
        if col.endswith("-fused"):
            fused_cols.append(col.replace("-fused", ""))
        elif col.endswith("-prev-prev"):
            prev_prev_cols.append(col.replace("-prev-prev", ""))
        elif col.endswith("-prev"):
            prev_cols.append(col.replace("-prev", ""))
        elif col.endswith("-cur:prev"):
            cur_prev_cols.append(col.replace("-cur:prev", ""))
        else:
            regular_cols.append(col)
    fused_vals = list(
        (prev_metadata[fused_cols] + metadata[fused_cols]).values
    )
    prev_vals = list(prev_metadata[prev_cols].values)
    prev_prev_vals = list(prev_prev_metadata[prev_prev_cols].values)
    cur_prev_vals = list((metadata[cur_prev_cols] / prev_metadata[cur_prev_cols]).values)
    regular_vals = list(metadata[regular_cols].values)
    fused_cols = map(lambda x: "{}-fused".format(x), fused_cols)
    prev_cols = map(lambda x: "{}-prev".format(x), prev_cols)
    prev_prev_cols = map(lambda x: "{}-prev-prev".format(x), prev_prev_cols)
    cur_prev_cols = map(lambda x: "{}-cur:prev".format(x), cur_prev_cols)
    vals = fused_vals + prev_vals + prev_prev_vals + regular_vals + cur_prev_vals
    cols = fused_cols + prev_cols + prev_prev_cols + regular_cols + cur_prev_cols
    return vals, cols


def _only_retro(_, __, metadata, prev_metadata, prev_prev_metadata):
    cols = SETTINGS_COLS + DERIVED_COLS
    fused_cols = map(lambda x: "{}-fused".format(x), cols)
    prev_cols = map(lambda x: "{}-prev".format(x), cols)
    prev_prev_cols = map(lambda x: "{}-prev-prev".format(x), cols)
    # XXX DEBUG turn this off
    #cur_prev_cols = map(lambda x: "{}-cur:prev".format(x), cols)
    all_cols = fused_cols + prev_cols + prev_prev_cols #+ cur_prev_cols
    return _selective_retro(all_cols, metadata, prev_metadata, prev_prev_metadata)


def _retrospective(_, __, metadata, prev_metadata, prev_prev_metadata):
    vals1, cols1 = _only_retro(_, __, metadata, prev_metadata, prev_prev_metadata)
    cols2 = SETTINGS_COLS + DERIVED_COLS
    all_cols = cols1 + cols2
    return _selective_retro(all_cols, metadata, prev_metadata, prev_prev_metadata)


def _dbl_trigger_heuristic(_, __, metadata, *args):
    cols = ["dbl.2", "dbl.3"]
    results = [0, 0]
    if metadata['tve:tvi ratio'] < 0.25 and metadata['eTime'] <= 0.3:
        results[0] = 1
    if 0.25 <= metadata['tve:tvi ratio'] < 0.5 and metadata['eTime'] <= 0.3 and metadata['tve'] <= 100:
        results[1] = 1
    return results, cols


def _vent_settings_func(_, __, metadata, *args):
    return metadata[SETTINGS_COLS], SETTINGS_COLS


def _derived_features_func(_, __, metadata, *args):
    return metadata[DERIVED_COLS], DERIVED_COLS


def _vent_and_derived_func(_, __, metadata, *args):
    vf_vals, vf_cols = _vent_settings_func(_, __, metadata, None)
    df_vals, df_cols = _derived_features_func(_, __, metadata, None)
    return list(vf_vals) + list(df_vals), vf_cols + df_cols


def _do_slope_ipauc_binning(baskets, i_pressure_data):
    row = []
    total = simps(i_pressure_data, dx=DT)
    bin_colnames = []
    for idx, basket in enumerate(baskets):
        slope = (basket[-1] - basket[0]) / (len(basket) * DT)
        frac = simps(basket, dx=DT) / float(total)
        row.extend([slope, frac])
        bin_colnames.extend(["slope_bin_{}".format(idx), "ipAUC_bin_{}".format(idx)])
    return row, bin_colnames


def _v2_func(baskets, i_pressure_data, metadata, *args):
    heur_row, heur_colnames = _fa_heuristic_func(None, i_pressure_data, metadata)
    bin_row, bin_colnames = _do_slope_ipauc_binning(baskets, i_pressure_data)
    return heur_row + bin_row, heur_colnames + bin_colnames


def _v2_and_metadata_func(baskets, i_pressure_data, metadata, *args):
    first_part, v2_colnames = _v2_func(baskets, i_pressure_data, metadata)
    second_part, metadata_colnames = _vent_and_derived_func(None, None, metadata)
    return first_part + second_part, v2_colnames + metadata_colnames


def _fa_heuristic_func(_, i_pressure_data, metadata, *args):
    min_ip = min(i_pressure_data)
    peep = metadata['PEEP']
    below_zero = 1 if min_ip < 0 else 0
    below_peep = 1 if min_ip < peep and min_ip >= 0 else 0
    # XXX I need to figure out what jasons rationale for this is
    mildish = 1 if min_ip >= peep and min_ip < peep + 5 else 0
    colnames = ["min_i_pressure", "is_below_0", "is_below_peep", "looks_mild"]
    return [min_ip, below_zero, below_peep, mildish], colnames


def get_suction_heuristic(_, gold_stnd_func):
    return do_basket_function(0, _suction_heuristic, gold_stnd_func)


def get_suction_all(_, gold_stnd_func):
    def suction_all(_, __, metadata, *args):
        first_vals, first_cols = _suction_heuristic(None, None, metadata)
        second_vals, second_cols = _vent_and_derived_func(None, None, metadata)
        return first_vals + second_vals, first_cols + second_cols
    return do_basket_function(0, suction_all, gold_stnd_func)


def get_suction_curated(_, gold_stnd_func):
    def suction_curated(_, __, metadata, *args):
        cols = ['dyn_compliance', 'iTime', 'eTime']
        vals = list(metadata[cols].values)
        return vals, cols
    return do_basket_function(0, suction_curated, gold_stnd_func)


def get_co_heuristic(_, gold_stnd_func):
    return do_basket_function(0, _co_heuristic, gold_stnd_func)


def get_co_all(_, gold_stnd_func):
    def co_all(_, __, metadata, *args):
        first_vals, first_cols = _co_heuristic(None, None, metadata)
        second_vals, second_cols = _vent_and_derived_func(None, None, metadata)
        return first_vals + second_vals, first_cols + second_cols
    return do_basket_function(0, co_all, gold_stnd_func)


def get_co_curated_no_tvi(_, gold_stnd_func):
    # XXX I probably need to reconsider this algo
    def co_curated(_, __, metadata, *args):
        cols = ["eTime", "dyn_compliance"]
        vals = list(metadata[cols].values)
        return vals, cols
    return do_basket_function(0, co_curated, gold_stnd_func)


def get_co_curated_with_tvi(_, gold_stnd_func):
    # XXX I probably need to reconsider this algo
    def co_curated(_, __, metadata, *args):
        cols = ["eTime", "dyn_compliance", "tvi"]
        vals = list(metadata[cols].values)
        return vals, cols
    return do_basket_function(0, co_curated, gold_stnd_func)


def get_bs_heuristic(_, gold_stnd_func):
    return do_basket_function(0, _bs_heuristic, gold_stnd_func)


def get_bs_all(_, gold_stnd_func):
    def heuristic_and_metadata(_, __, metadata, *args):
        first_vals, first_cols = _bs_heuristic(None, None, metadata)
        second_vals, second_cols = _vent_and_derived_func(None, None, metadata)
        return first_vals + second_vals, first_cols + second_cols
    return do_basket_function(0, heuristic_and_metadata, gold_stnd_func)


def get_bs_curated(_, gold_stnd_func):
    def curated(_, __, metadata, *args):
        cols = ["tve:tvi ratio", "eTime"]
        vals = list(metadata[cols].values)
        return vals, cols
    return do_basket_function(0, curated, gold_stnd_func)


def get_bs_chi2(_, gold_stnd_func):
    def chi2(_, __, metadata, *args):
        cols = ['tve:tvi ratio', 'tve', 'I:E ratio', "epAUC", 'eTime', 'inst_RR', 'mean_flow_from_pef', "ipAUC", "minF", "tvi"]
        vals = list(metadata[cols].values)
        return vals, cols
    return do_basket_function(0, chi2, gold_stnd_func)


def get_retro_non_noisy(_, gold_stnd_func):
    def non_noisy(_, __, metadata, prev_metadata, prev_prev_metadata):
        prevs = map(lambda x: "{}-prev".format(x), NON_NOISE)
        cols = NON_NOISE + prevs
        return _selective_retro(cols, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, non_noisy, gold_stnd_func)


def get_dbl_chi2(_, gold_stnd_func):
    def chi2(_, __, metadata, *args):
        cols = ["inst_RR", "I:E ratio", "maxP", "iTime", "eTime", "PEEP", "Maw", "ipAUC"]
        vals = list(metadata[cols].values)
        return vals, cols
    return do_basket_function(0, chi2, gold_stnd_func)


def get_dbl_retro_chi2(_, gold_stnd_func):
    def dbl_retro(_, __, metadata, prev_metadata, prev_prev_metadata):
        cols = ['I:E ratio-prev', 'inst_RR-prev', 'epAUC-prev', 'eTime-prev', 'tve-prev', 'mean_flow_from_pef-prev', 'tve', 'minF-prev', 'inst_RR', 'eTime', 'maxF-prev', 'PEEP-prev', 'minF', 'iTime', 'ipAUC-prev', 'mean_flow_from_pef', 'epAUC', 'Maw-prev', 'I:E ratio', 'ipAUC', 'maxP']
        return _selective_retro(cols, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, dbl_retro, gold_stnd_func)


def get_dbl_trigger_heuristic_features(_, gold_stnd_func):
    # I'm sure we can use this for something that isn't double trigger, but
    # why??
    return do_basket_function(0, _dbl_trigger_heuristic, gold_stnd_func)


def get_dbl_trigger_heuristic_and_metadata_features(_, gold_stnd_func):
    def heuristic_and_metadata(_, __, metadata, *args):
        first_vals, first_cols = _dbl_trigger_heuristic(None, None, metadata)
        second_vals, second_cols = _vent_and_derived_func(None, None, metadata)
        return first_vals + second_vals, first_cols + second_cols
    return do_basket_function(0, heuristic_and_metadata, gold_stnd_func)


def get_retro_stripped_lower_prec(_, gold_stnd_func):
    def retro_stripped_lower_prec(_, __, metadata, prev_metadata, prev_prev_metadata):
        cols = ["tve:tvi ratio", "tve:tvi ratio-prev", "eTime-prev"]
        return _selective_retro(cols, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, retro_stripped_lower_prec, gold_stnd_func)


def get_retro_stripped_expert_plus_chi2_2(_, gold_stnd_func):
    def retro_stripped_lower_prec(_, __, metadata, prev_metadata, prev_prev_metadata):
        cols = ["tve:tvi ratio", "tve:tvi ratio-prev", "eTime-prev", "inst_RR", "inst_RR-prev", "epAUC"]
        return _selective_retro(cols, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, retro_stripped_lower_prec, gold_stnd_func)


def get_retro_stripped_expert_plus_chi2_3(_, gold_stnd_func):
    def retro_stripped_lower_prec(_, __, metadata, prev_metadata, prev_prev_metadata):
        cols = ["tve:tvi ratio", "tve:tvi ratio-prev", "eTime-prev", "inst_RR", "inst_RR-prev", "epAUC", "I:E ratio-prev"]
        return _selective_retro(cols, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, retro_stripped_lower_prec, gold_stnd_func)


def get_retro_stripped_expert_plus_chi2_4(_, gold_stnd_func):
    def retro_stripped_lower_prec(_, __, metadata, prev_metadata, prev_prev_metadata):
        cols = ["tve:tvi ratio", "tve:tvi ratio-prev", "eTime-prev", "inst_RR", "inst_RR-prev", "tve-prev"]
        return _selective_retro(cols, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, retro_stripped_lower_prec, gold_stnd_func)


def get_retro_stripped_expert_plus_chi2_5(_, gold_stnd_func):
    def retro_stripped_lower_prec(_, __, metadata, prev_metadata, prev_prev_metadata):
        cols = ["tve:tvi ratio", "tve:tvi ratio-prev", "eTime-prev", "inst_RR", "inst_RR-prev", "tve-prev", "inst_RR-prev-prev"]
        return _selective_retro(cols, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, retro_stripped_lower_prec, gold_stnd_func)


def get_retro_stripped_expert_plus_chi2_6(_, gold_stnd_func):
    def retro_stripped_lower_prec(_, __, metadata, prev_metadata, prev_prev_metadata):
        cols = ["tve:tvi ratio", "tve:tvi ratio-prev", "eTime-prev", "inst_RR", "inst_RR-prev", "tve-prev", "inst_RR-prev-prev", "eTime-prev-prev"]
        return _selective_retro(cols, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, retro_stripped_lower_prec, gold_stnd_func)


def get_retro_stripped_expert_plus_chi2_7(_, gold_stnd_func):
    def retro_stripped_lower_prec(_, __, metadata, prev_metadata, prev_prev_metadata):
        cols = ["tve:tvi ratio", "tve:tvi ratio-prev", "eTime-prev", "inst_RR", "inst_RR-prev", "tve-prev", "inst_RR-prev-prev", "tve:tvi ratio-prev-prev"]
        return _selective_retro(cols, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, retro_stripped_lower_prec, gold_stnd_func)


def get_retro_stripped_expert_plus_chi2(_, gold_stnd_func):
    def retro_stripped_lower_prec(_, __, metadata, prev_metadata, prev_prev_metadata):
        cols = ["tve:tvi ratio", "tve:tvi ratio-prev", "eTime-prev", "inst_RR", "inst_RR-prev"]
        return _selective_retro(cols, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, retro_stripped_lower_prec, gold_stnd_func)


def get_retro_stripped_expert_plus_instrr_prev(_, gold_stnd_func):
    def retro_stripped_lower_prec(_, __, metadata, prev_metadata, prev_prev_metadata):
        cols = ["tve:tvi ratio", "tve:tvi ratio-prev", "eTime-prev", "inst_RR-prev"]
        return _selective_retro(cols, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, retro_stripped_lower_prec, gold_stnd_func)


def get_retro_stripped_expert_plus_instrr(_, gold_stnd_func):
    def retro_stripped_lower_prec(_, __, metadata, prev_metadata, prev_prev_metadata):
        cols = ["tve:tvi ratio", "tve:tvi ratio-prev", "eTime-prev", "inst_RR"]
        return _selective_retro(cols, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, retro_stripped_lower_prec, gold_stnd_func)


def get_retro_stripped_higher_prec(_, gold_stnd_func):
    def retro_stripped_lower_prec(_, __, metadata, prev_metadata, prev_prev_metadata):
        cols = ["tve:tvi ratio", "tve:tvi ratio-prev", "tve:tvi ratio-prev-prev", "eTime-prev", "I:E ratio", "I:E ratio-prev"]
        return _selective_retro(cols, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, retro_stripped_lower_prec, gold_stnd_func)


def get_retro_plus_metadata(_, gold_stnd_func):
    def retro_plus_metadata(_, __, metadata, prev_metadata, prev_prev_metadata):
        return _retrospective(_, __, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, retro_plus_metadata, gold_stnd_func)


def get_retro_prev_plus_metadata(_, gold_stnd_func):
    def _retro_prev_plus_metadata(_, __, metadata, prev_metadata, prev_prev_metadata):
        cols = SETTINGS_COLS + DERIVED_COLS
        retro = map(lambda x: "{}-prev".format(x), cols)
        return _selective_retro(cols + retro, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, _retro_prev_plus_metadata, gold_stnd_func)


def get_retro_prev_prev_plus_metadata(_, gold_stnd_func):
    def _retro_prev_plus_metadata(_, __, metadata, prev_metadata, prev_prev_metadata):
        cols = SETTINGS_COLS + DERIVED_COLS
        retro = map(lambda x: "{}-prev".format(x), cols)
        retro_prev_prev = map(lambda x: "{}-prev-prev".format(x), cols)
        return _selective_retro(cols + retro + retro_prev_prev, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, _retro_prev_plus_metadata, gold_stnd_func)


def get_all_greg_selected_features(_, gold_stnd_func):
    def greg_selection(_, __, metadata, prev_metadata, prev_prev_metadata):
        cols = [
            "eTime", "tve:tvi ratio", "tve", "I:E ratio", 'iTime', "inst_RR", "ipAUC", "epAUC",
            "eTime-prev", "tve:tvi ratio-prev", "tve-prev", "inst_RR-prev", "ipAUC-prev", "epAUC-prev",
            "I:E ratio-prev", "iTime-prev",  "eTime-prev-prev", "tve:tvi ratio-prev-prev",
            "tve-prev-prev", "I:E ratio-prev-prev", "iTime-prev-prev", "inst_RR-prev-prev",
            "epAUC-prev-prev", "ipAUC-prev-prev",
        ]
        return _selective_retro(cols, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, greg_selection, gold_stnd_func)


def get_dbl_trigger_retrospective_plus_metadata_features(_, gold_stnd_func):
    def _dbl_retrospective(_, __, metadata, prev_metadata, prev_prev_metadata):
        metadata_only = SETTINGS_COLS + DERIVED_COLS
        retro_meta = map(lambda x: "{}-prev".format(x), metadata_only)
        return _selective_retro(metadata_only + retro_meta, metadata, prev_metadata, prev_prev_metadata)
    return do_basket_function(0, _dbl_retrospective, gold_stnd_func)


def get_dbl_curated(_, gold_stnd_func):
    def curated(_, __, metadata, *args):
        cols = ["tve:tvi ratio", "tve", "I:E ratio", "eTime"]
        vals = list(metadata[cols].values)
        return vals, cols
    return do_basket_function(0, curated, gold_stnd_func)


def get_vent_settings_features(_, gold_stnd_func):
    return do_basket_function(0, _vent_settings_func, gold_stnd_func)


def get_derived_metadata_features(_, gold_stnd_func):
    return do_basket_function(0, _derived_features_func, gold_stnd_func)


def get_settings_and_derived_features(_, gold_stnd_func):
    # technically this is not a basket function but I don't want to spend the time to
    # re-engineer when its not going to be worth much.
    return do_basket_function(0, _vent_and_derived_func, gold_stnd_func)


def get_fa_heuristic(_, gold_stnd_func):
    return do_basket_function(1, _fa_heuristic_func, gold_stnd_func)


def get_chi2_optimum(n_regions, gold_stnd_func):
    return do_basket_function(n_regions, _v2_and_metadata_func, gold_stnd_func)


def get_v2_and_metadata(n_regions, gold_stnd_func):
    return do_basket_function(n_regions, _v2_and_metadata_func, gold_stnd_func)


def get_v2(n_regions, gold_stnd_func):
    return do_basket_function(n_regions, _v2_func, gold_stnd_func)


def get_v1(n_regions, gold_stnd_func):
    def v1_func(baskets, i_pressure_data, metadata, *args):
        colnames = ["min_i_pressure", "max_i_pressure"]
        row = [min(i_pressure_data), max(i_pressure_data)]
        bin_row, bin_colnames = _do_slope_ipauc_binning(baskets, i_pressure_data)
        return row + bin_row, colnames + bin_colnames
    return do_basket_function(n_regions, v1_func, gold_stnd_func)


def get_slopes_of_pressure_curve_df(n_regions, gold_stnd_func):
    # This function is basically broken. I am not going to take the time
    # to fix it until it is necessary
    def slope_func(baskets, i_pressure_data, metadata, *args):
        row = []
        for basket in baskets:
            # XXX Most intelligent thing to do here would be a regression
            slope = (basket[-1] - basket[0]) / (len(basket) * DT)
            row.append(slope)
        return row
    return do_basket_function(n_regions, slope_func, gold_stnd_func)


def get_integrated_pressure_curve_df(n_regions, gold_stnd_func):
    # This function is basically broken. I am not going to take the time
    # to fix it until it is necessary
    def integration_func(baskets, i_pressure_data, metadata, *args):
        row = []
        total = simps(i_pressure_data, dx=DT)
        for basket in baskets:
            frac = simps(basket, dx=DT) / float(total)
            row.append(frac)
        return row
    return do_basket_function(n_regions, integration_func, gold_stnd_func)
