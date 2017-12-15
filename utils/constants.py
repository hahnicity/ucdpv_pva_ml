ROW_PREFIX_NAMES = ['BN', 'ventBN', 'BS']
META_HEADER = ROW_PREFIX_NAMES + [
    'IEnd', 'BE', 'I:E ratio', 'iTime', 'eTime', 'inst_RR', 'tvi', 'tve',
    'tve:tvi ratio', 'maxF', 'minF', 'maxP', 'PIP', 'Maw', 'PEEP', 'ipAUC',
    'epAUC', ' ', 'BS.1', 'x01', 'tvi1', 'tve1', 'x02', 'tvi2', 'tve2',
    'x0_index', 'abs_time_at_BS', 'abs_time_at_x0', 'abs_time_at_BE', 'rel_time_at_BS',
    'rel_time_at_x0', 'rel_time_at_BE']

EXPERIMENTAL_META_HEADER = META_HEADER + [
    'minF_to_zero', 'pef_+0.16_to_zero', 'mean_flow_from_pef', 'dyn_compliance',
    'vol_at_.5_sec', 'vol_at_.76_sec', 'vol_at_1_sec',
    'pressure_itime_4', 'pressure_itime_5', 'pressure_itime_6',
    'pressure_itime_by_pip5', 'pressure_itime_by_pip6',
]

# input datetime format from raw breath files
IN_DATETIME_FORMAT = "%Y-%m-%d-%H-%M-%S.%f"
# output datetime format into breath array and metadata. Why did we change this??
OUT_DATETIME_FORMAT = "%Y-%m-%d %H-%M-%S.%f"

DT = 0.02

META_HEADER_TOR_3 = ROW_PREFIX_NAMES + [
    'IEnd', 'I:E ratio', 'iTime', 'eTime', 'inst_RR', 'tvi', 'tve',
    'tve:tvi ratio', 'maxF', 'minF', 'maxP', 'PIP', 'Maw', 'PEEP', 'ipAUC',
    'epAUC', ' ', 'BS', 'x01', 'tvi1', 'tve1', 'x02', 'tvi2', 'tve2']

SUBSET_HEADER = ROW_PREFIX_NAMES + [
    'IEnd', 'I:E ratio', 'iTime', 'eTime', 'inst_RR',
    'tvi', 'tve', 'tve:tvi ratio',
    'maxF', 'minF', 'maxP', 'PIP', 'Maw', 'PEEP',
    'ipAUC', 'epAUC', ' ', 'flow', 'pressure']
