import pandas as pd
import numpy as np
from pyhive import hive

#--------------------------------------------------------------
'''
# Create Tables
!beeline -u 'jdbc:hive2://wuwvc9hddn24.prod.wudip.com:10000/risk_user;principal=hive/wuwvc9hddn24.prod.wudip.com@PROD.WUDIP.COM' -f 'WOE_Dashboard/base.sql'
!beeline -u 'jdbc:hive2://wuwvc9hddn24.prod.wudip.com:10000/risk_user;principal=hive/wuwvc9hddn24.prod.wudip.com@PROD.WUDIP.COM' -f 'WOE_Dashboard/label.sql'

# Build Hive Connection
conn = hive.Connection(host="wuwvc9hddn24.prod.wudip.com", port=10000, auth="KERBEROS", kerberos_service_name="hive") 

# Read from Hive
base = pd.read_sql("SELECT * FROM risk_user.woe_dashboard_base", conn)
perf = pd.read_sql("SELECT * FROM risk_user.woe_dashboard_perf", conn)


# Save to local
base.to_csv('WOE_Dashboard/data/base.csv', index=False, encoding = 'utf-8')
perf.to_csv('WOE_Dashboard/data/perf.csv', index=False)
'''

#--------------------------------------------------------------
'''
# Read from disc
base = pd.read_csv("WOE_Dashboard/data/base.csv")
perf = pd.read_csv("WOE_Dashboard/data/perf.csv")

# Join
data = base.merge(perf, on='csntransactionid')

# use determined good or bad only
data_use = data.loc[data['fraud'].isin([0,1])]
data_use = data_use.rename(index=str, columns={"fraud": "target"})

# Save to local
data_use.to_csv('WOE_Dashboard/data/raw_data.csv', index=False)
'''

#--------------------------------------------------------------
# Build config file

data_use = pd.read_csv('./data/raw_data.csv')
data_use.loc[data_use['target'].isnull(), 'target'] = 0
data_use['weight'] = data_use['target']
print(data_use.head(5))
data_use.to_csv('./data/raw_data_for_woe.csv', index=False)

var_name = list(data_use.columns)

# Make sure these are mutually exclusive lists
tobe_bin = ['maxmind_riskscore','modelcalc_dist_real_ip_vs_sender_location']
candidate = ['opacket_channel','type','card_type','opacket_preauth_disposition',
             'spacket_disposition','risksegment_controlgroup','opacket_dfp_device_type',
             'opacket_dfp_device_os','opacket_dfp_device_browser_type',
             'opacket_receiver_country','opacket_receiver_state','ownership_group']
rebin = ['ea_score', 'grossamount']

is_tobe_bin = [1 if v in tobe_bin + rebin else 0 for v in var_name]
is_candidate = [1 if v in candidate else 0 for v in var_name]
is_rebin_provided = [1 if v in rebin else 0 for v in var_name]

modelfeature = tobe_bin + candidate + rebin
is_modelfeature = [1 if v in modelfeature else 0 for v in var_name]

config_dic = {"var_name": var_name,
              "var_dtype": list(data_use.dtypes.astype('str')),
              "is_tobe_bin": is_tobe_bin,
              "is_candidate": is_candidate,
              "is_modelfeature": is_modelfeature,
              "is_rebin_provided": is_rebin_provided}

config = pd.DataFrame(config_dic)
config = config[['var_name', 'var_dtype', 'is_tobe_bin', 'is_candidate', 'is_modelfeature', 'is_rebin_provided']]

config.to_csv('./data/config.csv', index=False)