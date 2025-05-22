import utils

# cp query exactly as it appears in the model config
d=0
query_date= f"search earliest={-8+d}d@d latest={-1+d}d@d index=summary source=corelight_notice_lateral_movement dce_rpc_operation=*  dce_rpc_endpoint=* smb_action=* smb_filename_extension=*"
query_context= '''| table *'''
downloaded_data_dir = "/opt/mlshare/temp/hsi-partnersite/cl_lateral_data.pkl" # This will only be used in dev, config to data dir.

# query_date= f"search earliest={-11+d}d@d latest={-1+d}d@d index=summary source=corelight_notice_long_connection "
# query_context= '''| regex src_ip="^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
# | regex dest_ip="^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$"
# | table  _time index sourcetype src_ip dest_ip src_port dest_port connections_seen duration src_bytes src_pkts dest_bytes dest_pkts missed_bytes is_broadcast is_dest_internal_ip is_src_internal_ip first_uid src_loc dest_loc is_foreign src_ip_country dest_ip_country direction proto_count src_port_count dest_port_count peer_count proto splunk_server'''
# downloaded_data_dir = "/opt/mlshare/temp/hsi-partnersite/cl_notice_data.pkl" # This will only be used in dev, config to data dir.


query= query_date+query_context

df = utils.get_data_from_splunk(query, {})

df.to_pickle(downloaded_data_dir)
