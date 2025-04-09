from data.parser import parse_labels
from data.process_pcap import process_pcap
from data.flow_utils import count_flows
import numpy as np
import os, glob, time, pickle
from multiprocessing import Process, Manager
from utils.constants import MAX_FLOW_LEN, TIME_WINDOW

def parse_dataset_from_pcap(args, command_options):
    manager = Manager()

    if args.output_folder is not None and os.path.isdir(args.output_folder[0]):
        output_folder = args.output_folder[0]
    else:
        output_folder = args.dataset_folder[0]

    filelist = glob.glob(args.dataset_folder[0]+ '/*.pcap')
    in_labels = parse_labels(args.dataset_type[0], args.dataset_folder[0], label=args.label)

    max_flow_len = int(args.packets_per_flow[0]) if args.packets_per_flow else MAX_FLOW_LEN
    time_window = float(args.time_window[0]) if args.time_window else TIME_WINDOW
    dataset_id = str(args.dataset_id[0]) if args.dataset_id else str(args.dataset_type[0])
    # Only first value of --traffic_type is used (e.g., "all", "ddos", "benign")
    traffic_type = str(args.traffic_type[0]) if args.traffic_type else 'all'

    process_list = []
    flows_list = []

    start_time = time.time()
    for file in filelist:
        flows = manager.list()
        p = Process(
            target=process_pcap,
            args=(file, args.dataset_type[0], in_labels, max_flow_len, flows, args.max_flows, traffic_type, time_window)
        )
        process_list.append(p)
        flows_list.append(flows)

    for p in process_list:
        p.start()
    for p in process_list:
        p.join()

    np.seterr(divide='ignore', invalid='ignore')

    try:
        preprocessed_flows = list(flows_list[0])
    except:
        print("ERROR: No traffic flows. Please check dataset folder and .pcap files.")
        exit(1)

    for results in flows_list[1:]:
        preprocessed_flows += list(results)

    filename = f"{int(time_window)}t-{max_flow_len}n-{dataset_id}-preprocess"
    output_file = os.path.join(output_folder, filename).replace("//", "/")

    with open(output_file + '.data', 'wb') as filehandle:
        pickle.dump(preprocessed_flows, filehandle)

    (total_flows, ddos_flows, benign_flows),  (total_fragments, ddos_fragments, benign_fragments) = count_flows(preprocessed_flows)

    log_string = time.strftime("%Y-%m-%d %H:%M:%S") + " | dataset_type:" + args.dataset_type[0] + \
                 " | flows (tot,ben,ddos):(" + str(total_flows) + "," + str(benign_flows) + "," + str(ddos_flows) + \
                 ") | fragments (tot,ben,ddos):(" + str(total_fragments) + "," + str(benign_fragments) + "," + str(ddos_fragments) + \
                 ") | options:" + command_options + " | process_time:" + str(time.time() - start_time) + " |\n"

    print(log_string)
    with open(os.path.join(output_folder, 'history.log'), 'a') as myfile:
        myfile.write(log_string)
