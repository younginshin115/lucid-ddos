# Copyright (c) 2022 @ FBK - Fondazione Bruno Kessler
# Author: Roberto Doriguzzi-Corin
# Project: LUCID: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import ipaddress
import socket

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from utils.constants import PROTOCOLS, POWERS_OF_TWO

# Global protocol vectorizer
vector_proto = CountVectorizer()
vector_proto.fit_transform(PROTOCOLS).todense()


class PacketFeatures:
    """
    Represents the extracted features from a single network packet.

    Attributes:
        id_fwd (tuple): 5-tuple (src_ip, src_port, dst_ip, dst_port, protocol)
        id_bwd (tuple): reversed 5-tuple (dst_ip, dst_port, src_ip, src_port, protocol)
        features_list (List[int | float]): extracted numerical features
    """
    def __init__(self):
        self.id_fwd = (0, 0, 0, 0, 0)
        self.id_bwd = (0, 0, 0, 0, 0)
        self.features_list = []

    def __str__(self):
        return f"{self.id_fwd} -> {self.features_list}"


def get_ddos_flows(attackers, victims):
    """
    Expand attackers and victims from IP or subnet strings into IP lists.

    Args:
        attackers (str): attacker IP or CIDR subnet (e.g., '192.168.0.0/24')
        victims (str): victim IP or CIDR subnet

    Returns:
        dict: dictionary with 'attackers' and 'victims' as lists of IP strings
    """
    ddos_flows = {}

    ddos_flows['attackers'] = (
        [str(ip) for ip in ipaddress.IPv4Network(attackers).hosts()]
        if '/' in attackers else [str(ipaddress.IPv4Address(attackers))]
    )
    ddos_flows['victims'] = (
        [str(ip) for ip in ipaddress.IPv4Network(victims).hosts()]
        if '/' in victims else [str(ipaddress.IPv4Address(victims))]
    )

    return ddos_flows


def parse_labels(dataset_type=None, attackers=None, victims=None, label=1):
    """
    Construct a label dictionary for attacker-victim flows.

    Args:
        dataset_type (str): predefined dataset name (e.g., 'DOS2018')
        attackers (str): optional attacker IP or subnet (overrides dataset_type)
        victims (str): optional victim IP or subnet (overrides dataset_type)
        label (int): label value to assign (default = 1)

    Returns:
        dict: mapping of 5-tuple flows (src, dst) to label
    """
    from data.ddos_specs import DDOS_ATTACK_SPECS

    output_dict = {}

    if attackers and victims:
        ddos_flows = get_ddos_flows(attackers, victims)
    elif dataset_type and dataset_type in DDOS_ATTACK_SPECS:
        ddos_flows = DDOS_ATTACK_SPECS[dataset_type]
    else:
        return None

    for attacker in ddos_flows['attackers']:
        for victim in ddos_flows['victims']:
            key_fwd = (attacker, victim)
            key_bwd = (victim, attacker)

            output_dict[key_fwd] = label
            output_dict[key_bwd] = label

    return output_dict


def parse_packet(pkt):
    """
    Extract a feature vector from a packet using pyshark.

    Args:
        pkt (pyshark.packet.packet.Packet): parsed packet object

    Returns:
        PacketFeatures: parsed feature vector object or None if packet is invalid
    """
    pf = PacketFeatures()
    tmp_id = [0] * 5

    try:
        pf.features_list.append(float(pkt.sniff_timestamp))  # timestamp
        pf.features_list.append(int(pkt.ip.len))             # packet length
        pf.features_list.append(
            int(hashlib.sha256(str(pkt.highest_layer).encode()).hexdigest(), 16) % 10**8
        )
        pf.features_list.append(int(int(pkt.ip.flags, 16)))  # IP flags

        tmp_id[0] = str(pkt.ip.src)
        tmp_id[2] = str(pkt.ip.dst)

        protocols = vector_proto.transform([pkt.frame_info.protocols]).toarray()[0]
        protocols = [1 if i >= 1 else 0 for i in protocols]
        proto_value = int(np.dot(np.array(protocols), POWERS_OF_TWO))
        pf.features_list.append(proto_value)

        protocol = int(pkt.ip.proto)
        tmp_id[4] = protocol

        if pkt.transport_layer:
            if protocol == socket.IPPROTO_TCP:
                tmp_id[1] = int(pkt.tcp.srcport)
                tmp_id[3] = int(pkt.tcp.dstport)
                pf.features_list += [
                    int(pkt.tcp.len),
                    int(pkt.tcp.ack),
                    int(pkt.tcp.flags, 16),
                    int(pkt.tcp.window_size_value),
                    0, 0  # UDP, ICMP placeholders
                ]
            elif protocol == socket.IPPROTO_UDP:
                pf.features_list += [0, 0, 0, 0]
                tmp_id[1] = int(pkt.udp.srcport)
                tmp_id[3] = int(pkt.udp.dstport)
                pf.features_list.append(int(pkt.udp.length))
                pf.features_list.append(0)  # ICMP placeholder
        elif protocol == socket.IPPROTO_ICMP:
            pf.features_list += [0, 0, 0, 0, 0]
            pf.features_list.append(int(pkt.icmp.type))
        else:
            pf.features_list += [0, 0, 0, 0, 0, 0]
            tmp_id[4] = 0

        pf.id_fwd = tuple(tmp_id)
        pf.id_bwd = (tmp_id[2], tmp_id[3], tmp_id[0], tmp_id[1], tmp_id[4])

        return pf

    except AttributeError:
        # Packet does not contain required fields (e.g., not IPv4)
        return None
