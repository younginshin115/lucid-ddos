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

# DDoS flow specifications per dataset

IDS2018_DDOS_FLOWS = {
    'attackers': [
        '18.218.115.60', '18.219.9.1', '18.219.32.43',
        '18.218.55.126', '52.14.136.135', '18.219.5.43',
        '18.216.200.189', '18.218.229.235', '18.218.11.51', '18.216.24.42'
    ],
    'victims': ['18.218.83.150', '172.31.69.28']
}

IDS2017_DDOS_FLOWS = {
    'attackers': ['172.16.0.1'],
    'victims': ['192.168.10.50']
}

CUSTOM_DDOS_SYN = {
    'attackers': ['11.0.0.' + str(x) for x in range(1, 255)],
    'victims': ['10.42.0.2']
}

DOS2019_FLOWS = {
    'attackers': ['172.16.0.5'],
    'victims': ['192.168.50.1', '192.168.50.4']
}

DDOS_ATTACK_SPECS = {
    'DOS2017': IDS2017_DDOS_FLOWS,
    'DOS2018': IDS2018_DDOS_FLOWS,
    'SYN2020': CUSTOM_DDOS_SYN,
    'DOS2019': DOS2019_FLOWS
}

DDOS_ATTACK_SPECS_EXTENDED = {
    'CIC-IDS-2017': {
        'PortScan': {
            'attackers': ['192.168.10.8'],
            'victims': ['192.168.10.50']
        },
        'DoS_GoldenEye': {
            'attackers': ['192.168.10.9'],
            'victims': ['192.168.10.50']
        },
        'DoS_Slowloris': {
            'attackers': ['192.168.10.9'],
            'victims': ['192.168.10.50']
        },
        'DoS_Hulk': {
            'attackers': ['192.168.10.9'],
            'victims': ['192.168.10.50']
        },
        'DoS_Slowhttptest': {
            'attackers': ['192.168.10.9'],
            'victims': ['192.168.10.50']
        },
        'DoS_Smack': {
            'attackers': ['192.168.10.5'],
            'victims': ['192.168.10.50']
        },
        'DoS_Backdoor': {
            'attackers': ['192.168.10.3'],
            'victims': ['192.168.10.50']
        }
    },

    'CIC-IDS-2018': {
        'DDoS_LOIC_HTTP': {
            'attackers': ['18.218.115.60'],
            'victims': ['18.218.83.150']
        },
        'DDoS_HOIC': {
            'attackers': ['18.219.9.1'],
            'victims': ['18.218.83.150']
        },
        'DDoS_UDP': {
            'attackers': ['18.219.32.43'],
            'victims': ['18.218.83.150']
        },
        'DDoS_Syn': {
            'attackers': ['18.218.55.126'],
            'victims': ['18.218.83.150']
        }
    },

    'CIC-DDoS-2019': {
        'DNS': {
            'attackers': ['172.16.0.5'],
            'victims': ['192.168.50.1', '192.168.50.4']
        },
        'NetBIOS': {
            'attackers': ['192.168.20.5'],
            'victims': ['192.168.20.1']
        },
        'LDAP': {
            'attackers': ['192.168.10.5'],
            'victims': ['192.168.10.1']
        },
        'MSSQL': {
            'attackers': ['192.168.15.5'],
            'victims': ['192.168.15.1']
        },
        'NTP': {
            'attackers': ['192.168.25.5'],
            'victims': ['192.168.25.1']
        },
        'SNMP': {
            'attackers': ['192.168.35.5'],
            'victims': ['192.168.35.1']
        },
        'UDP': {
            'attackers': ['192.168.40.5'],
            'victims': ['192.168.40.1']
        },
        'WebDDoS': {
            'attackers': ['192.168.30.5'],
            'victims': ['192.168.30.1']
        }
    }
}
