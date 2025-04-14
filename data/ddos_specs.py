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
