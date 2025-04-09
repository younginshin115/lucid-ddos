# Copyright (c) 2022 @ FBK - Fondazione Bruno Kessler
# Author: Roberto Doriguzzi-Corin
# Licensed under the Apache License, Version 2.0

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
