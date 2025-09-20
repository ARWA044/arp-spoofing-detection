import scapy.all as scapy

def sniff_arp_packets(interface=None, count=0, timeout=None):
    """
    Sniff ARP packets from the network.
    :param interface: Network interface to sniff on.
    :param count: Number of packets to capture (0 = infinite).
    :param timeout: Timeout in seconds.
    :return: List of ARP packets.
    """
    def arp_filter(pkt):
        return pkt.haslayer(scapy.ARP)
    packets = scapy.sniff(
        iface=interface,
        filter="arp",
        prn=None,
        count=count,
        timeout=timeout,
        lfilter=arp_filter
    )
    return packets

