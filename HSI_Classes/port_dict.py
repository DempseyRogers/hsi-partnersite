port_range = [0, 1024, 49152, 65536]
port_types = ["Well Known", "Registered", "Dynamic"]

three_port_dict = {}
for t in range(len(port_types)):
    for i in range(port_range[t], port_range[t + 1]):
        three_port_dict.update({i: port_types[t]})
