from collections import defaultdict
import numpy as np
from disqco.graphs.QC_hypergraph import QuantumCircuitHyperGraph

def link_demand_windows(hypergraph, assignment):
    """Return a list of link records:
    {kind, root, p_root, partition, first, last}."""
    links = []

    for t, layer in hypergraph.layers.items():
        for gate in layer:

            if gate['type'] == 'group':
                root = gate['root']
                start = gate['time']
                p_root = int(assignment[start][root])
                sub_gates = gate['sub-gates']

                final_t = start
                for sub in sub_gates:
                    if sub['type'] == 'two-qubit':
                        final_t = max(final_t, sub['time'])

                first, last = {}, {}
                for sub in sub_gates:
                    if sub['type'] != 'two-qubit':
                        continue
                    q1 = sub['qargs'][1]
                    ts = sub['time']
                    p_rec = int(assignment[ts][q1])
                    if p_rec == p_root:
                        continue          # local sub-gate: no link needed
                    first[p_rec] = min(first.get(p_rec, ts), ts)
                    last[p_rec] = max(last.get(p_rec, ts), ts)

                for tau in range(start, final_t + 1):
                    p_visit = int(assignment[tau][root])
                    if p_visit == p_root:
                        continue
                    first[p_visit] = min(first.get(p_visit, start), start)
                    last[p_visit] = max(last.get(p_visit, final_t), final_t)

                for p in first:
                    links.append({'kind': 'group', 'root': root,
                                  'p_root': p_root, 'partition': p,
                                  'first': first[p], 'last': last[p]})

            elif gate['type'] == 'two-qubit':
                q0, q1 = gate['qargs']
                p0 = int(assignment[t][q0])
                p1 = int(assignment[t][q1])
                if p0 != p1:
                    links.append({'kind': 'point', 'root': q0,
                                  'p_root': p0, 'partition': p1,
                                  'first': t, 'last': t})

    num_qubits = len(assignment[0])
    depth = len(assignment)
    for q in range(num_qubits):
        for t in range(depth - 1):
            p_now = int(assignment[t][q])
            p_next = int(assignment[t + 1][q])
            if p_now != p_next:
                links.append({'kind': 'teleport', 'root': q,
                              'p_root': p_now, 'partition': p_next,
                              'first': t + 1, 'last': t + 1})

    return links


def peak_concurrent_links(links):
    """Highest number of simultaneously-open links (sweep line)."""
    events = sorted([(l['first'], 1) for l in links]
                    + [(l['last'] + 1, -1) for l in links])
    running = peak = 0
    for _, delta in events:
        running += delta
        peak = max(peak, running)
    return peak


def demand_profile(links, depth):
    """Concurrent open links at every layer t -> list of length depth."""
    profile = [0] * depth
    for l in links:
        for t in range(l['first'], min(l['last'] + 1, depth)):
            profile[t] += 1
    return profile
