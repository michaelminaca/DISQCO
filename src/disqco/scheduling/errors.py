import math
from disqco.scheduling.sessions import sessions_opening_at, first_instruction_using_the_pair


class ErrorModel:
    def __init__(self, T2_data=20000.0, T2_comm=2000.0, F_epr=0.98, F_meas=0.99, p2=0.003):
        self.T2_data = T2_data
        self.T2_comm = T2_comm
        self.F_epr = F_epr
        self.F_meas = F_meas
        self.p2 = p2


def dephasing_fidelity(t, T2):
    """fidelity of a superposition that waited t on a qubit with coherence time T2"""
    return (1 + math.exp(-t / T2)) / 2

def swap_error_cost(error_model):
    """-log of the fidelity three two-qubit gates leave: a SWAP is three CX."""
    return -3 * math.log(1 - error_model.p2)

def two_qubit_gates(circuit):
    return sum(1 for inst in circuit.data if len(inst.qubits) == 2 and inst.operation.name != "EPR")


def idle_of_each_data_qubit(state, circuit, session_of):
    """{data qubit: time between its first and last instruction that it did not spend executing gates}"""
    first, last, busy = {}, {}, {}
    for idx, inst in enumerate(circuit.data):
        for qubit in inst.qubits:
            if (idx, qubit) in session_of:
                continue
            first[qubit] = min(first.get(qubit, state.start[idx]), state.start[idx])
            last[qubit] = max(last.get(qubit, state.end[idx]), state.end[idx])
            busy[qubit] = busy.get(qubit, 0) + state.end[idx] - state.start[idx]
    return {qubit: last[qubit] - first[qubit] - busy[qubit] for qubit in first}


def herald_of(state, epr):
    """Time when the epr finished generating"""
    return getattr(state, "epr_generated_at", {}).get(epr, state.end[epr])


def hold_of_each_half(state, circuit, sessions, session_of):
    """[(time between herald and consumer, whether the half was on a data qubit: boolean)] for each half of each pair epr"""
    holds = []
    for epr, session_ids in sessions_opening_at(sessions).items():
        herald = herald_of(state, epr)
        for session_id in session_ids:
            consumer = first_instruction_using_the_pair(circuit, sessions, session_of, session_id)
            holds.append((state.start[consumer] - herald, session_id in state.moved))
    return holds


def log_fidelity(state, circuit, sessions, session_of, model):
    """log of the estimated fidelity of a finished schedule, with its parts by source"""
    logs = {}
    idle = idle_of_each_data_qubit(state, circuit, session_of)
    logs["idle dephasing"] = sum(math.log(dephasing_fidelity(t, model.T2_data)) for t in idle.values())
    holds = hold_of_each_half(state, circuit, sessions, session_of)
    logs["pair decay"] = sum(math.log(dephasing_fidelity(hold, model.T2_data if on_slot else model.T2_comm)) for hold, on_slot in holds)
    pairs = len(sessions_opening_at(sessions))
    logs["pair infidelity"] = pairs * math.log(model.F_epr)
    logs["measurement"] = 2 * pairs * math.log(model.F_meas)
    logs["two-qubit gates"] = (two_qubit_gates(circuit) + 3 * len(state.moved)) * math.log(1 - model.p2)
    return sum(logs.values()), logs


def fidelity(state, circuit, sessions, session_of, model):
    return math.exp(log_fidelity(state, circuit, sessions, session_of, model)[0])


def best_by_fidelity(candidates, circuit, sessions, session_of, model, per_unit_time=False):
    """(state, scheduled circuit) pair with the highest fidelity, or the highest fidelity per unit of makespan"""
    def score(candidate):
        state, _ = candidate
        F = fidelity(state, circuit, sessions, session_of, model)
        return F / max(state.end) if per_unit_time else F
    return max(candidates, key=score)
