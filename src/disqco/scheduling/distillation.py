from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from disqco.scheduling.commutation import DISTILL
from disqco.scheduling.helper import RAW_PAIR_REG
from disqco.scheduling.sessions import first_instruction_using_the_pair, map_comm_qubits_to_qpus

def has_raw_pairs(circuit: QuantumCircuit):
    return any(RAW_PAIR_REG.match(register.name) for register in circuit.qregs)

def is_raw_half(circuit: QuantumCircuit, qubit):
    return RAW_PAIR_REG.match(circuit.find_bit(qubit).registers[0][0].name) is not None

def raw_pair_instructions(circuit: QuantumCircuit):
    return [idx for idx, inst in enumerate(circuit.data)
            if inst.operation.name == "EPR" and is_raw_half(circuit, inst.qubits[0])]

def raw_pair_registers(circuit: QuantumCircuit, pairs_per_operation):
    qpus = sorted(set(map_comm_qubits_to_qpus(circuit).values()))
    return {qpu: QuantumRegister(pairs_per_operation - 1, "C%d_raw" % qpu) for qpu in qpus}

def append_raw_pairs_and_distill(new_circuit: QuantumCircuit, epr, raw_register_of_qpu, qpu_of):
    raw_halves_of_each_end = [raw_register_of_qpu[qpu_of[kept_half]] for kept_half in epr.qubits]
    for raw_pair in zip(*raw_halves_of_each_end):
        new_circuit.append(epr.operation.copy(), raw_pair)
    for kept_half, raw_halves in zip(epr.qubits, raw_halves_of_each_end):
        new_circuit.append(Gate(DISTILL, 1 + len(raw_halves), []), [kept_half, *raw_halves])
    for raw_halves in raw_halves_of_each_end:
        for raw_half in raw_halves:
            new_circuit.reset(raw_half)

def with_raw_pairs(circuit: QuantumCircuit, pairs_per_operation):
    if pairs_per_operation < 1:
        raise ValueError("an operation needs at least one pair")
    if pairs_per_operation == 1:
        return circuit
    if has_raw_pairs(circuit):
        raise ValueError("the circuit already has raw pairs")

    qpu_of = map_comm_qubits_to_qpus(circuit)
    raw_register_of_qpu = raw_pair_registers(circuit, pairs_per_operation)
    new_circuit = QuantumCircuit(*circuit.qregs, *raw_register_of_qpu.values(), *circuit.cregs, name=circuit.name)
    new_circuit.global_phase = circuit.global_phase
    for inst in circuit.data:
        new_circuit.append(inst.operation, inst.qubits, inst.clbits)
        if inst.operation.name == "EPR":
            append_raw_pairs_and_distill(new_circuit, inst, raw_register_of_qpu, qpu_of)
    return new_circuit

def consumer_of_half(circuit: QuantumCircuit, sessions, session_of, session_id):
    consumer = first_instruction_using_the_pair(circuit, sessions, session_of, session_id)
    if circuit.data[consumer].operation.name != DISTILL:
        return consumer
    kept_half = circuit.data[consumer].qubits[0]
    if session_of.get((consumer, kept_half)) != session_id:
        return consumer
    session_from_the_distillation = [idx for idx in sessions[session_id] if idx >= consumer]
    return first_instruction_using_the_pair(circuit, {session_id: session_from_the_distillation}, session_of, session_id)
