from dv_harness.measurement_system_fault_sweep import run_fault_sweep


def test_24_of_24_seeded_faults_caught():
    result = run_fault_sweep()
    assert result.n_faults_total == 24
    assert result.n_faults_caught == 24


def test_0_false_flags_over_30_healthy_studies():
    result = run_fault_sweep()
    assert result.n_healthy_total == 30
    assert result.n_false_flags == 0, result.false_flag_detail
