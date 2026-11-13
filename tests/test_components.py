import numpy as np
import pytest
from gas_spring.hydraulics import piston_area, DigressiveCurve, restriction_pressure_drop, pipe_pressure_drop, pipe_inertance
from gas_spring.sensors import Sensor
from gas_spring.thermal import pipe_heat_rates, actuator_wetted_wall_areas, initial_pipe_temperature_profiles
from gas_spring.config import load_inputs, validate_config, load_yaml
from gas_spring.force_input import transfer_force

def test_damper_matches_points_and_is_dissipative():
    a = piston_area(.038, .014)
    c = DigressiveCurve.from_points(a,.15,4000,.8,6000)
    e = DigressiveCurve.from_points(a,.15,5000,.8,7000)
    assert c.pressure_drop(a*.15)*a == pytest.approx(4000)
    assert c.pressure_drop(a*.8)*a == pytest.approx(6000)
    assert restriction_pressure_drop(-a*.15,c,e)*a == pytest.approx(-5000)
    q=np.linspace(0,.002,100)
    assert np.all(np.diff([c.pressure_drop(x) for x in q]) > 0)
    assert c.pressure_drop(0) == 0

def test_reject_non_digressive_points():
    with pytest.raises(ValueError): DigressiveCurve.from_points(1,1,1,2,3)

def test_pipe_laminar_odd_and_inertance():
    p=pipe_pressure_drop(1e-8,1050,.01,1,.004)
    expected=128*.01*1*1e-8/(np.pi*.004**4)
    assert p == pytest.approx(expected)
    assert pipe_pressure_drop(-1e-8,1050,.01,1,.004) == pytest.approx(-p)
    assert pipe_inertance(1050,1,.004) == pytest.approx(1050/(np.pi*.004**2/4))

def test_sensor_reproducibility_and_temperature():
    s=Sensor(.01,.005, gain=1, offset=0, temperature_coefficient=.1, reference_temperature_c=20)
    x=np.array([1.,1.,1.]); t=np.array([0.,1.,2.]); amb=np.array([20.,21.,22.])
    assert np.array_equal(s.sample(x,t,np.random.default_rng(3),amb), s.sample(x,t,np.random.default_rng(3),amb))
    no_noise=Sensor(0, temperature_coefficient=.1)
    np.testing.assert_allclose(no_noise.sample(x,t,np.random.default_rng(3),amb),[1,1.1,1.2])

def test_heat_transport_reverses_and_conserves():
    a=pipe_heat_rates([10,20,30],0,1,1,1,1,1,100)
    assert sum(a) == pytest.approx(0)
    assert list(a) == [10,0,-10]
    assert list(pipe_heat_rates([10,20,30],1,1,1,0,1,1,100)) == [90,-10,-10]
    assert list(pipe_heat_rates([10,20,30],-1,1,1,0,1,1,100)) == [10,10,70]

def test_wetted_area_changes_with_position():
    f,g=actuator_wetted_wall_areas(.038,.1,.03)
    assert f == pytest.approx(np.pi*.038*.03)
    assert f+g == pytest.approx(np.pi*.038*.1)

def test_config_and_passthrough():
    p=load_yaml('config/physical.yaml'); r=load_yaml('config/run.yaml')
    validate_config(p,r)
    assert transfer_force(-125,10) == -125

def test_inputs_reject_nonmonotonic_time(tmp_path):
    import pandas as pd
    f=tmp_path/'in.parquet'
    pd.DataFrame({'time_s':[0,0], 'actuator_a_force_n':[1,1], 'actuator_b_force_n':[1,1], 'hot_ambient_c':[20,20], 'duct_air_speed_m_s':[2,2]}).to_parquet(f)
    with pytest.raises(ValueError,match='strictly increasing'):
        load_inputs({'input_parquet':str(f)}, tmp_path/'run.yaml')

def test_initial_pipe_profiles_use_each_connected_temperature():
    run = load_yaml('config/run.yaml')
    t = run['initial_temperatures_c']
    t.update(actuator_a_fluid=40, actuator_b_fluid=60,
             main_accumulator_fluid=20, blowoff_fluid=30,
             actuator_a_gas=30, actuator_b_gas=50, low_pressure_reservoir_gas=20)
    profiles = initial_pipe_temperature_profiles(run, 2)
    # Cell centres are at 1/4 and 3/4 of the pipe, with boundaries 50 -> 25
    # for the hydraulic pipe and 40 -> 20 for the gas pipe.
    np.testing.assert_allclose(profiles['hydraulic_pipe_c'], [43.75, 31.25])
    np.testing.assert_allclose(profiles['gas_pipe_c'], [35, 25])

def test_invalid_initial_temperatures_and_masses_are_rejected():
    physical = load_yaml('config/physical.yaml')
    run = load_yaml('config/run.yaml')
    run['initial_temperatures_c']['actuator_b_gas'] = -274
    with pytest.raises(ValueError, match='actuator_b_gas'):
        validate_config(physical, run)
    run['initial_temperatures_c']['actuator_b_gas'] = 50
    physical['moving_masses_kg']['blowoff'] = 0
    with pytest.raises(ValueError, match='moving_masses_kg.blowoff'):
        validate_config(physical, run)
