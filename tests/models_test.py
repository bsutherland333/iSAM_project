import numpy as np
import pytest

from src.models import motion_model, inverse_motion_model, sensor_model, _wrap_within_pi


def test_motion_model():
    x0 = np.array([[0, 0, 0]], float).T
    u = np.array([[0, 1, np.pi/4]], float).T
    x1 = np.array([[1, 0, np.pi/4]], float).T
    assert np.allclose(motion_model(x0, u), x1)

    u = np.array([[-np.pi/2, 2, -np.pi/4]], float).T
    x2 = np.array([[np.sqrt(2) + 1, -np.sqrt(2), -np.pi/2]], float).T
    assert np.allclose(motion_model(x1, u), x2)

    u = np.array([[np.deg2rad(-170), 0, 0]], float).T
    x3 = np.array([[np.sqrt(2) + 1, -np.sqrt(2), np.deg2rad(100)]], float).T
    assert np.allclose(motion_model(x2, u), x3)


def test_inverse_motion_model():
    x0 = np.array([[0, 0, 0]], float).T
    u = np.array([[0, 1, np.pi/4]], float).T
    x1 = motion_model(x0, u)
    assert np.allclose(inverse_motion_model(x0, x1), u)

    u = np.array([[-np.pi/2, 2, -np.pi/4]], float).T
    x2 = motion_model(x1, u)
    assert np.allclose(inverse_motion_model(x1, x2), u)

    u = np.array([[np.deg2rad(-170), 0.1, 0]], float).T
    x3 = motion_model(x2, u)
    assert np.allclose(inverse_motion_model(x2, x3), u)


def test_sensor_model():
    x = np.array([[1, 1, np.pi/4]], float).T

    # In front of the robot
    landmark = np.array([[3, 3]], float).T
    assert np.allclose(sensor_model(x, landmark), np.array([[2*np.sqrt(2), 0]]).T)

    # 45 degrees to the left of the robot
    landmark = np.array([[1, 3]], float).T
    assert np.allclose(sensor_model(x, landmark), np.array([[2, np.pi/4]]).T)

    # Behind the robot
    landmark = np.array ([[-1, -1]], float).T
    assert np.allclose(sensor_model(x, landmark), np.array([[2*np.sqrt(2), -np.pi]]).T)

    # 90 degrees right of the robot
    landmark = np.array([[2, 0]], float).T
    assert np.allclose(sensor_model(x, landmark), np.array([[np.sqrt(2), -np.pi/2]]).T)


def test_wrap_within_pi():
    assert _wrap_within_pi(0) == 0
    assert _wrap_within_pi(np.pi) == -np.pi
    assert _wrap_within_pi(-np.pi) == -np.pi
    assert _wrap_within_pi(2 * np.pi) == 0
    assert _wrap_within_pi(-2 * np.pi) == 0
    assert _wrap_within_pi(3 * np.pi) == -np.pi
    assert _wrap_within_pi(-3 * np.pi) == -np.pi
    assert _wrap_within_pi(0.5 * np.pi) == 0.5 * np.pi
    assert _wrap_within_pi(-0.5 * np.pi) == -0.5 * np.pi
    assert _wrap_within_pi(np.pi + 0.01) == -np.pi + 0.01
    assert _wrap_within_pi(np.pi - 0.01) == np.pi - 0.01
    assert _wrap_within_pi(-np.pi + 0.01) == -np.pi + 0.01
    assert _wrap_within_pi(-np.pi - 0.01) == np.pi - 0.01

