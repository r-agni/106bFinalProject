import unittest

import torch

from omni_drones.utils.drone_race_geometry import (
    detect_gate_crossings_from_frames,
    detect_two_body_gate_completion,
)


class DroneRaceGeometryTest(unittest.TestCase):
    def test_center_crossing_passes(self):
        prev = torch.tensor([[-1.0, 0.0, 0.0]])
        curr = torch.tensor([[1.0, 0.0, 0.0]])

        crossed, inside, crossing, passed = detect_gate_crossings_from_frames(
            prev, curr, gate_width=2.0, gate_height=2.0
        )

        self.assertTrue(crossed.item())
        self.assertTrue(inside.item())
        self.assertTrue(passed.item())
        self.assertTrue(torch.allclose(crossing, torch.zeros_like(crossing)))

    def test_outside_aperture_does_not_pass(self):
        prev = torch.tensor([[-1.0, 1.2, 0.0]])
        curr = torch.tensor([[1.0, 1.2, 0.0]])

        crossed, inside, _, passed = detect_gate_crossings_from_frames(
            prev, curr, gate_width=2.0, gate_height=2.0
        )

        self.assertTrue(crossed.item())
        self.assertFalse(inside.item())
        self.assertFalse(passed.item())

    def test_reverse_crossing_does_not_pass(self):
        prev = torch.tensor([[1.0, 0.0, 0.0]])
        curr = torch.tensor([[-1.0, 0.0, 0.0]])

        crossed, inside, _, passed = detect_gate_crossings_from_frames(
            prev, curr, gate_width=2.0, gate_height=2.0
        )

        self.assertFalse(crossed.item())
        self.assertTrue(inside.item())
        self.assertFalse(passed.item())

    def test_high_speed_segment_detects_crossing(self):
        prev = torch.tensor([[-10.0, -0.25, 0.25]])
        curr = torch.tensor([[10.0, 0.25, -0.25]])

        crossed, inside, crossing, passed = detect_gate_crossings_from_frames(
            prev, curr, gate_width=2.0, gate_height=2.0
        )

        self.assertTrue(crossed.item())
        self.assertTrue(inside.item())
        self.assertTrue(passed.item())
        self.assertAlmostEqual(crossing[0, 0].item(), 0.0, places=6)

    def test_two_body_drone_then_payload_completes_gate(self):
        result = detect_two_body_gate_completion(
            torch.tensor([[-1.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[-2.0, 0.0, 0.0]]),
            torch.tensor([[0.5, 0.0, 0.0]]),
            gate_width=2.0,
            gate_height=2.0,
            drone_already_entered=torch.tensor([False]),
        )

        self.assertTrue(result["gate_completed"].item())
        self.assertFalse(result["payload_first"].item())
        self.assertFalse(result["aperture_missed"].item())

    def test_two_body_payload_first_fails(self):
        result = detect_two_body_gate_completion(
            torch.tensor([[-2.0, 0.0, 0.0]]),
            torch.tensor([[-1.0, 0.0, 0.0]]),
            torch.tensor([[-1.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            gate_width=2.0,
            gate_height=2.0,
            drone_already_entered=torch.tensor([False]),
        )

        self.assertFalse(result["gate_completed"].item())
        self.assertTrue(result["payload_first"].item())

    def test_two_body_payload_outside_aperture_fails(self):
        result = detect_two_body_gate_completion(
            torch.tensor([[-1.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[-1.0, 1.2, 0.0]]),
            torch.tensor([[1.0, 1.2, 0.0]]),
            gate_width=2.0,
            gate_height=2.0,
            drone_already_entered=torch.tensor([False]),
        )

        self.assertFalse(result["gate_completed"].item())
        self.assertTrue(result["aperture_missed"].item())

    def test_two_body_pending_drone_allows_later_payload(self):
        result = detect_two_body_gate_completion(
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[1.5, 0.0, 0.0]]),
            torch.tensor([[-1.0, 0.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
            gate_width=2.0,
            gate_height=2.0,
            drone_already_entered=torch.tensor([True]),
        )

        self.assertTrue(result["gate_completed"].item())
        self.assertFalse(result["payload_first"].item())


if __name__ == "__main__":
    unittest.main()
