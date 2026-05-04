import torch


def detect_gate_crossings_from_frames(
    prev_gate_frame: torch.Tensor,
    curr_gate_frame: torch.Tensor,
    gate_width: float,
    gate_height: float,
    *,
    eps: float = 1e-6,
):
    """Detect positive-x segment crossings through a gate aperture.

    Args:
        prev_gate_frame: Previous position in the centered gate frame, shape (..., 3).
        curr_gate_frame: Current position in the centered gate frame, shape (..., 3).
        gate_width: Full gate width along local y.
        gate_height: Full gate height along local z.
        eps: Small denominator guard for near-parallel segments.

    Returns:
        crossed_plane: Segment crossed from local x <= 0 to local x > 0.
        inside_aperture: Interpolated crossing point lies within width/height.
        crossing_point: Interpolated point at local x == 0, shape (..., 3).
        valid_pass: crossed_plane & inside_aperture.
    """
    prev_x = prev_gate_frame[..., 0]
    curr_x = curr_gate_frame[..., 0]
    crossed_plane = (prev_x <= 0.0) & (curr_x > 0.0)

    denom = curr_x - prev_x
    safe_denom = torch.where(
        denom.abs() > eps,
        denom,
        torch.full_like(denom, eps),
    )
    t = (-prev_x / safe_denom).clamp(0.0, 1.0)
    crossing_point = prev_gate_frame + t.unsqueeze(-1) * (curr_gate_frame - prev_gate_frame)

    half_width = gate_width * 0.5
    half_height = gate_height * 0.5
    inside_aperture = (
        (crossing_point[..., 1].abs() <= half_width)
        & (crossing_point[..., 2].abs() <= half_height)
    )
    valid_pass = crossed_plane & inside_aperture
    return crossed_plane, inside_aperture, crossing_point, valid_pass


def detect_two_body_gate_completion(
    prev_drone_gate_frame: torch.Tensor,
    curr_drone_gate_frame: torch.Tensor,
    prev_payload_gate_frame: torch.Tensor,
    curr_payload_gate_frame: torch.Tensor,
    gate_width: float,
    gate_height: float,
    drone_already_entered: torch.Tensor,
    *,
    eps: float = 1e-6,
):
    """Detect ordered drone+payload gate completion for a slung-load race.

    A gate is completed only after the drone has crossed the target aperture and
    the payload then crosses the same target aperture. Simultaneous valid
    crossings are allowed. A payload crossing before the drone has entered is a
    sequencing failure because it means the policy is not carrying the load
    through the gate in the intended order.
    """
    (
        drone_crossed,
        drone_inside,
        drone_crossing,
        drone_valid,
    ) = detect_gate_crossings_from_frames(
        prev_drone_gate_frame,
        curr_drone_gate_frame,
        gate_width,
        gate_height,
        eps=eps,
    )
    (
        payload_crossed,
        payload_inside,
        payload_crossing,
        payload_valid,
    ) = detect_gate_crossings_from_frames(
        prev_payload_gate_frame,
        curr_payload_gate_frame,
        gate_width,
        gate_height,
        eps=eps,
    )

    drone_entered = drone_already_entered | drone_valid
    gate_completed = drone_entered & payload_valid
    payload_first = payload_valid & (~drone_already_entered) & (~drone_valid)
    aperture_missed = (drone_crossed & ~drone_inside) | (payload_crossed & ~payload_inside)

    return {
        "drone_crossed": drone_crossed,
        "drone_inside": drone_inside,
        "drone_crossing": drone_crossing,
        "drone_valid": drone_valid,
        "payload_crossed": payload_crossed,
        "payload_inside": payload_inside,
        "payload_crossing": payload_crossing,
        "payload_valid": payload_valid,
        "drone_entered": drone_entered,
        "gate_completed": gate_completed,
        "payload_first": payload_first,
        "aperture_missed": aperture_missed,
    }
