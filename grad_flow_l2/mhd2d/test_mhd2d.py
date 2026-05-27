import torch

from grad_flow_l2.mhd2d.mhd_data import generate_mhd2d_dataset_splits
from grad_flow_l2.mhd2d.mhd_solver import (
    magnetic_potential_to_field_and_current,
    prepare_mhd2d_periodic_spectral_cache,
    solve_mhd2d_trajectory_pseudospectral,
    solve_poisson_from_vorticity,
    spectral_gradient,
    streamfunction_to_velocity,
)


def test_spectral_reconstructions_and_divergence_free_fields():
    torch.manual_seed(0)
    cache = prepare_mhd2d_periodic_spectral_cache(16, 16, dtype=torch.float64)
    omega = torch.randn(2, 16, 16, dtype=torch.float64)
    omega = omega - omega.mean(dim=(-2, -1), keepdim=True)
    psi = solve_poisson_from_vorticity(omega, cache)
    psi_hat = torch.fft.fft2(psi)
    recon = torch.fft.ifft2(cache["laplace_eigs"].unsqueeze(0) * psi_hat).real
    assert torch.allclose(recon, omega, atol=2e-5, rtol=2e-5)

    u_x, u_y = streamfunction_to_velocity(psi, cache)
    dux_dx, _ = spectral_gradient(u_x, cache)
    _, duy_dy = spectral_gradient(u_y, cache)
    assert (dux_dx + duy_dy).abs().max() < 2e-5

    a = torch.randn(2, 16, 16, dtype=torch.float64)
    b_x, b_y, j = magnetic_potential_to_field_and_current(a, cache)
    dbx_dx, _ = spectral_gradient(b_x, cache)
    _, dby_dy = spectral_gradient(b_y, cache)
    assert (dbx_dx + dby_dy).abs().max() < 2e-5
    expected_j = torch.fft.ifft2(cache["laplace_eigs"].unsqueeze(0) * torch.fft.fft2(a)).real
    assert torch.allclose(j, expected_j, atol=2e-5, rtol=2e-5)


def test_zero_state_is_invariant_without_forcing():
    u0 = torch.zeros(2, 16, 16)
    traj = solve_mhd2d_trajectory_pseudospectral(
        u0=u0,
        forcing=None,
        t_final=0.2,
        dt=0.01,
        record_dt=0.1,
        nu=1e-3,
        eta=1e-3,
    )
    assert traj.shape == (3, 2, 16, 16)
    assert torch.count_nonzero(traj) == 0


def test_dataset_shapes_for_both_forcing_modes_and_warmup():
    kwargs = dict(
        n_x=8,
        n_y=8,
        n_steps=2,
        t_final=0.3,
        n_train=2,
        n_val=1,
        n_test=1,
        solver_n_x=8,
        solver_n_y=8,
        solver_dt=0.01,
        record_dt=0.1,
        warmup_time=0.1,
        omega_rms=0.2,
        magnetic_field_rms=0.2,
        chunk_size=2,
    )
    for forcing_mode in ("zero", "vorticity"):
        splits = generate_mhd2d_dataset_splits(forcing_mode=forcing_mode, **kwargs)
        assert splits["train"]["f"].shape == (2, 8, 8)
        assert splits["train"]["u0"].shape == (2, 2, 8, 8)
        assert splits["train"]["u_traj"].shape == (2, 3, 2, 8, 8)
        assert splits["meta"]["state_channels"] == 2
        assert splits["meta"]["state_names"] == ["omega", "a"]
        assert splits["meta"]["forcing_mode"] == forcing_mode
        if forcing_mode == "zero":
            assert torch.count_nonzero(splits["train"]["f"]) == 0


def _total_energy(state: torch.Tensor) -> torch.Tensor:
    cache = prepare_mhd2d_periodic_spectral_cache(
        int(state.shape[-2]), int(state.shape[-1]), dtype=state.dtype
    )
    from grad_flow_l2.mhd2d.mhd_solver import velocity_and_magnetic_field

    u_x, u_y, b_x, b_y, _ = velocity_and_magnetic_field(state, cache)
    return 0.5 * (u_x.square() + u_y.square() + b_x.square() + b_y.square()).mean()


def test_unforced_energy_decays_and_forced_dataset_is_nontrivial():
    torch.manual_seed(1)
    u0 = torch.randn(2, 16, 16, dtype=torch.float64)
    u0 = 0.02 * u0
    u0 = u0 - u0.mean(dim=(-2, -1), keepdim=True)
    traj = solve_mhd2d_trajectory_pseudospectral(
        u0=u0,
        forcing=None,
        t_final=0.2,
        dt=0.005,
        record_dt=0.1,
        nu=0.05,
        eta=0.05,
    )
    assert _total_energy(traj[-1]) < _total_energy(traj[0])

    splits = generate_mhd2d_dataset_splits(
        n_x=8,
        n_y=8,
        n_steps=1,
        t_final=0.1,
        n_train=1,
        n_val=0,
        n_test=0,
        solver_n_x=8,
        solver_n_y=8,
        solver_dt=0.01,
        record_dt=0.1,
        omega_rms=0.2,
        magnetic_field_rms=0.2,
        forcing_mode="vorticity",
        chunk_size=1,
    )
    assert torch.count_nonzero(splits["train"]["f"]) > 0
    assert splits["train"]["u_traj"].abs().sum() > 0
