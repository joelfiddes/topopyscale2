"""Cross-backend radiation kernel tests."""

import numpy as np

from tests.conftest import RTOL


class TestPartitionShortwave:
    def test_clear_sky(self, radiation_mod):
        """High clearness index → mostly direct."""
        direct, diffuse = radiation_mod.partition_shortwave(
            np.float64(800.0), np.float64(45.0), np.float64(0.85)
        )
        assert float(direct) > float(diffuse)

    def test_overcast(self, radiation_mod):
        """Low clearness index → mostly diffuse."""
        direct, diffuse = radiation_mod.partition_shortwave(
            np.float64(200.0), np.float64(45.0), np.float64(0.15)
        )
        assert float(diffuse) > float(direct)

    def test_night(self, radiation_mod):
        """Solar elevation <= 0: direct = 0, diffuse = total."""
        direct, diffuse = radiation_mod.partition_shortwave(
            np.float64(100.0), np.float64(-5.0), np.float64(0.5)
        )
        np.testing.assert_allclose(float(direct), 0.0, atol=1e-15)
        np.testing.assert_allclose(float(diffuse), 100.0, rtol=RTOL)

    def test_energy_conservation(self, radiation_mod):
        """Direct + diffuse should equal total."""
        sw_total = np.array([500.0, 800.0, 100.0])
        solar_elev = np.array([30.0, 60.0, 10.0])
        kt = np.array([0.5, 0.8, 0.2])

        direct, diffuse = radiation_mod.partition_shortwave(sw_total, solar_elev, kt)
        np.testing.assert_allclose(
            np.asarray(direct) + np.asarray(diffuse), sw_total, rtol=RTOL
        )

    def test_kt_clamping(self, radiation_mod):
        """kt > 1 or < 0 should be clamped."""
        direct, diffuse = radiation_mod.partition_shortwave(
            np.float64(500.0), np.float64(45.0), np.float64(1.5)
        )
        # Should not crash, direct + diffuse = total
        np.testing.assert_allclose(float(direct) + float(diffuse), 500.0, rtol=RTOL)


class TestSlopeCorrection:
    def test_flat_surface(self, radiation_mod):
        """Flat surface: correction = direct / sin(solar_elev) * sin(solar_elev) = direct."""
        result = radiation_mod.slope_correction(
            np.float64(500.0), np.float64(45.0), np.float64(180.0),
            np.float64(0.0), np.float64(180.0)
        )
        np.testing.assert_allclose(float(result), 500.0, rtol=1e-6)

    def test_south_facing_enhancement(self, radiation_mod):
        """South-facing slope with sun from south should get more radiation."""
        # Sun from south (azimuth=180), moderate elevation
        flat = radiation_mod.slope_correction(
            np.float64(500.0), np.float64(30.0), np.float64(180.0),
            np.float64(0.0), np.float64(180.0)
        )
        tilted = radiation_mod.slope_correction(
            np.float64(500.0), np.float64(30.0), np.float64(180.0),
            np.float64(30.0), np.float64(180.0)
        )
        assert float(tilted) > float(flat)

    def test_night_zero(self, radiation_mod):
        """Low sun: correction should be 0."""
        result = radiation_mod.slope_correction(
            np.float64(500.0), np.float64(0.1), np.float64(180.0),
            np.float64(30.0), np.float64(180.0)
        )
        np.testing.assert_allclose(float(result), 0.0, atol=1e-10)


class TestDiffuseCorrection:
    def test_svf_1(self, radiation_mod):
        """SVF=1 → unchanged."""
        result = radiation_mod.diffuse_correction(np.float64(200.0), np.float64(1.0))
        np.testing.assert_allclose(float(result), 200.0, rtol=RTOL)

    def test_svf_half(self, radiation_mod):
        """SVF=0.5 → halved."""
        result = radiation_mod.diffuse_correction(np.float64(200.0), np.float64(0.5))
        np.testing.assert_allclose(float(result), 100.0, rtol=RTOL)

    def test_svf_0(self, radiation_mod):
        """SVF=0 → zero."""
        result = radiation_mod.diffuse_correction(np.float64(200.0), np.float64(0.0))
        np.testing.assert_allclose(float(result), 0.0, atol=1e-15)


class TestLongwaveCorrection:
    """Tests for Brutsaert (1975) emissivity-based longwave correction.

    Signature: longwave_correction(lw_source, t_source, t_unit, vp_source, vp_unit, svf)
    """

    def test_svf_1_same_conditions(self, radiation_mod):
        """SVF=1, same T and VP → emissivity transfer preserves LW closely."""
        # Same temperature and vapor pressure: clear-sky emissivity is identical
        # at source and target, so cloud emissivity transfers exactly.
        result = radiation_mod.longwave_correction(
            np.float64(300.0), np.float64(280.0), np.float64(280.0),
            np.float64(800.0), np.float64(800.0), np.float64(1.0),
        )
        # With identical conditions, output ≈ input
        np.testing.assert_allclose(float(result), 300.0, rtol=1e-2)

    def test_svf_0(self, radiation_mod):
        """SVF=0 → pure terrain emission at σT⁴."""
        SIGMA = 5.670374419e-8
        t_unit = 270.0
        result = radiation_mod.longwave_correction(
            np.float64(300.0), np.float64(280.0), np.float64(t_unit),
            np.float64(800.0), np.float64(600.0), np.float64(0.0),
        )
        expected = SIGMA * t_unit**4
        np.testing.assert_allclose(float(result), expected, rtol=RTOL)

    def test_colder_target_less_lw(self, radiation_mod):
        """Colder unit should receive less longwave than warmer unit."""
        warm = radiation_mod.longwave_correction(
            np.float64(300.0), np.float64(280.0), np.float64(280.0),
            np.float64(800.0), np.float64(800.0), np.float64(1.0),
        )
        cold = radiation_mod.longwave_correction(
            np.float64(300.0), np.float64(280.0), np.float64(260.0),
            np.float64(800.0), np.float64(600.0), np.float64(1.0),
        )
        assert float(cold) < float(warm)

    def test_physical_range(self, radiation_mod):
        """Output should stay within physical bounds (0 to ~500 W/m²)."""
        SIGMA = 5.670374419e-8
        # Typical mountain scenario: source at 2000m (280K), target at 4000m (260K)
        result = radiation_mod.longwave_correction(
            np.float64(300.0), np.float64(280.0), np.float64(260.0),
            np.float64(800.0), np.float64(400.0), np.float64(0.8),
        )
        val = float(result)
        assert val > 0.0, f"LW should be positive, got {val}"
        assert val < SIGMA * 330.0**4, f"LW exceeds blackbody at 330K, got {val}"

    def test_brutsaert_emissivity_bounds(self, radiation_mod):
        """All-sky emissivity should be capped at 1.0."""
        # High vapor pressure + high LW → potential emissivity > 1
        result = radiation_mod.longwave_correction(
            np.float64(400.0), np.float64(280.0), np.float64(280.0),
            np.float64(2000.0), np.float64(2000.0), np.float64(1.0),
        )
        SIGMA = 5.670374419e-8
        # With capped emissivity, output ≤ σT⁴
        assert float(result) <= SIGMA * 280.0**4 + 1e-6

    def test_array_input(self, radiation_mod):
        """Test with array inputs for cross-backend agreement."""
        lw = np.array([250.0, 300.0, 350.0])
        t_src = np.array([280.0, 285.0, 290.0])
        t_tgt = np.array([270.0, 275.0, 260.0])
        vp_src = np.array([800.0, 900.0, 1000.0])
        vp_tgt = np.array([600.0, 700.0, 500.0])
        svf = np.array([0.9, 0.7, 1.0])

        result = radiation_mod.longwave_correction(lw, t_src, t_tgt, vp_src, vp_tgt, svf)
        result = np.asarray(result)
        assert result.shape == (3,)
        assert np.all(result > 0)
        assert np.all(result < 500)
