# -----------------------------------------------------------------------------
# 8 time-delay lenses used in arXiv:2506.03023 (TDCOSMO 2025)
#
# Units:
#   zl, zs : redshift (dimensionless)
#   sigma_ap_los_kms : km/s  (aperture LOS velocity dispersion used in Table A.1)
#   time delays : days
#
# IMPORTANT about sign conventions:
#   The sign conventions for time delays differ slightly across COSMOGRAIL papers.
#   Here I store the numbers *as reported in the cited time-delay measurement sources*
#   and attach a short "convention_note" per system.
#
# Velocity dispersions and redshifts:
#   From arXiv:2506.03023 Table A.1 (uniform aperture definition in that work).
#
# Time delays:
#   - DESJ0408: arXiv:1706.09424 abstract
#   - HE0435: Bonvin+2017 Fig.3, using the "free-knot splines" (blue) values
#   - PG1115: arXiv:1804.09183 abstract
#   - RXJ1131: Tewes+2013 Fig.6, using the "regression difference technique" values
#   - SDSSJ1206: arXiv:1304.4474 abstract
#   - B1608: astro-ph/0208420 Table 5 (68% confidence intervals)
#   - WFI2033: arXiv:1905.08260 abstract + conclusion line for A1A2
#   - WGD2038: arXiv:2406.02683 Fig.2 inset (delays relative to image A + covariance)
# -----------------------------------------------------------------------------

tdcosmo_8lens = {
    "DESJ0408-5354": {
        "zl": 0.597,
        "zs": 2.375,
        "sigma_ap_los_kms": 242.3,
        "sigma_ap_los_err_kms": 12.2,
        "sigma_source": "arXiv:2506.03023 Table A.1",
        "time_delay_source": "arXiv:1706.09424 (Courbin+; COSMOGRAIL XVI)",
        "convention_note": "Dt(AB), Dt(AD), Dt(BD) reported in the paper; keep sign as published.",
        "time_delays_days": {
            "AB": {"dt": -112.1, "err_minus": 2.1, "err_plus": 2.1},
            "AD": {"dt": -155.5, "err_minus": 12.8, "err_plus": 12.8},
            "BD": {"dt": -42.4, "err_minus": 17.6, "err_plus": 17.6},
        },
    },

    "HE0435-1223": {
        "zl": 0.455,
        "zs": 1.693,
        "sigma_ap_los_kms": 226.6,
        "sigma_ap_los_err_kms": 5.8,
        "sigma_source": "arXiv:2506.03023 Table A.1",
        "time_delay_source": "Bonvin+2017 (COSMOGRAIL XIII), Fig.3",
        "convention_note": "Values below are the 'free-knot splines' estimates shown in Fig.3; keep sign as published.",
        "time_delays_days": {
            "AB": {"dt": -8.8,  "err_minus": 0.8, "err_plus": 0.8},
            "AC": {"dt": -1.1,  "err_minus": 0.7, "err_plus": 0.7},
            "BC": {"dt": +7.7,  "err_minus": 0.6, "err_plus": 0.6},
            "AD": {"dt": -13.8, "err_minus": 0.9, "err_plus": 0.9},
            "BD": {"dt": -5.1,  "err_minus": 0.7, "err_plus": 0.7},
            "CD": {"dt": -12.7, "err_minus": 0.9, "err_plus": 0.9},
        },
    },

    "PG1115+080": {
        "zl": 0.311,
        "zs": 1.722,
        "sigma_ap_los_kms": 235.7,
        "sigma_ap_los_err_kms": 6.6,
        "sigma_source": "arXiv:2506.03023 Table A.1",
        "time_delay_source": "arXiv:1804.09183 (Bonvin+; COSMOGRAIL XVII) abstract",
        "convention_note": "Dt(AB), Dt(AC), Dt(BC) as in abstract; keep sign as published.",
        "time_delays_days": {
            "AB": {"dt": 8.3,  "err_minus": 1.6, "err_plus": 1.5},
            "AC": {"dt": 9.9,  "err_minus": 1.1, "err_plus": 1.1},
            "BC": {"dt": 18.8, "err_minus": 1.6, "err_plus": 1.6},
        },
        "note_images": "PG1115 has a close pair (A1/A2) often treated as 'A' in time-delay reporting; check the original paper if you need A1 vs A2 explicitly.",
    },

    "RXJ1131-1231": {
        "zl": 0.295,
        "zs": 0.654,
        "sigma_ap_los_kms": 303.0,
        "sigma_ap_los_err_kms": 8.3,
        "sigma_source": "arXiv:2506.03023 Table A.1",
        "time_delay_source": "Tewes+2013 (COSMOGRAIL IX), Fig.6",
        "convention_note": "Using 'regression difference technique' values from Fig.6; keep sign as published.",
        "time_delays_days": {
            "AB": {"dt": -0.7,  "err_minus": 1.0, "err_plus": 1.0},
            "AC": {"dt": +1.1,  "err_minus": 1.5, "err_plus": 1.5},
            "BC": {"dt": +0.4,  "err_minus": 1.6, "err_plus": 1.6},
            "AD": {"dt": -90.6, "err_minus": 1.4, "err_plus": 1.4},
            "BD": {"dt": -91.4, "err_minus": 1.2, "err_plus": 1.2},
            "CD": {"dt": -91.7, "err_minus": 1.5, "err_plus": 1.5},
        },
    },

    "SDSSJ1206+4332": {
        "zl": 0.745,
        "zs": 1.789,
        "sigma_ap_los_kms": 290.5,
        "sigma_ap_los_err_kms": 9.5,
        "sigma_source": "arXiv:2506.03023 Table A.1",
        "time_delay_source": "arXiv:1304.4474 (Eulaers+; COSMOGRAIL XII) abstract",
        "convention_note": "Reported as Delta_t AB = 111.3 +/- 3 days with A leading B.",
        "time_delays_days": {
            "AB": {"dt": 111.3, "err_minus": 3.0, "err_plus": 3.0, "leading": "A"},
        },
    },

    "B1608+656": {
        "zl": 0.630,
        "zs": 1.394,
        "sigma_ap_los_kms": 305.3,
        "sigma_ap_los_err_kms": 11.0,
        "sigma_source": "arXiv:2506.03023 Table A.1",
        "time_delay_source": "astro-ph/0208420 (Fassnacht+2002) Table 5",
        "convention_note": "Table 5 gives 68% confidence intervals for delays relative to component B (tau_BA, tau_BC, tau_BD).",
        "time_delays_days": {
            # Using 68% CI as asymmetric 1-sigma-like errors
            "BA": {"dt": 31.5, "err_minus": 31.5 - 30.5, "err_plus": 33.5 - 31.5, "ci68": (30.5, 33.5)},
            "BC": {"dt": 36.0, "err_minus": 36.0 - 34.5, "err_plus": 37.5 - 36.0, "ci68": (34.5, 37.5)},
            "BD": {"dt": 77.0, "err_minus": 77.0 - 76.0, "err_plus": 79.0 - 77.0, "ci68": (76.0, 79.0)},
        },
    },

    "WFI2033-4723": {
        "zl": 0.657,
        "zs": 1.662,
        "sigma_ap_los_kms": 210.7,
        "sigma_ap_los_err_kms": 10.5,
        "sigma_source": "arXiv:2506.03023 Table A.1",
        "time_delay_source": "arXiv:1905.08260 (Bonvin+; COSMOGRAIL XVIII) abstract + conclusions",
        "convention_note": "Dt(AB), Dt(AC), Dt(BC), and Dt(A1A2) as published; keep sign as published.",
        "time_delays_days": {
            "AB":   {"dt": 36.2,  "err_minus": 0.8, "err_plus": 0.7},
            "AC":   {"dt": -23.3, "err_minus": 1.4, "err_plus": 1.2},
            "BC":   {"dt": -59.4, "err_minus": 1.3, "err_plus": 1.3},
            "A1A2": {"dt": -1.0,  "err_minus": 2.7, "err_plus": 3.1},
        },
    },

    "WGD2038-4008": {
        "zl": 0.228,
        "zs": 0.777,
        "sigma_ap_los_kms": 254.7,
        "sigma_ap_los_err_kms": 16.3,
        "sigma_source": "arXiv:2506.03023 Table A.1",
        "time_delay_source": "arXiv:2406.02683 (Wong+2024), Fig.2 inset",
        "convention_note": "Delays reported relative to image A; the paper uses notation Dt_AX = t_A - t_X.",
        "time_delays_days": {
            # errors derived from the diagonal of the published covariance matrix (days^2)
            "AB": {"dt": -12.4, "err_minus": 3.7683, "err_plus": 3.7683},
            "AC": {"dt":  -5.3, "err_minus": 3.8471, "err_plus": 3.8471},
            "AD": {"dt": -33.3, "err_minus": 6.3166, "err_plus": 6.3166},
        },
        # Order corresponds to [AB, AC, AD] (days^2)
        "covariance_days2": [
            [14.2, 6.1, 7.5],
            [6.1, 14.8, 7.1],
            [7.5, 7.1, 39.9],
        ],
    },
}

# Optional helper: flatten time delays into a "long" table-like list of rows
time_delay_rows = []
for lens, d in tdcosmo_8lens.items():
    for pair, td in d["time_delays_days"].items():
        time_delay_rows.append({
            "lens": lens,
            "zl": d["zl"],
            "zs": d["zs"],
            "sigma_ap_los_kms": d["sigma_ap_los_kms"],
            "sigma_ap_los_err_kms": d["sigma_ap_los_err_kms"],
            "pair": pair,
            "dt_days": td["dt"],
            "err_minus": td.get("err_minus"),
            "err_plus": td.get("err_plus"),
            "source": d["time_delay_source"],
        })

time_delay_rows[:3], len(time_delay_rows)
