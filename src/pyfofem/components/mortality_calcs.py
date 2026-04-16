# -*- coding: utf-8 -*-
"""
mortality_calcs.py – FOFEM post-fire tree mortality models.

Provides three mortality equations:

* ``mort_bolchar`` – bole char model (BOLCHAR; Keyser 2018) for broadleaf spp.
* ``mort_crnsch``  – crown scorch model (CRNSCH) for conifers and other spp.
* ``mort_crcabe``  – cambium kill / post-fire model (CRCABE; Hood & Lutes 2017)
                     for conifers.
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import numpy as np
from typing import Optional, Union

from ._component_helpers import _is_scalar
from .tree_flame_calcs import (
    calc_bark_thickness,
    calc_char_ht,
    calc_crown_length_vol_scorched,
    calc_flame_length,
    calc_scorch_ht,
    SPP_CODES,
)


def mort_bolchar(
    spp: Union[str, int, np.ndarray],
    dbh: Union[float, np.ndarray],
    char_ht: Union[float, np.ndarray],
    tree_code_dict: dict = None,
) -> Union[float, np.ndarray]:
    """
    FOFEM bole char post-fire mortality model (BOLCHAR).

    Based on Keyser (2018). Accepts a single tree (scalar inputs) or multiple
    trees (array inputs) of equal length. Models are available for the 10
    broadleaf species listed below; unsupported species codes return ``np.nan``
    with a printed warning.

    Available species:
        - Red maple        – RURU5, ACRU, ACRUD, ACRUD2, ACRUR, ACRUT2, ACRUT, ACRUT3
        - Flowering dogwood – COFL2
        - Blackgum         – NYSY, NYSYB, NYSYC, NYSYD, NYSYT, NYSYU, NYUR2, NYBI
        - Sourwood         – OXAR
        - White oak        – QUAL, QUALS, QUALS2, QUAL3, QUBI, QUGA4, QUGAG2, QUGAS
        - Scarlet oak      – QUCO2, QUCOC, QUCOT
        - Blackjack oak    – QUMA3, QUMAA2, QUMAA, QUMAM2
        - Chestnut oak     – QUMI, QUPR4
        - Black oak        – QUVE, QUVEM, QUKE
        - Sassafras        – SAAL5

    :param spp: Species code(s) (str, int, or np.ndarray). A single string or
        int may be passed for a single tree. If int, codes are mapped to FOFEM
        species codes using ``tree_code_dict`` if provided; otherwise via the
        lookup in ``species_codes_lut.csv``. Unknown codes map to ``'UNK'``.
    :param dbh: Diameter at breast height (cm), measured at 1.3 m above ground.
        Scalar float or np.ndarray.
    :param char_ht: Bole char height (m), measured in the field. Scalar float
        or np.ndarray.
    :param tree_code_dict: Optional dict mapping numeric species codes to FOFEM
        species code strings (e.g., ``{316: 'ACRU'}``).

    :return: Mortality probability (float in [0, 1], or ``np.nan`` for
        unsupported species). Returns a scalar ``float`` when all inputs are
        scalars, otherwise a 1D ``np.ndarray`` of the same length as the inputs.
    """
    # Detect whether the caller passed scalar inputs
    scalar_input = _is_scalar(spp) and _is_scalar(dbh) and _is_scalar(char_ht)

    # Verify tree_code_dict
    if tree_code_dict is not None and not isinstance(tree_code_dict, dict):
        print('tree_code_dict must be a dictionary, mapping numeric species codes to FOFEM species code strings. '
              'Using default species code mapping from species_codes_lut.csv.')
        tree_code_dict = None

    # Coerce all inputs to np.ndarray
    spp = np.atleast_1d(np.array(spp))
    dbh = np.atleast_1d(np.asarray(dbh, dtype=float))
    char_ht = np.atleast_1d(np.asarray(char_ht, dtype=float))

    # Map numeric spp codes to FOFEM string codes if needed
    if np.issubdtype(spp.dtype, np.integer):
        unique_num_cds = np.unique(spp)
        for num_cd in unique_num_cds:
            mask = spp == num_cd
            if tree_code_dict is None:
                spp[mask] = (SPP_CODES.loc[SPP_CODES['num_cd'] == num_cd, 'fofem_cd'].iloc[0]
                             if num_cd in SPP_CODES['num_cd'].values else 'UNK')
            else:
                spp[mask] = tree_code_dict.get(num_cd, 'UNK')
    else:
        spp = spp.astype(str)

    # Output array – NaN by default (unsupported species remain NaN)
    Pm = np.full(len(spp), np.nan)

    # --- Species masks ---
    mask_acru  = np.isin(spp, ['RURU5', 'ACRU', 'ACRUD', 'ACRUD2', 'ACRUR', 'ACRUT2', 'ACRUT', 'ACRUT3'])
    mask_cofl2 = spp == 'COFL2'
    mask_nysy  = np.isin(spp, ['NYSY', 'NYSYB', 'NYSYC', 'NYSYD', 'NYSYT', 'NYSYU', 'NYUR2', 'NYBI'])
    mask_oxar  = spp == 'OXAR'
    mask_qual  = np.isin(spp, ['QUAL', 'QUALS', 'QUALS2', 'QUAL3', 'QUBI', 'QUGA4', 'QUGAG2', 'QUGAS'])
    mask_quco2 = np.isin(spp, ['QUCO2', 'QUCOC', 'QUCOT'])
    mask_quma3 = np.isin(spp, ['QUMA3', 'QUMAA2', 'QUMAA', 'QUMAM2'])
    mask_qumi  = np.isin(spp, ['QUMI', 'QUPR4'])
    mask_quve  = np.isin(spp, ['QUVE', 'QUVEM', 'QUKE'])
    mask_saal5 = spp == 'SAAL5'

    # FOFEM Eq 100 - Red Maple
    if np.any(mask_acru):
        Pm[mask_acru] = 1 / (1 + np.exp(
            -(2.3017 + (-0.3267 * dbh[mask_acru]) + (1.1137 * char_ht[mask_acru]))))

    # FOFEM Eq 101 - Flowering Dogwood
    if np.any(mask_cofl2):
        Pm[mask_cofl2] = 1 / (1 + np.exp(
            -(-0.8727 + (-0.1814 * dbh[mask_cofl2]) + (4.1947 * char_ht[mask_cofl2]))))

    # FOFEM Eq 102 - Blackgum
    if np.any(mask_nysy):
        Pm[mask_nysy] = 1 / (1 + np.exp(
            -(-2.7899 + (-0.5511 * dbh[mask_nysy]) + (1.2888 * char_ht[mask_nysy]))))

    # FOFEM Eq 103 - Sourwood
    if np.any(mask_oxar):
        Pm[mask_oxar] = 1 / (1 + np.exp(
            -(1.9438 + (-0.4602 * dbh[mask_oxar]) + (1.6352 * char_ht[mask_oxar]))))

    # FOFEM Eq 104 - White Oak
    if np.any(mask_qual):
        Pm[mask_qual] = 1 / (1 + np.exp(
            -(-1.8137 + (-0.0603 * dbh[mask_qual]) + (0.8666 * char_ht[mask_qual]))))

    # FOFEM Eq 105 - Scarlet Oak
    if np.any(mask_quco2):
        Pm[mask_quco2] = 1 / (1 + np.exp(
            -(-1.6262 + (-0.0339 * dbh[mask_quco2]) + (0.6901 * char_ht[mask_quco2]))))

    # FOFEM Eq 106 - Blackjack Oak
    if np.any(mask_quma3):
        Pm[mask_quma3] = 1 / (1 + np.exp(
            -(0.3714 + (-0.1005 * dbh[mask_quma3]) + (1.5577 * char_ht[mask_quma3]))))

    # FOFEM Eq 107 - Chestnut Oak
    if np.any(mask_qumi):
        Pm[mask_qumi] = 1 / (1 + np.exp(
            -(-1.8137 + (-0.0603 * dbh[mask_qumi]) + (0.8666 * char_ht[mask_qumi]))))

    # FOFEM Eq 108 - Black Oak
    if np.any(mask_quve):
        Pm[mask_quve] = 1 / (1 + np.exp(
            -(0.1122 + (-0.1287 * dbh[mask_quve]) + (1.2612 * char_ht[mask_quve]))))

    # FOFEM Eq 109 - Sassafras
    if np.any(mask_saal5):
        Pm[mask_saal5] = 1 / (1 + np.exp(
            -(-1.8137 + (-0.0603 * dbh[mask_saal5]) + (0.8666 * char_ht[mask_saal5]))))

    # Warn about any unsupported species
    mask_supported = (mask_acru | mask_cofl2 | mask_nysy | mask_oxar | mask_qual |
                      mask_quco2 | mask_quma3 | mask_qumi | mask_quve | mask_saal5)
    if np.any(~mask_supported):
        unsupported = np.unique(spp[~mask_supported])
        print(f'Warning: BOLCHAR mortality model unavailable for species: {unsupported.tolist()}. '
              f'Mortality set to np.nan for those trees.')

    return float(Pm[0]) if scalar_input else Pm


def mort_crnsch(
    spp: Union[str, int, np.ndarray],
    dbh: Union[float, np.ndarray],
    ht: Union[float, np.ndarray],
    crown_depth: Union[float, np.ndarray],
    bark_thickness: Optional[Union[float, np.ndarray]] = None,
    fire_intensity: Optional[Union[float, np.ndarray]] = None,
    amb_t: Optional[Union[float, np.ndarray]] = None,
    flame_length: Optional[Union[float, np.ndarray]] = None,
    char_ht: Optional[Union[float, np.ndarray]] = None,
    scorch_ht: Optional[Union[float, np.ndarray]] = None,
    instand_ws: Optional[Union[float, np.ndarray]] = None,
    aspen_sev: str = 'low',
    tree_code_dict: dict = None
) -> Union[float, np.ndarray]:
    """
    FOFEM crown scorch mortality model (CRNSCH).

    Accepts a single tree (scalar inputs) or multiple trees (array inputs) of
    equal length. Species without a dedicated equation fall back to the general
    bark-thickness model (FOFEM Eq 1).

    :param spp: Species code(s) (str, int, or np.ndarray). A single string or
        int may be passed for a single tree. If int, codes are mapped to FOFEM
        species codes using ``tree_code_dict`` if provided; otherwise via the
        lookup in ``species_codes_lut.csv``. Unknown codes map to ``'UNK'``.
    :param dbh: Diameter at breast height (cm). Scalar or np.ndarray.
    :param ht: Total tree height (m). Scalar or np.ndarray.
    :param crown_depth: Crown depth (m). Used with scorch height to derive
        percent crown volume/length scorched. Scalar or np.ndarray.
    :param bark_thickness: Bark thickness (cm). Scalar or np.ndarray. Optional;
        if ``None``, estimated from species and DBH using the lookup table.
    :param fire_intensity: Surface fire intensity (kW/m). Scalar or np.ndarray.
        Optional; used to derive flame length, char height, and/or scorch height
        when those are not supplied.
    :param amb_t: Ambient air temperature (°C). Scalar or np.ndarray. Default
        25 °C. Used when estimating scorch height.
    :param flame_length: Flame length (m). Scalar or np.ndarray. Optional; if
        not provided, derived from ``fire_intensity`` or ``char_ht``.
    :param char_ht: Char height (m). Scalar or np.ndarray. Optional; if not
        provided, derived from ``flame_length``.
    :param scorch_ht: Scorch height (m). Scalar or np.ndarray. Optional; if not
        provided, estimated from ``fire_intensity``, ``amb_t``, and
        ``instand_ws``.
    :param instand_ws: In-stand windspeed (m/s). Scalar or np.ndarray.
        Default 1 m/s. Used when estimating scorch height.
    :param aspen_sev: Aspen severity class for equation selection; ``'low'`` or
        ``'high'``. Default ``'low'``.
    :param tree_code_dict: Optional dict mapping numeric species codes to FOFEM
        species code strings (e.g., ``{201: 'PIPO'}``).

    :return: Mortality probability (float in [0, 1]). Returns a scalar ``float``
        when all primary inputs (``spp``, ``dbh``, ``ht``, ``crown_depth``) are
        scalars, otherwise a 1D ``np.ndarray`` of the same length as the inputs.
    """
    # Detect whether the caller passed scalar inputs
    scalar_input = (_is_scalar(spp) and _is_scalar(dbh) and _is_scalar(ht)
                    and _is_scalar(crown_depth))

    # Verify tree_code_dict is dictionary if provided
    if tree_code_dict is not None and not isinstance(tree_code_dict, dict):
        print('tree_code_dict must be a dictionary, mapping numeric species codes to FOFEM species code strings.'
              'Using default species code mapping from species_codes_lut.csv.')
        tree_code_dict = None

    # Ensure all inputs are np.ndarrays (at least 1-D)
    spp = np.atleast_1d(np.array(spp))
    dbh = np.atleast_1d(np.asarray(dbh))
    ht = np.atleast_1d(np.asarray(ht))
    crown_depth = np.atleast_1d(np.asarray(crown_depth))
    if bark_thickness is not None:
        bark_thickness = np.atleast_1d(np.asarray(bark_thickness))
    if fire_intensity is not None:
        fire_intensity = np.atleast_1d(np.asarray(fire_intensity))
    if amb_t is None:
        amb_t = np.full(len(spp), 25.0)
    else:
        amb_t = np.atleast_1d(np.asarray(amb_t))
    if flame_length is not None:
        flame_length = np.atleast_1d(np.asarray(flame_length))
    if char_ht is not None:
        char_ht = np.atleast_1d(np.asarray(char_ht))
    if scorch_ht is not None:
        scorch_ht = np.atleast_1d(np.asarray(scorch_ht))
    if instand_ws is None:
        instand_ws = np.ones(len(spp))
    else:
        instand_ws = np.atleast_1d(np.asarray(instand_ws))

    # Map numeric spp to FOFEM_sppCD if needed
    if np.issubdtype(spp.dtype, np.integer):
        unique_num_cds = np.unique(spp)
        for num_cd in unique_num_cds:
            mask = spp == num_cd
            if tree_code_dict is None:
                spp[mask] = (SPP_CODES.loc[SPP_CODES['num_cd'] == num_cd, 'fofem_cd'].iloc[0]
                             if num_cd in SPP_CODES['num_cd'].values else 'UNK')
            else:
                spp[mask] = tree_code_dict.get(num_cd, 'UNK')
    else:
        spp = spp.astype(str)

    # Fill missing bark_thickness
    if bark_thickness is None:
        bark_thickness = calc_bark_thickness(spp, dbh)
    # Fill missing flame_length/char_ht
    if (flame_length is None) and (char_ht is None) and (fire_intensity is None):
        raise Exception('The CRNSCH mortality model requires either flame length, char height or surface fire intensity (kW/m) as an input')
    if (fire_intensity is not None) and (flame_length is None) and (char_ht is None):
        flame_length = calc_flame_length(fire_intensity, char_ht)
        char_ht = calc_char_ht(flame_length)
    if ((fire_intensity is not None) or (char_ht is not None)) and (flame_length is None):
        flame_length = calc_flame_length(fire_intensity, char_ht)
    if (flame_length is not None) and (char_ht is None):
        char_ht = calc_char_ht(flame_length)
    if scorch_ht is None:
        scorch_ht = calc_scorch_ht(fire_intensity, amb_t, instand_ws)

    # Calculate cvs, cls
    _, cvs, cls = calc_crown_length_vol_scorched(scorch_ht, ht, crown_depth)

    # Output array
    Pm = np.zeros(len(spp), dtype=float)

    # Masks for species
    mask_abco = np.isin(spp, ['ABCO', 'ABCOC'])
    mask_abgr = np.isin(spp, ['ABGR', 'ABGRI2', 'ABGRG', 'ABGRI', 'ABGRJ', 'ABLA', 'ABLAL'])
    mask_abma = spp == 'ABMA'
    mask_cade = np.isin(spp, ['CADE27', 'LIDE'])
    mask_laoc = spp == 'LAOC'
    mask_pial = np.isin(spp, ['PIAL', 'PICO', 'PICOL', 'PICOL2'])
    mask_spruce = np.isin(spp, ['PICSPP', 'PIMA', 'PIMAM4', 'PIPU', 'PIPUA', 'PIPUG3', 'PIAB', 'PIRU', 'PISI', 'PIGL'])
    mask_pien = np.isin(spp, ['PIEN', 'PIENE', 'PIENM', 'PIENM2'])
    mask_pila = spp == 'PILA'
    mask_pipa2 = spp == 'PIPA2'
    mask_pipo = np.isin(spp, [
        'PIPO', 'PIPOK', 'PIPOB', 'PIPOBK', 'PIPOB2', 'PIPOB3', 'PIPOB3K',
        'PIPOP', 'PIPOPK', 'PIPOP2', 'PIPOP2K', 'PIPOS', 'PIPOSK', 'PIPOS2', 'PIPOS2K',
        'PIJE', 'PIJEK'
    ])
    mask_pipo_bh = spp == 'PIPO_BH'
    mask_aspen = np.isin(spp, ['POTR12', 'POTR5', 'POTRA', 'POTRC2', 'POTRM', 'POTRR', 'POTRV'])
    mask_psme = np.isin(spp, ['PSME', 'PSMEF', 'PSMEM'])

    # FOFEM Eq 10 - White Fir
    if np.any(mask_abco):
        Pm[mask_abco] = 1 / (1 + np.exp(-(-3.5083 +
                                          (cls[mask_abco] * 0.0956) -
                                          (np.power(cls[mask_abco], 2) * 0.00184) +
                                          (np.power(cls[mask_abco], 3) * 0.000017))))
    # FOFEM Eq 11 - Subalpine/Grand Fir
    if np.any(mask_abgr):
        Pm[mask_abgr] = 1 / (1 + np.exp(-(-1.6950 +
                                          (cvs[mask_abgr] * 0.2071) -
                                          (np.power(cvs[mask_abgr], 2) * 0.0047) +
                                          (np.power(cvs[mask_abgr], 3) * 0.000035))))
    # FOFEM Eq 16 - Red Fir
    if np.any(mask_abma):
        Pm[mask_abma] = 1 / (1 + np.exp(-(-2.3085 + (np.power(cls[mask_abma], 3) * 0.000004059))))
    # FOFEM Eq 12 - Incense Cedar
    if np.any(mask_cade):
        Pm[mask_cade] = 1 / (1 + np.exp(-(-4.2466 + (np.power(cls[mask_cade], 3) * 0.000007172))))
    # FOFEM Eq 14 - Western Larch
    if np.any(mask_laoc):
        Pm[mask_laoc] = 1 / (1 + np.exp(-(-1.6594 + (cvs[mask_laoc] * 0.0327) - (dbh[mask_laoc] * 0.0489))))
    # FOFEM Eq 17 - Whitebark/Logepole Pine
    if np.any(mask_pial):
        Pm[mask_pial] = 1 / (1 + np.exp(-(-0.3268 +
                                          (cvs[mask_pial] * 0.1387) -
                                          (np.power(cvs[mask_pial], 2) * 0.0033) +
                                          (np.power(cvs[mask_pial], 3) * 0.000025) -
                                          (dbh[mask_pial] * 0.0266))))
    # FOFEM Eq 3 - All other spruce species
    if np.any(mask_spruce):
        dbh_in = dbh[mask_spruce] / 2.54
        _Pm = 1 / (1 + np.exp(-1.941 +
                              (6.316 * (1 - np.exp(-bark_thickness[mask_spruce] / 2.54))) -
                              (np.power(cvs[mask_spruce], 2) * 0.000535)))
        _Pm = np.where(dbh_in >= 1, np.maximum(_Pm, 0.8), _Pm)
        _Pm = np.where((dbh_in < 1) & (cls[mask_spruce] > 50), 1, _Pm)
        _Pm = np.where((dbh_in < 1) & (ht[mask_spruce] < (3 * 0.3048)), 1, _Pm)
        Pm[mask_spruce] = _Pm
    # FOFEM Eq 15 - Engelmann spruce
    if np.any(mask_pien):
        Pm[mask_pien] = 1 / (1 + np.exp(-(0.0845 + (cvs[mask_pien] * 0.0445))))
    # FOFEM Eq 18 - Sugar Pine
    if np.any(mask_pila):
        Pm[mask_pila] = 1 / (1 + np.exp(-(-2.0588 + (np.power(cls[mask_pila], 2) * 0.000814))))
    # FOFEM Eq 5 - Longleaf Pine
    if np.any(mask_pipa2):
        barkT = 0.435 + (0.031 * dbh[mask_pipa2])
        Pm[mask_pipa2] = np.where(
            scorch_ht[mask_pipa2] == 0,
            0,
            1 / (1 + np.exp(0.169 +
                            (5.136 * barkT) +
                            (14.429 * np.power(barkT, 2)) -
                            (0.348 * np.power(cvs[mask_pipa2] / 100, 2))))
        )
    # FOFEM Eq 19 - Ponderosa/Jeffrey Pine
    Pm[mask_pipo] = 1 / (1 + np.exp(-(-2.7103 + (np.power(cvs[mask_pipo], 3) * 0.000004093))))
    # FOFEM Eq 21 - Black Hills Ponderosa Pine
    if np.any(mask_pipo_bh):
        ht_bh = ht[mask_pipo_bh]
        dbh_bh = dbh[mask_pipo_bh]
        flame_bh = flame_length[mask_pipo_bh]
        cls_bh = cls[mask_pipo_bh]
        cbh_bh = ht_bh - crown_depth[mask_pipo_bh]
        scorch_ht_bh = scorch_ht[mask_pipo_bh]
        # Seedlings
        mask_seed = ht_bh <= 1.37
        Pm[mask_pipo_bh] = np.where(
            mask_seed,
            1 / (1 + np.exp(-(2.714 + (4.08 * flame_bh) - (3.63 * ht_bh)))),
            Pm[mask_pipo_bh]
        )
        # Saplings
        mask_sap = (ht_bh > 1.37) & (dbh_bh < 10.2)
        Pm[mask_pipo_bh] = np.where(
            mask_sap,
            1 / (1 + np.exp(-(-0.7661 + (2.7981 * flame_bh) - (1.2487 * ht_bh)))),
            Pm[mask_pipo_bh]
        )
        # Trees
        mask_tree = dbh_bh >= 10.2
        cls_tree = ((scorch_ht_bh - cbh_bh) / (ht_bh - cbh_bh)) * 100
        Pm[mask_pipo_bh] = np.where(
            mask_tree,
            1 / (1 + np.exp(-(1.104 - (dbh_bh * 0.156) + (0.013 * cls_tree) + (0.001 * dbh_bh * cls_tree)))),
            Pm[mask_pipo_bh]
        )
    # FOFEM Eq 4 - Aspen
    if np.any(mask_aspen):
        dbh_a = dbh[mask_aspen]
        char_ht_a = char_ht[mask_aspen]
        Pm[mask_aspen] = np.where(
            aspen_sev == 'low',
            1 / (1 + np.exp((0.251 * dbh_a) - (0.07 * char_ht_a * 12) - 4.407)),
            1 / (1 + np.exp((0.0858 * dbh_a) - (0.118 * char_ht_a * 12) - 2.157))
        )
    # FOFEM Eq 20 - Douglas-fir
    if np.any(mask_psme):
        Pm[mask_psme] = 1 / (1 + np.exp(-(-2.0346 +
                                          (cvs[mask_psme] * 0.0906) -
                                          (np.power(cvs[mask_psme], 2) * 0.0022) +
                                          (np.power(cvs[mask_psme], 3) * 0.000019))))
    # FOFEM Eq 1 - All other species
    mask_other = ~(mask_abco | mask_abgr | mask_abma | mask_cade | mask_laoc |
                   mask_pial | mask_spruce | mask_pien | mask_pila | mask_pipa2 |
                   mask_pipo | mask_pipo_bh | mask_aspen | mask_psme)
    if np.any(mask_other):
        dbh_in = dbh[mask_other] / 2.54
        _Pm = 1 / (1 + np.exp(-1.941 +
                              (6.316 * (1 - np.exp(-bark_thickness[mask_other] / 2.54))) -
                              (np.power(cvs[mask_other], 2) * 0.000535)))
        _Pm = np.where(dbh_in >= 1, _Pm, _Pm)
        _Pm = np.where((dbh_in < 1) & (cls[mask_other] > 50), 1, _Pm)
        _Pm = np.where((dbh_in < 1) & (ht[mask_other] < (3 * 0.3048)), 1, _Pm)
        Pm[mask_other] = _Pm

    return float(Pm[0]) if scalar_input else Pm


def mort_crcabe(
    spp: Union[str, int, np.ndarray],
    dbh: Union[float, np.ndarray],
    ht: Union[float, np.ndarray],
    crown_depth: Union[float, np.ndarray],
    ckr: Union[float, np.ndarray],
    scorch_ht: Union[float, np.ndarray],
    beetles: Union[bool, np.ndarray] = False,
    cvk: Optional[Union[float, np.ndarray]] = None,
    tree_code_dict: dict = None,
) -> Union[float, np.ndarray]:
    """
    FOFEM cambium kill / post-fire mortality model (CRCABE).

    Based on Hood and Lutes (2017). Accepts a single tree (scalar inputs) or
    multiple trees (array inputs) of equal length. Models are available for the
    12 conifer species listed below; unsupported species codes return ``np.nan``
    with a printed warning.

    Available species:
        - White fir              – ABCO, ABCOC
        - Grand/Subalpine fir    – ABGR, ABGRI2, ABGRG, ABGRI, ABGRJ, ABLA, ABLAL
        - Red fir                – ABMA
        - Incense Cedar          – CADE27, LIDE
        - Engelmann spruce       – PIEN, PIENE, PIENM, PIENM2
        - Western Larch          – LAOC
        - Douglas-fir            – PSME, PSMEF, PSMEM
        - Whitebark/Lodgepole pine – PIAL, PICO, PICOL, PICOL2
        - Sugar pine             – PILA
        - Ponderosa/Jeffrey pine – PIPO, PIPOK, PIPOB, PIPOBK, PIPOB2, PIPOB3,
          PIPOB3K, PIPOP, PIPOPK, PIPOP2, PIPOP2K, PIPOS, PIPOSK, PIPOS2,
          PIPOS2K, PIPO_BH, PIJE, PIJEK

    :param spp: Species code(s) (str, int, or np.ndarray). A single string or
        int may be passed for a single tree. If int, codes are mapped to FOFEM
        species codes using ``tree_code_dict`` if provided; otherwise via the
        lookup in ``species_codes_lut.csv``. Unknown codes map to ``'UNK'``.
    :param dbh: Diameter at breast height (cm). Scalar or np.ndarray.
    :param ht: Total tree height (m). Scalar or np.ndarray.
    :param crown_depth: Crown depth (m). Scalar or np.ndarray.
    :param ckr: Cambium Kill Rating (0–4), measured in the field. Scalar or
        np.ndarray.
    :param scorch_ht: Scorch height (m). Scalar or np.ndarray.
    :param beetles: Beetle attack status. A single ``bool`` (applied to all
        trees) or a boolean np.ndarray of the same length as ``spp``. Default
        ``False``. Species-specific ``atk`` factor values are assigned
        internally. Relevant beetle species: Ambrosia, Red turpentine, Mountain
        pine, Douglas-fir beetle, IPS.
    :param cvk: Percent total crown volume killed by bud kill (%). Used only
        for Ponderosa/Jeffrey pine; selects the ``PK`` (bud-kill) equation over
        the ``PP`` (scorch) equation when provided. Scalar or np.ndarray of the
        same length as ``spp``. Default ``None`` (uses scorch-based equation).
    :param tree_code_dict: Optional dict mapping numeric species codes to FOFEM
        species code strings (e.g., ``{201: 'PIPO'}``).

    :return: Mortality probability (float in [0, 1], or ``np.nan`` for
        unsupported species). Returns a scalar ``float`` when all primary inputs
        (``spp``, ``dbh``, ``ht``, ``crown_depth``, ``ckr``, ``scorch_ht``) are
        scalars, otherwise a 1D ``np.ndarray`` of the same length as the inputs.
    """
    # Detect whether the caller passed scalar inputs
    scalar_input = (_is_scalar(spp) and _is_scalar(dbh) and _is_scalar(ht)
                    and _is_scalar(crown_depth) and _is_scalar(ckr)
                    and _is_scalar(scorch_ht))

    # Verify tree_code_dict
    if tree_code_dict is not None and not isinstance(tree_code_dict, dict):
        print('tree_code_dict must be a dictionary, mapping numeric species codes to FOFEM species code strings. '
              'Using default species code mapping from species_codes_lut.csv.')
        tree_code_dict = None

    # Coerce all inputs to np.ndarray (at least 1-D)
    spp = np.atleast_1d(np.array(spp))
    dbh = np.atleast_1d(np.asarray(dbh))
    ht = np.atleast_1d(np.asarray(ht))
    crown_depth = np.atleast_1d(np.asarray(crown_depth))
    ckr = np.atleast_1d(np.asarray(ckr))
    scorch_ht = np.atleast_1d(np.asarray(scorch_ht))

    # Broadcast beetles to a per-tree boolean array
    beetles = np.broadcast_to(np.asarray(beetles, dtype=bool), spp.shape).copy()

    # Broadcast cvk to a per-tree float array (NaN where not provided)
    if cvk is None:
        cvk_arr = np.full(spp.shape, np.nan)
    else:
        cvk_arr = np.broadcast_to(np.asarray(cvk, dtype=float), spp.shape).copy()

    # Map numeric spp codes to FOFEM string codes if needed
    if np.issubdtype(spp.dtype, np.integer):
        unique_num_cds = np.unique(spp)
        for num_cd in unique_num_cds:
            mask = spp == num_cd
            if tree_code_dict is None:
                spp[mask] = (SPP_CODES.loc[SPP_CODES['num_cd'] == num_cd, 'fofem_cd'].iloc[0]
                             if num_cd in SPP_CODES['num_cd'].values else 'UNK')
            else:
                spp[mask] = tree_code_dict.get(num_cd, 'UNK')
    else:
        spp = spp.astype(str)

    # Calculate crown volume scorched (cvs, %) and crown length scorched (cls, %)
    _, cvs, cls = calc_crown_length_vol_scorched(scorch_ht, ht, crown_depth)

    # Output array – NaN by default (unsupported species remain NaN)
    Pm = np.full(len(spp), np.nan)

    # --- Species masks ---
    mask_abco = np.isin(spp, ['ABCO', 'ABCOC'])
    mask_abgr = np.isin(spp, ['ABGR', 'ABGRI2', 'ABGRG', 'ABGRI', 'ABGRJ', 'ABLA', 'ABLAL'])
    mask_abma = spp == 'ABMA'
    mask_cade = np.isin(spp, ['CADE27', 'LIDE'])
    mask_pien = np.isin(spp, ['PIEN', 'PIENE', 'PIENM', 'PIENM2'])
    mask_laoc = spp == 'LAOC'
    mask_psme = np.isin(spp, ['PSME', 'PSMEF', 'PSMEM'])
    mask_pial = np.isin(spp, ['PIAL', 'PICO', 'PICOL', 'PICOL2'])
    mask_pila = spp == 'PILA'
    mask_pipo = np.isin(spp, [
        'PIPO', 'PIPOK', 'PIPOB', 'PIPOBK', 'PIPOB2', 'PIPOB3', 'PIPOB3K',
        'PIPOP', 'PIPOPK', 'PIPOP2', 'PIPOP2K', 'PIPOS', 'PIPOSK', 'PIPOS2', 'PIPOS2K', 'PIPO_BH',
        'PIJE', 'PIJEK'
    ])

    # FOFEM Eq WF - White Fir (ambrosia beetle; atk: attacked=1, unattacked=-1)
    if np.any(mask_abco):
        atk = np.where(beetles[mask_abco], 1, -1).astype(float)
        Pm[mask_abco] = 1 / (1 + np.exp(
            -(-3.5964 + (np.power(cls[mask_abco], 3) * 0.00000628) +
              (ckr[mask_abco] * 0.3019) + (dbh[mask_abco] * 0.019) + (atk * 0.5209))))

    # FOFEM Eq SF - Grand Fir and Subalpine Fir
    if np.any(mask_abgr):
        Pm[mask_abgr] = 1 / (1 + np.exp(
            -(-2.6036 + (np.power(cvs[mask_abgr], 3) * 0.000004587) + (ckr[mask_abgr] * 1.3554))))

    # FOFEM Eq RF - Red Fir
    if np.any(mask_abma):
        Pm[mask_abma] = 1 / (1 + np.exp(
            -(-4.7515 + (np.power(cls[mask_abma], 3) * 0.000005989) + (ckr[mask_abma] * 1.0668))))

    # FOFEM Eq IC - Incense Cedar
    if np.any(mask_cade):
        Pm[mask_cade] = 1 / (1 + np.exp(
            -(-5.6465 + (np.power(cls[mask_cade], 3) * 0.000007274) + (ckr[mask_cade] * 0.5428))))

    # FOFEM Eq ES - Engelmann Spruce
    if np.any(mask_pien):
        Pm[mask_pien] = 1 / (1 + np.exp(
            -(-2.9791 + (cvs[mask_pien] * 0.0405) + (ckr[mask_pien] * 1.1596))))

    # FOFEM Eq WL - Western Larch
    if np.any(mask_laoc):
        Pm[mask_laoc] = 1 / (1 + np.exp(
            -(-3.8458 + (np.power(cvs[mask_laoc], 2) * 0.0004) + (ckr[mask_laoc] * 0.6266))))

    # FOFEM Eq DF - Douglas-fir (Douglas-fir beetle; atk: attacked=1, unattacked=0)
    if np.any(mask_psme):
        atk = np.where(beetles[mask_psme], 1, 0).astype(float)
        Pm[mask_psme] = 1 / (1 + np.exp(
            -(-1.8912 + (cvs[mask_psme] * 0.07) -
              (np.power(cvs[mask_psme], 2) * 0.0019) +
              (np.power(cvs[mask_psme], 3) * 0.000018) +
              (ckr[mask_psme] * 0.5840) - (dbh[mask_psme] * 0.031) -
              (atk * 0.7959) + (dbh[mask_psme] * atk * 0.0492))))

    # FOFEM Eq WP - Whitebark Pine and Lodgepole Pine
    if np.any(mask_pial):
        Pm[mask_pial] = 1 / (1 + np.exp(
            -(-1.4059 + (np.power(cvs[mask_pial], 3) * 0.000004459) +
              (np.power(ckr[mask_pial], 2) * 0.2843) - (dbh[mask_pial] * 0.0485))))

    # FOFEM Eq SP - Sugar Pine (red turpentine / mountain pine beetle; atk: attacked=1, unattacked=-1)
    if np.any(mask_pila):
        atk = np.where(beetles[mask_pila], 1, -1).astype(float)
        Pm[mask_pila] = 1 / (1 + np.exp(
            -(-2.7598 + (np.power(cls[mask_pila], 2) * 0.000642) +
              (np.power(ckr[mask_pila], 3) * 0.0386) + (atk * 0.8485))))

    # FOFEM Eq PP / PK - Ponderosa / Jeffrey Pine
    # (mountain pine, red turpentine, or ips beetle; atk: attacked=1, unattacked=0)
    # Uses PK (cvk-based) equation where cvk is provided, otherwise PP (scorch-based).
    if np.any(mask_pipo):
        atk = np.where(beetles[mask_pipo], 1, 0).astype(float)
        cvk_sub = cvk_arr[mask_pipo]
        has_cvk = ~np.isnan(cvk_sub)
        _Pm = np.empty(int(np.sum(mask_pipo)))
        # PP equation (scorch-based)
        _Pm[~has_cvk] = 1 / (1 + np.exp(
            -(-4.1914 + (np.power(cvs[mask_pipo][~has_cvk], 2) * 0.000376) +
              (ckr[mask_pipo][~has_cvk] * 0.5130) + (atk[~has_cvk] * 1.5873))))
        # PK equation (cvk/bud-kill-based)
        _Pm[has_cvk] = 1 / (1 + np.exp(
            -(-3.5729 + (np.power(cvk_sub[has_cvk], 2) * 0.000567) +
              (ckr[mask_pipo][has_cvk] * 0.4573) + (atk[has_cvk] * 1.6075))))
        Pm[mask_pipo] = _Pm

    # Warn about any unsupported species
    mask_supported = (mask_abco | mask_abgr | mask_abma | mask_cade | mask_pien |
                      mask_laoc | mask_psme | mask_pial | mask_pila | mask_pipo)
    mask_unsupported = ~mask_supported
    if np.any(mask_unsupported):
        unsupported = np.unique(spp[mask_unsupported])
        print(f'Warning: CRCABE mortality model unavailable for species: {unsupported.tolist()}. '
              f'Mortality set to np.nan for those trees.')

    return float(Pm[0]) if scalar_input else Pm

