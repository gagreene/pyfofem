# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13, 12:00:00 2025

@author: Gregory A. Greene
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import numpy as np
from numpy import ma as mask
from typing import Union, Optional

class FOFEM(object):
    def __init__(self):
        self.species = None  # Tree species code
        self.dbh = None  # Diameter at breast height (cm)
        self.ht = None  # Tree height (m)
        self.crown_depth = None  # Crown depth (m)
        self.bark_thickness = None  # Bark thickness (cm)
        self.fire_intensity = None  # Fire intensity (kW/m)
        self.ambient_temp = None  # Ambient air temperature (°C)
        self.flame_length = None  # Flame length (m)
        self.char_ht = None  # Char height (m)
        self.scorch_ht = None  # Scorch height (m)
        self.in_stand_ws = None  # In-stand wind speed (km/h)
        self.aspen_severity = None  # Aspen mortality severity (1-5)

        # Load species codes lookup table
        import os
        from pandas import read_csv
        self.spp_codes = read_csv(os.path.join(os.path.dirname(__file__), 'species_codes_lut.csv'))

        # Initialize flags
        self.fofem_return_array = False  # Flag to indicate if output should be an array
        self.fofem_inputs_verified = False  # Flag to indicate if inputs have been verified
        return
    def _check_array_fofem(self):
        return
    def _verify_inputs_fofem(self):
        # Verify that required inputs are provided
        if (self.flame_length is None) and (self.char_ht is None) and (self.fire_intensity is None):
            raise Exception('The CRNSCH mortality model requires either flame length, char height or '
                            'surface fire intensity (kW/m) as an input')

        if not isinstance(self.species, (int, str, np.ndarray)):
            raise TypeError('species must be either int, string, or numpy ndarray data types')
        elif isinstance(self.species, str):
            # If self.species is not in self.sepecies_codes "species_cd" or "fofem_cd" columns, raise error
            if (self.species not in self.spp_codes['species_cd'].values) and \
               (self.species not in self.spp_codes['fofem_cd'].values):
                raise ValueError(f'species code {self.species} not found in lookup table.')
            else:
                self.species = self.spp_codes.loc[
                    (self.spp_codes['species_cd'] == self.species) |
                    (self.spp_codes['fofem_cd'] == self.species),
                    'num_cd'
                ].values[0]
        elif isinstance(self.species, np.ndarray):
            # Cover str case
            if '<U' in str(self.species.dtype):
                # Alphabetic: match against species_cd and fofem_cd, mask non-matches
                mask_valid = np.isin(self.species, self.spp_codes['species_cd'].values) | \
                             np.isin(self.species, self.spp_codes['fofem_cd'].values)
                converted_species = np.full(self.species.shape, np.nan)
                for idx, sp in enumerate(self.species):
                    if mask_valid[idx]:
                        num_cd = self.spp_codes.loc[
                            (self.spp_codes['species_cd'] == sp) | (self.spp_codes['fofem_cd'] == sp),
                            'num_cd'
                        ].values[0]
                        converted_species[idx] = num_cd
                self.species = mask.array(converted_species, mask=~mask_valid)
            # Cover int case
            elif np.issubdtype(self.species.dtype, np.number):
                # Numeric: match against num_cd, mask non-matches
                mask_valid = np.isin(self.species, self.spp_codes['num_cd'].values)
                converted_species = np.where(mask_valid, self.species, np.nan)
                self.species = mask.array(converted_species, mask=~mask_valid)
            else:
                raise TypeError('species array must be of string or numeric dtype')
        else:  # int case
            #Ensure code is in lookup table
            if self.species not in self.spp_codes['num_cd'].values:
                raise ValueError(f'species code {self.species} not found in species codes lookup table.')
            self.species = mask.array([self.species], mask=np.isnan([self.species]))
        return

    def initialize_fofem(self,
                         species: Union[int, str, np.ndarray],
                         dbh: Union[float, np.ndarray],
                         ht: Union[float, np.ndarray],
                         crown_depth: Union[float, np.ndarray],
                         bark_thickness: Union[float, np.ndarray],
                         fire_intensity: Union[float, np.ndarray],
                         ambient_temp: Union[float, np.ndarray],
                         flame_length: Union[float, np.ndarray],
                         char_ht: Union[float, np.ndarray],
                         scorch_ht: Union[float, np.ndarray],
                         in_stand_ws: Union[float, np.ndarray],
                         aspen_severity: Union[int, np.ndarray]) -> None:
        """
        Function to initialize FOFEM class inputs.

        :param species: Tree species code
        :param dbh: Diameter at breast height (cm)
        :param ht: Tree height (m)
        :param crown_depth: Crown depth (m)
        :param bark_thickness: Bark thickness (cm)
        :param fire_intensity: Fire intensity (kW/m)
        :param ambient_temp: Ambient air temperature (°C)
        :param flame_length: Flame length (m)
        :param char_ht: Char height (m)
        :param scorch_ht: Scorch height (m)
        :param in_stand_ws: In-stand wind speed (km/h)
        :param aspen_severity: Aspen mortality severity (1-5)
        :return: None
        """
        # Initialize inputs
        self.species = species
        self.dbh = dbh
        self.ht = ht
        self.crown_depth = crown_depth
        self.bark_thickness = bark_thickness
        self.fire_intensity = fire_intensity
        self.ambient_temp = ambient_temp
        self.flame_length = flame_length
        self.char_ht = char_ht
        self.scorch_ht = scorch_ht
        self.in_stand_ws = in_stand_ws
        self.aspen_severity = aspen_severity

        return

    def calc_mortality_crnsch(self):
        return

    def run_mort_crnsch(self):
        if not self.fofem_inputs_verified:
            self._check_array_fofem()
            self._verify_inputs_fofem()

        self.calc_mortality_crnsch()
        return