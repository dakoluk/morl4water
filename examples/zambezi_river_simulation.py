import numpy as np
from pathlib import Path
from gymnasium.spaces import Box
from gymnasium.wrappers import TimeLimit
from core.envs.water_management_system import WaterManagementSystem
from core.models.reservoir import Reservoir
from core.models.flow import Flow, Inflow
from core.models.objective import Objective
from core.models.power_plant import PowerPlant
from core.models.irrigation_district import IrrigationDistrict
from core.models.catchment import Catchment
from core.wrappers.transform_action import ReshapeArrayAction
from datetime import datetime
from dateutil.relativedelta import relativedelta
from gymnasium.envs.registration import register

# Define data directory for files
data_directory = Path(__file__).parents[1] / "examples" / "data" / "zambezi_river"

# Environment registration
register(
    id='zambezi-v0',
    entry_point='examples.zambezi_river_simulation:create_zambezi_river_env',
)

def create_zambezi_river_env(custom_obj=None, render_mode=None) -> WaterManagementSystem:
    # Reservoirs
    Kafue_reservoir = Reservoir(
        "Kafue",
        max_capacity=1000000000.0,
        max_action=[5000],
        integration_timestep_size=relativedelta(minutes=240),
        objective_function=Objective.no_objective,
        stored_water=500000000.0,
        evap_rates=np.loadtxt(data_directory / "reservoirs" / "evap_Kafue.txt"),
        evap_rates_timestep_size=relativedelta(months=1),
        storage_to_minmax_rel=np.loadtxt(data_directory / "reservoirs" / "store_min_max_release_Kafue.txt"),
        storage_to_level_rel=np.loadtxt(data_directory / "reservoirs" / "store_level_rel_Kafue.txt"),
        storage_to_surface_rel=np.loadtxt(data_directory / "reservoirs" / "store_surf_rel_Kafue.txt"),
    )
    
    ItezhiTezhi_reservoir = Reservoir(
        "ItezhiTezhi",
        max_capacity=450000000.0,
        max_action=[2000],
        integration_timestep_size=relativedelta(minutes=240),
        objective_function=Objective.no_objective,
        stored_water=200000000.0,
        evap_rates=np.loadtxt(data_directory / "reservoirs" / "evap_ItezhiTezhi.txt"),
        evap_rates_timestep_size=relativedelta(months=1),
        storage_to_minmax_rel=np.loadtxt(data_directory / "reservoirs" / "store_min_max_release_ItezhiTezhi.txt"),
        storage_to_level_rel=np.loadtxt(data_directory / "reservoirs" / "store_level_rel_ItezhiTezhi.txt"),
        storage_to_surface_rel=np.loadtxt(data_directory / "reservoirs" / "store_surf_rel_ItezhiTezhi.txt"),
    )
    
    Kariba_reservoir = Reservoir(
        "Kariba",
        max_capacity=1800000000.0,
        max_action=[7000],
        integration_timestep_size=relativedelta(minutes=240),
        objective_function=Objective.no_objective,
        stored_water=900000000.0,
        evap_rates=np.loadtxt(data_directory / "reservoirs" / "evap_Kariba.txt"),
        evap_rates_timestep_size=relativedelta(months=1),
        storage_to_minmax_rel=np.loadtxt(data_directory / "reservoirs" / "store_min_max_release_Kariba.txt"),
        storage_to_level_rel=np.loadtxt(data_directory / "reservoirs" / "store_level_rel_Kariba.txt"),
        storage_to_surface_rel=np.loadtxt(data_directory / "reservoirs" / "store_surf_rel_Kariba.txt"),
    )
    
    CahoraBassa_reservoir = Reservoir(
        "CahoraBassa",
        max_capacity=900000000.0,
        max_action=[4000],
        integration_timestep_size=relativedelta(minutes=240),
        objective_function=Objective.no_objective,
        stored_water=300000000.0,
        evap_rates=np.loadtxt(data_directory / "reservoirs" / "evap_CahoraBassa.txt"),
        evap_rates_timestep_size=relativedelta(months=1),
        storage_to_minmax_rel=np.loadtxt(data_directory / "reservoirs" / "store_min_max_release_CahoraBassa.txt"),
        storage_to_level_rel=np.loadtxt(data_directory / "reservoirs" / "store_level_rel_CahoraBassa.txt"),
        storage_to_surface_rel=np.loadtxt(data_directory / "reservoirs" / "store_surf_rel_CahoraBassa.txt"),
    )

    # Inflows
    Kafue_inflow = Inflow(
        "Kafue_inflow",
        Kafue_reservoir,
        float("inf"),
        np.loadtxt(data_directory / "catchments" / "InflowKafue.txt"),
    )

    ItezhiTezhi_inflow = Inflow(
        "ItezhiTezhi_inflow",
        ItezhiTezhi_reservoir,
        float("inf"),
        np.loadtxt(data_directory / "catchments" / "InflowItezhiTezhi.txt"),
    )

    # Flows
    Kafue_flow = Flow("Kafue_flow", [Kafue_reservoir], None, float("inf"))

    ItezhiTezhi_flow = Flow("ItezhiTezhi_flow", [ItezhiTezhi_reservoir], None, float("inf"))

    # Irrigation Districts
    IrrigationDistrict_1 = IrrigationDistrict(
        "IrrigationDistrict_1",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_1.txt"),
        Objective.deficit_minimised,
        "irrigation_1_deficit_minimised",
        normalize_objective=100.0  # Example normalization factor
    )

    IrrigationDistrict_2 = IrrigationDistrict(
        "IrrigationDistrict_2",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_2.txt"),
        Objective.deficit_minimised,
        "irrigation_2_deficit_minimised",
        normalize_objective=150.0  # Example normalization factor
    )

    # Water management system creation
    water_management_system = WaterManagementSystem(
        water_systems=[
            Kafue_inflow,
            Kafue_reservoir,
            Kafue_flow,
            ItezhiTezhi_inflow,
            ItezhiTezhi_reservoir,
            ItezhiTezhi_flow,
            Kariba_reservoir,
            // Add additional components here
            CahoraBassa_reservoir,
            IrrigationDistrict_1,
            IrrigationDistrict_2,
        ],
        rewards={
            "kafue_water_yield": 0,
            "itezhi_tezhi_water_yield": 0,
            "kariba_water_yield": 0,
            "cahora_bassa_water_yield": 0,
        },
        start_date=datetime(2025, 1, 1),
        timestep_size=relativedelta(months=1),
        seed=42,
        custom_obj=custom_obj,
        add_timestamp='m',
    )

    # Wrap the environment action space
    water_management_system = ReshapeArrayAction(water_management_system)
    water_management_system = TimeLimit(water_management_system, max_episode_steps=240)

    return water_management_system
