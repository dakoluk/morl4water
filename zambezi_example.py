import numpy as np
import mo_gymnasium
import examples.zambezi_river_simulation 


water_management_system = mo_gymnasium.make('zambezi-v0')


def run_zambezi():
    # Reset the environment
    obs, info = water_management_system.reset()
    print(f'Initial Observation: {obs}')

    final_truncated = False
    final_terminated = False
    
    # Example: running the simulation for a defined number of timesteps
    for t in range(240):  
        if not final_terminated and not final_truncated:
            # Sample random action from the action space
            action = water_management_system.action_space.sample()
            print(f'Action for timestep: {t}: {action}')

            # Take the sampled action in the environment
            (
                final_observation,
                final_reward,
                final_terminated,
                final_truncated,
                final_info
            ) = water_management_system.step(action)

            # Log observations and rewards
            print(f'Observation: {final_observation}')
            print(f'Reward: {final_reward}')
            print(f'Info: {final_info}')
        else:
            break
    
    return final_observation


if __name__ == "__main__":
    run_zambezi()