import numpy as np
from typing import Generator, Optional
# from typing import Literal #Can't seem to import??

# No longer used (was initial random waypoint creation but switched to 
# generator for more flexibility and reduced memory usage). Left here in-case I want to use it, but commented out :) 

# def generate_vertical_waypoints():
#     """Generate a series of waypoints straight up from the rocket launch position
#     assuming launch position is 0, 0, 0 (always is so far)
    
#     Returns array of coordiantes in the form of:
#     [[x, y, z]
#      [x, y, z]
#      ...
#      [x, y, z]]
#     """

#     waypoints = []
#     max_height = 800.0  # Maximum height [m]
#     n_points = 8        # Number of waypoints
#     for i in range(n_points):
#         t_norm = i / (n_points - 1)
#         x, y, z = 0.0, 0.0, max_height * (t_norm**0.8)
#         waypoints.append([x, y, z])
#     return np.array(waypoints)

# def generate_random_waypoints(max_height=800.0, n_points=8, max_horizontal_step=50.0, seed=None):
#     """Generate a series of random waypoints with even z spacing.
    
#     Parameters:
#     -----------
#     max_height : float
#         Maximum altitude in meters
#     n_points : int
#         Number of waypoints to generate
#     max_horizontal_step : float
#         Maximum horizontal displacement per step (controls how far from vertical)
#     seed : int, optional
#         Random seed for reproducibility
    
#     Returns:
#     --------
#     numpy.ndarray
#         Array of waypoints with shape (n_points, 3)
#     """
#     if seed is not None:
#         np.random.seed(seed)
    
#     waypoints = []
#     z_spacing = max_height / (n_points - 1)
    
#     # Start at origin
#     current_pos = np.array([0.0, 0.0, 0.0])
#     waypoints.append(current_pos.copy())
    
#     # Previous direction vector (initially pointing up)
#     prev_direction = np.array([0.0, 0.0, 1.0])
    
#     for i in range(1, n_points):
#         z = i * z_spacing
        
#         # Calculate maximum horizontal distance from previous point
#         # Based on 45 degree constraint from vertical
#         dz = z - current_pos[2]
#         max_horizontal_distance = dz  # tan(45°) = 1
        
#         # Generate random horizontal displacement
#         # Scale it down to ensure we stay within the 45 degree cone
#         angle = np.random.uniform(0, 2 * np.pi)
#         distance = np.random.uniform(0, min(max_horizontal_step, max_horizontal_distance * 0.8))
        
#         dx = distance * np.cos(angle)
#         dy = distance * np.sin(angle)
        
#         # Proposed new position
#         new_pos = np.array([current_pos[0] + dx, current_pos[1] + dy, z])
        
#         # Calculate new direction vector
#         new_direction = new_pos - current_pos
#         new_direction = new_direction / np.linalg.norm(new_direction)
        
#         # Check angle constraint with previous direction (avoid sharp turns)
#         dot_product = np.dot(prev_direction, new_direction)
#         angle_between = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
#         # If turn is too sharp (more than 60 degrees), reduce horizontal displacement
#         max_turn_angle = np.radians(60)
#         if angle_between > max_turn_angle:
#             # Scale down the horizontal displacement
#             scale_factor = 0.3
#             dx *= scale_factor
#             dy *= scale_factor
#             new_pos = np.array([current_pos[0] + dx, current_pos[1] + dy, z])
#             new_direction = new_pos - current_pos
#             new_direction = new_direction / np.linalg.norm(new_direction)
        
#         waypoints.append(new_pos)
#         current_pos = new_pos
#         prev_direction = new_direction
    
#     return np.array(waypoints)

def generate_vertical_waypoints(
        max_height: float=800.0, 
        n_points: int=8
        ) -> Generator[np.ndarray, None, None]:
    """Generate vertical waypoints straight up from the rocket launch position
    assuming launch position is 0, 0, 0 (always is so far)
    
    Returns array of coordiantes in the form of:
    [[x, y, z]
     [x, y, z]
     ...
     [x, y, z]]
     """
    for i in range(n_points):
        t_norm = i / (n_points - 1)
        x, y, z = 0.0, 0.0, max_height * (t_norm**0.8)
        yield np.array([x, y, z])

def generate_random_waypoints(
        max_height: float=800.0, 
        n_points: int=8, 
        max_horizontal_step: float=50.0, 
        seed: Optional[int]=None
        ) -> Generator[np.ndarray, None, None]:
    """Generate a series of random waypoints with even z spacing.
    
    Parameters:
    -----------
    max_height : float
        Maximum altitude in meters
    n_points : int
        Number of waypoints to generate
    max_horizontal_step : float
        Maximum horizontal displacement per step (controls how far from vertical)
    seed : int, optional
        Random seed for reproducibility
    
    Yields:
    -------
    numpy.ndarray
        Individual waypoint with shape (3,) for [x, y, z]
        resulting in an array of coordiantes in the form of:
    [[x, y, z]
     [x, y, z]
     ...
     [x, y, z]]

    Example usage:
    --------------

    Iterate through waypoints one at a time:
    for waypoint in generate_random_waypoints(max_height=800.0, n_points=8, seed=42):
        print(waypoint)

    Or collect all waypoints into an array:
    waypoints = np.array(list(generate_random_waypoints(max_height=800.0, n_points=8, seed=42)))
    """
    if seed is not None:
        np.random.seed(seed)
    
    z_spacing = max_height / (n_points - 1)
    
    # Start at origin
    current_pos = np.array([0.0, 0.0, 0.0])
    yield current_pos.copy()
    
    # Previous direction vector (initially pointing up)
    prev_direction = np.array([0.0, 0.0, 1.0])
    
    for i in range(1, n_points):
        z = i * z_spacing
        
        # Calculate maximum horizontal distance from previous point
        # Based on 45 degree constraint from vertical
        dz = z - current_pos[2]
        max_horizontal_distance = dz  # tan(45°) = 1
        
        # Generate random horizontal displacement
        # Scale it down to ensure we stay within the 45 degree cone
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, min(max_horizontal_step, max_horizontal_distance * 0.8))
        
        dx = distance * np.cos(angle)
        dy = distance * np.sin(angle)
        
        # Proposed new position
        new_pos = np.array([current_pos[0] + dx, current_pos[1] + dy, z])
        
        # Calculate new direction vector
        new_direction = new_pos - current_pos
        new_direction = new_direction / np.linalg.norm(new_direction)
        
        # Check angle constraint with previous direction (avoid sharp turns)
        dot_product = np.dot(prev_direction, new_direction)
        angle_between = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # If turn is too sharp (more than 60 degrees), reduce horizontal displacement
        max_turn_angle = np.radians(60)
        if angle_between > max_turn_angle:
            # Scale down the horizontal displacement
            scale_factor = 0.3
            dx *= scale_factor
            dy *= scale_factor
            new_pos = np.array([current_pos[0] + dx, current_pos[1] + dy, z])
            new_direction = new_pos - current_pos
            new_direction = new_direction / np.linalg.norm(new_direction)
        
        yield new_pos
        current_pos = new_pos
        prev_direction = new_direction

def generate_waypoints(
        waypoint_type: str='vertical', 
        max_height: float=800.0, 
        n_points: int=8, 
        max_horizontal_step: float=50.0, 
        seed: Optional[int]=None
        ) -> Generator[np.ndarray, None, None]:
    """Generate waypoints based on the specified type.
    
    Parameters:
    -----------
    waypoint_type : str
        Type of waypoints to generate: 'vertical' or 'random'
    max_height : float
        Maximum altitude in meters
    n_points : int
        Number of waypoints to generate
    max_horizontal_step : float
        Maximum horizontal displacement per step (only for 'random' type)
    seed : int, optional
        Random seed for reproducibility (only for 'random' type)
    
    Yields:
    -------
    numpy.ndarray
        Individual waypoint with shape (3,) for [x, y, z]
    """

    # Dictionary to map type to generator function
    generators = {
        'vertical': lambda: generate_vertical_waypoints(max_height, n_points),
        'random': lambda: generate_random_waypoints(max_height, n_points, max_horizontal_step, seed)
    }
    
    # Get the appropriate generator
    if waypoint_type not in generators:
        raise ValueError(f"Invalid waypoint_type '{waypoint_type}'. Must be one of {list(generators.keys())}")
    
    # Yield from the selected generator
    yield from generators[waypoint_type]()

if __name__ == "__main__":
    print(generate_vertical_waypoints())
    # waypoints = generate_random_waypoints(max_height=800.0, n_points=8, seed=42)
    print(np.array(list(generate_random_waypoints(max_height=800.0, n_points=8, seed=42))))
    for waypoint in generate_random_waypoints(max_height=800.0, n_points=8, seed=42):
        print(waypoint)

    # Generate vertical waypoints
    print(f"Generating vertical waypoints {10*'-'}")
    for waypoint in generate_waypoints(waypoint_type='vertical', max_height=800.0, n_points=8):
        print(waypoint)

    # Generate random waypoints
    print(f"Generating random waypoints {10*'-'}")
    for waypoint in generate_waypoints(waypoint_type='random', max_height=800.0, n_points=8, seed=42):
        print(waypoint)

    # Or collect into an array
    # waypoints = np.array(list(generate_waypoints(waypoint_type='random', max_height=800.0, n_points=8, seed=42)))