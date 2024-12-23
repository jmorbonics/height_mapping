import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from datetime import datetime
from scipy.stats import norm
from typing import Optional, List, Dict, Tuple, Any


class Length:
    def __init__(self, x: float, y: float):
        self._x = x
        self._y = y

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y
    
    def __repr__(self) -> str:
        return f"Length(x-dir={self.x}, y-dir={self.y})"
    

class Position:
    def __init__(self, x: float, y: float):
        self._x = x
        self._y = y

    def __add__(self, other: 'Position') -> 'Position':
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Position') -> 'Position':
        return Position(self.x - other.x, self.y - other.y)

    def __repr__(self) -> str:
        return f"Position(x={self.x}, y={self.y})"

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y


class Gridmap:
    def __init__(self):
        self.length: Optional[Tuple[float, float]] = None
        self.resolution: Optional[float] = None
        self.position: Optional[Tuple[float, float]] = None
        self.size: Optional[Tuple[int, int]] = None
        self.layers: Dict[str, np.ndarray] = {}


    def set_geometry(self, length: Length, resolution: float, position: Position):
        """ 
        length: (x-direction, y-direction)
        resolution: (m/cell)
        position: (x, y) of origin
        """
        # round up length to have perfect grid shape
        x_length = length.x
        y_length = length.y
        if x_length % resolution != 0:
            x_length += resolution - x_length % resolution
        if y_length % resolution != 0:
            y_length += resolution - y_length % resolution


        self.length = Length(x_length, y_length)
        self.resolution: float = resolution
        self.position = Position(position.x, position.y)

        # size of grid map: [rows,cols]
        self.size: Tuple[int,int] = (int(self.length.y / self.resolution), int(self.length.x / self.resolution))


    def get_length(self) -> Length:
        return self.length


    def get_resolution(self) -> float:
        return self.resolution
    

    def get_position(self) -> Position:
        return self.position
    
    
    def get_num_rows(self) -> int:
        return self.size[0]
    

    def get_num_cols(self) -> int:
        return self.size[1]
    

    def is_inside(self, position: Position) -> bool:
        """
        position: (x, y) position
        Returns if the position is inside the grid map
        """
        origin = self.get_position()
        if (
            position.x >= (origin.x - self.length.x / 2) and
            position.x <= (origin.x + self.length.x / 2) and
            position.y >= (origin.y - self.length.y / 2) and
            position.y <= (origin.y + self.length.y / 2)):
            return True
        else:
            return False
        

    def get_index(self, position: Position) -> Tuple[int, int]:
        """
        position: (x, y) position
        Gets the index in the grid map based on the position [row,col]
        """
        if not self.is_inside(position):
            raise ValueError("Position is outside of grid map")
        
        top_left = Position(self.position.x - self.length.x / 2, self.position.y + self.length.y / 2)
        relative_position = position - top_left
        index_col = int(relative_position.x / self.resolution)
        index_row = abs(int(relative_position.y / self.resolution))
        
        # temp fix for making upper bounds of length inclusive
        if index_row >= self.size[0]:
            index_row = self.size[0] - 1
        if index_col >= self.size[1]:
            index_col = self.size[1] - 1

        return index_row, index_col
    
    def get_indexes(self, positions: np.ndarray) -> np.ndarray:
        """
        positions: list of (x, y) positions
        Gets the indexes in the grid map based on the positions
        """
        if not self.is_inside(positions):
            raise ValueError("Position is outside of grid map")
        
        top_left = Position(self.position.x - self.length.x / 2, self.position.y + self.length.y / 2)
        relative_positions = positions - top_left

        indexes_col = int(relative_positions.x / self.resolution)
        indexes_row = abs(int(relative_positions.y / self.resolution))
        
        # temp fix for making upper bounds of length inclusive
        if index_row >= self.size[0]:
            index_row = self.size[0] - 1
        if index_col >= self.size[1]:
            index_col = self.size[1] - 1

        return np.array(list(zip(indexes_row, indexes_col)))


    def get_cell_position(self, index) -> Position:
        """
        index: (row, col) index
        Gets the position of the cell in the grid map based on the index
        """
        if index[0] < 0 or index[0] >= self.size[0] or index[1] < 0 or index[1] >= self.size[1]:
            raise ValueError("Index is outside of grid map")
        
        top_left = Position(self.position.x - self.length.x / 2, self.position.y + self.length.y / 2)
        
        return Position(top_left.x + (index[0] + 0.5) * self.resolution, top_left.y - (index[1] + 0.5) * self.resolution)


    def at(self, layer: str, index: Tuple[int, int]):
        """
        layer: name of layer
        index: (x, y) index
        Returns the value of the layer at the index
        """
        return self.layers[layer][index[0]][index[1]]


    def atPosition(self, layer: str, position: Position):
        """
        layer: name of layer
        position: (x, y) position
        Returns the value of the layer at the position
        """
        index = self.get_index(position)
        return self.at(layer, index)


    def move(self, position: Position):
        """
        position: (x, y) absolute position
        Moves the grid map to designated location
        Keeps data that is still in frame, destroys outside data.
        """
        # TODO: implement this
        # circular buffer?

        # build new gridmap with new position
        # loop through old gridmap and add to new gridmap if still in frame
        # return new gridmap and destroy old gridmap
        pass


    def get_submap(self, position: Position, length: Length) -> 'Gridmap':
        """
        position: (x, y) position
        length: (x, y) dimensions
        Returns a submap of the grid map based on the position and dimensions
        """
        # TODO: implement this
        pass


    def add_layer(self, name: str, value: float):
        """
        Adds information layer to heightmap. Populates layer with value given.
        """
        layer = np.full((self.get_num_rows(), self.get_num_cols()), value)
        self.layers[name] = layer

    def set_layer(self, layer: str, value: float):
        """
        Populates layer with value given.
        """
        if self.layers.get(layer) is not None:
            self.layers[layer] = np.full((self.get_num_rows(), self.get_num_cols()), value)
        else:
            raise ValueError("Layer not found")
        
    
    def get_layer(self, layer: str) -> np.ndarray:
        """
        layer: name of layer
        Returns the layer
        """
        return self.layers[layer]
    
        
    def update_at_position(self, layer: str, occupancy_layer: str, position: Position, value: float):
        """
        layer: name of layer
        position: (x, y) position
        value: value to update
        Updates the value of the layer at the position
        """
        if not self.is_inside(position):
            # print("Position is outside of grid map", position)
            # print("origin of gridmap: ", self.position)
            # print("length of gridmap: ", self.length)
            return
            raise ValueError("Position is outside of grid map")
        

        index = self.get_index(position)
        # if self.layers[occupancy_layer][index[0]][index[1]] > 0:
        #     # average of old estimate and new point
        #     self.layers[layer][index[0]][index[1]] = 0.5 * (value + self.layers[layer][index[0]][index[1]])
        # else:
        self.layers[layer][index[0]][index[1]] = value

        # TODO: add more sophisticated update function -> options to pick how to update
        # 1. average, 2. max, 3. min, 4. sum, 5. weighted sum, 6. kalman, etc.



    def increment_at_position(self, layer: str, position: Position):
        """
        layer: name of layer
        position: (x, y) position
        Increments the value of the layer at the position
        """
        if not self.is_inside(position):
            return
            raise ValueError("Position is outside of grid map")
    
        index = self.get_index(position)
        self.layers[layer][index[0]][index[1]] += 1


    # def multi_update(self, layer: str, occupancy_layer: str, positions: np.ndarray, values: np.ndarray):
    #     """
    #     layer: name of layer
    #     positions: list of (x, y) positions
    #     values: list of values
    #     Updates the value of the layer at the positions
    #     """
    #     indexes = self.get_indexes(positions)
    #     if self.layers[occupancy_layer][index[0]][index[1]] > 0:
    #         self.layers[layer][index[0]][index[1]] = 0.5 * (value + self.layers[layer][index[0]][index[1]])
    #     else:
    #         self.layers[layer][index[0]][index[1]] = value


    def visualize_layer(self, layer: str):
        """
        layer: name of layer
        Visualizes the layer
        """
        if layer not in self.layers:
            raise ValueError(f"Layer '{layer}' not found in the grid map.")
        
        origin_x = self.position.x
        origin_y = self.position.y
        x_start = origin_x - self.length.x / 2
        x_end = origin_x + self.length.x / 2

        y_start = origin_y - self.length.y / 2
        y_end = origin_y + self.length.y / 2

        plt.imshow(self.layers[layer], cmap='viridis', interpolation='none', extent=[x_start, x_end, y_start, y_end])
        plt.colorbar()
        plt.title(f"Visualization of layer: {layer}")
        plt.xlabel("X position (m)")
        plt.ylabel("Y position (m)")
        plt.show()





    # TODO image export processing functions

if __name__ == "__main__":
    print("test gridmap here")

    # position1 = Position(10.0, 20.0)
    # position2 = Position(5.0, 15.0)
    # result_add = position1 + position2
    # result_sub = position1 - position2
    # print(result_add)  # Output: Position(x=15.0, y=35.0)
    # print(result_sub) 

    gridmap = Gridmap()
    gridmap.set_geometry(Length(5.1, 5), 1, Position(0, 0))

    print(gridmap.get_length())  # Output: Length(x=10, y=10)
    print(gridmap.get_resolution())  # Output: 1
    print(gridmap.get_position())  # Output: Position(x=0, y=0)
    print(gridmap.get_num_rows())  # Output: 10
    print(gridmap.get_num_cols())  # Output: 10

    print(gridmap.is_inside(Position(5, 5)))  # Output: True
    print(gridmap.is_inside(Position(15, 15)))  # Output: False

    print(gridmap.get_index(Position(2.5, 2.5)))  # Output: (5, 5)
    print(gridmap.get_index(Position(0, 0)))  # Output: (5, 5)
    print(gridmap.get_index(Position(0, -2.5)))  # Output: (5, 5)
    
    print(gridmap.get_cell_position((2.5, 2.5)))  # Output: Position(x=-5.0, y=5.0)

    gridmap.add_layer("heightmap", 0)
    print(gridmap.at("heightmap", (1, 1)))  # Output: value
    print(gridmap.atPosition("heightmap", Position(-2.5, 2.5)))  # Output: value

    # gridmap.move(Position(5, 5))

    # submap = gridmap.get_submap(Position(5, 5), (5, 5))

    # gridmap.add_layer("layer1", 5)




