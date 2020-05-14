
# potentials-path-planning
Multi robot path planning with Aritificial Potential Functions.

Here we use Gaussian potential functions as described in [1]. Need to add Lagrangian dynamics to compute higher order control inputs. 
For now we only compute the next best position to follow the negrative gradient. 

Install scipy. 

    pip install 'scipy'

To run create a directory named 'data' to store the plot images and run the script. 

    mkdir data && python potential_field_planning.py

![Cover Image](https://raw.githubusercontent.com/Malintha/potentials-path-planning/master/cover.png)

[1] Gazi, Veysel. "On lagrangian dynamics based modeling of swarm behavior." Physica D: Nonlinear Phenomena 260 (2013): 159-175.
