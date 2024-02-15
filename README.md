# Lunar lander project

The idea is the use of genetics with neural networks to learn to control a LunarLander with the AI gym using Reinforcement Learning to learn based on last experience.

## Experiment execution

### Dependencies

Found on [requirements.txt](./requirements.txt)

* `gym`
* `matplotlib`
* `networkx`
* `numpy`
* `pygame`

### Steps

1. Download [Visual C++ Build tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
    1. Select **C++ compile applications**
2. Create python environment `python3 -m venv .venv`
3. Activate environment. Script is inside `source .venv/bin/activate`
4. Install dependencies `pip install -r requirements.txt`
5. Train LunarLander `python LunarLander/LunarLander_trainning.py --help`
6. Execute LunarLander to watch the best [notebook](./LunarLander/Gymnasium%20alumno.ipynb)

```sh
$ python LunarLander_training.py --help

usage: LunarLander_training.py [-h] [-p POPULATION_SIZE] [-g NUM_GENERATIONS] [-o OUTPUT_FILE] [-og BEST_IN_GENERATION_OUTPUT_DIR] [-of FITNESS_OUTPUT_FILE]

options:
  -h, --help            show this help message and exit
  -p POPULATION_SIZE, --population_size POPULATION_SIZE
                        Size of the population.
  -g NUM_GENERATIONS, --num_generations NUM_GENERATIONS
                        Number of generations.
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        File to save the winning genome.
  -og BEST_IN_GENERATION_OUTPUT_DIR, --best_in_generation_output_dir BEST_IN_GENERATION_OUTPUT_DIR
                        Directory to save the best genome of each generation.
  -of FITNESS_OUTPUT_FILE, --fitness_output_file FITNESS_OUTPUT_FILE
                        File to save fitness scores.
```

Example `python LunarLander_training.py -p 50 -g 100 -o winner_gnome -og ../output -of ../output_fitness`

## Structure

The project has three main modules.

* `LunarLander`. Definition of AI and human controller, fitness class to meassure best individual on each generation.
* `NEAT`. Source code for **NEAT** algorithm.
* `RNG`. Implemented random library for more customization on NEAT algorithm.

## References

* [Efficient Evolution of Neural Network Topologies (paper)](https://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf)
* [NEAT algorithm Tech with Nikola (video)](https://m.youtube.com/watch?v=lAjcH-hCusg)
* [LunarLander](https://www.gymlibrary.dev/environments/box2d/lunar_lander/)

## Authors

* [AlvaroTena](https://github.com/AlvaroTena)
* [blitty-codes](https://github.com/blitty-codes)
* [w-dan](https://github.com/w-dan/)
