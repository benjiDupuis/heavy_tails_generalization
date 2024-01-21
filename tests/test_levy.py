from levy.levy import generate_levy_for_simulation


DIMENSION = 2
N = 10
ALPHA = 1.8
ETA = 0.01



def test_levy_generation():

    noise = generate_levy_for_simulation(DIMENSION, \
                                         N,
                                         ALPHA,
                                         ETA)
    assert noise.shape == (N, DIMENSION), (noise.shape, (N, DIMENSION))

def test_levy_generation_brownian():

    noise = generate_levy_for_simulation(DIMENSION, \
                                         N,
                                         2.,
                                         ETA)
    assert noise.shape == (N, DIMENSION), (noise.shape, (N, DIMENSION))
