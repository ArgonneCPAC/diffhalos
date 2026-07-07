import jax
import jax.numpy as jnp

from .. import diffmahnet


def test_diffmahflow():
    randkey = jax.random.key(0)
    keys = jax.random.split(randkey, 6)

    ndata = 1000

    # m_obs and t_obs
    fake_conditions = jax.random.normal(keys[0], (ndata, 2)) + 1.5
    fake_mah_uparams = jax.random.normal(keys[1], (ndata, 5)) * 0.2 - 4.0
    scaler = diffmahnet.Scaler.compute(fake_mah_uparams, fake_conditions)

    flow = diffmahnet.DiffMahFlow(scaler)
    test_prediction = flow.sample(fake_conditions, keys[2])

    # Even without training, the flow should produce roughly the same
    # normal distribution as the fake training data
    assert jnp.allclose(
        test_prediction.mean(axis=0), fake_mah_uparams.mean(axis=0), atol=0.1
    ), "Mean of predictions should be close to the mean of fake data"
    assert jnp.allclose(
        test_prediction.std(axis=0), fake_mah_uparams.std(axis=0), atol=0.1
    ), "Std of predictions should be close to the std of fake data"

    # Make sure asparams=True gives tuple output
    mahparams_prediction = flow.sample(fake_conditions, keys[2], asparams=True)
    assert isinstance(mahparams_prediction, tuple)
