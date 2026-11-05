import numpy as np
import torch

from dendritic_dr.masks import local_rf_dendrite_mask, random_dendrite_mask, soma_mask
from dendritic_dr.metrics import ordinal_predict, ordinal_targets, quadratic_kappa
from dendritic_dr.models import DANN, MaskedLinear, VanillaMLP, effective_params


def test_random_mask_synapse_count():
    m = random_dendrite_mask(100, 20, 7, np.random.default_rng(0))
    assert m.shape == (20, 100)
    assert (m.sum(1) == 7).all()


def test_soma_mask_block_diagonal():
    m = soma_mask(3, 4)
    assert m.shape == (3, 12)
    assert (m.sum(1) == 4).all() and (m.sum(0) == 1).all()  # each dendrite -> exactly one soma


def test_lrf_mask_is_local_patch():
    m = local_rf_dendrite_mask((1, 28, 28), 5, 3, 4, np.random.default_rng(0))
    assert m.shape == (15, 784)
    assert (m.sum(1) == 16).all()  # 4x4 patch, one channel
    rows, cols = np.nonzero(m[0].reshape(28, 28).numpy())
    assert rows.max() - rows.min() == 3 and cols.max() - cols.min() == 3


def test_masked_weights_get_no_gradient():
    mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
    layer = MaskedLinear(mask)
    layer(torch.randn(4, 2)).sum().backward()
    assert layer.weight.grad[0, 1] == 0 and layer.weight.grad[1, 0] == 0


def test_dann_forward_and_param_count():
    model = DANN(512, 5, somas=[32], dendrites_per_soma=4, synapses=8, seed=0)
    assert model(torch.randn(10, 512)).shape == (10, 5)
    # dendrites: 32*4 dendrites * 8 synapses + 128 biases; somas: 128 + 32; output: 32*5 + 5
    expected = 128 * 8 + 128 + 128 + 32 + 32 * 5 + 5
    assert effective_params(model) == expected


def test_dann_lrf_on_images():
    model = DANN(784, 10, somas=[16], dendrites_per_soma=4, sampling="lrf", image_shape=(1, 28, 28), rf_size=4)
    assert model(torch.randn(3, 1, 28, 28)).shape == (3, 10)


def test_vanilla_param_count():
    model = VanillaMLP(10, 5, [8])
    assert effective_params(model) == 10 * 8 + 8 + 8 * 5 + 5


def test_ordinal_roundtrip():
    y = torch.tensor([0, 1, 2, 3, 4])
    t = ordinal_targets(y)
    logits = (t * 2 - 1) * 10.0  # confident, correct logits
    assert torch.equal(ordinal_predict(logits), y)


def test_qwk_perfect_and_worst():
    assert quadratic_kappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]) == 1.0
    assert quadratic_kappa([0, 0, 4, 4], [4, 4, 0, 0]) < 0


def test_lrf_channel_sparse_mask_is_local_and_sparse():
    m = local_rf_dendrite_mask((512, 7, 7), 8, 4, 3, np.random.default_rng(0), synapses=16)
    assert m.shape == (32, 512 * 7 * 7)
    assert (m.sum(1) == 16).all()  # 16 synapses, not 3*3*512
    for d in range(32):
        _, ys, xs = np.nonzero(m[d].reshape(512, 7, 7).numpy())
        assert ys.max() - ys.min() <= 2 and xs.max() - xs.min() <= 2  # inside one 3x3 window


def test_dann_lrf_on_feature_maps_matches_random_param_count():
    shape = (64, 7, 7)
    lrf = DANN(64 * 49, 5, somas=[32], dendrites_per_soma=4, synapses=16, sampling="lrf",
               image_shape=shape, rf_size=3, seed=0)
    rnd = DANN(64 * 49, 5, somas=[32], dendrites_per_soma=4, synapses=16, sampling="random", seed=0)
    assert lrf(torch.randn(2, *shape)).shape == (2, 5)
    assert effective_params(lrf) == effective_params(rnd)  # only locality differs


def test_fmnist_lrf_unchanged_full_patch():
    # single-channel images keep the full rf x rf patch (original replication setup)
    model = DANN(784, 10, somas=[4], dendrites_per_soma=2, synapses=16, sampling="lrf",
                 image_shape=(1, 28, 28), rf_size=4, seed=0)
    assert (model.body[0].dendrites.mask.sum(1) == 16).all()


def test_compact_state_dict_roundtrip():
    from dendritic_dr.models import compact_state_dict, load_compact_state_dict

    kw = dict(somas=[8], dendrites_per_soma=4, synapses=8, sampling="lrf", image_shape=(16, 5, 5), rf_size=3, seed=3)
    a = DANN(16 * 25, 5, **kw)
    with torch.no_grad():
        for p in a.parameters():
            p.add_(torch.randn_like(p))
        for m in a.modules():  # keep masked entries at zero, as training does
            if isinstance(m, MaskedLinear):
                m.weight.mul_(m.mask)
    sd = compact_state_dict(a)
    assert not any(k.endswith(".mask") for k in sd)
    b = DANN(16 * 25, 5, **kw)
    load_compact_state_dict(b, sd)
    x = torch.randn(4, 16 * 25)
    assert torch.equal(a(x), b(x))
