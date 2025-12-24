
import sys

try:
    import torch
    import neuralop
except ImportError as e:
    print(
        "Required packages torch,neuralop are missing.\n"
        "Please install them first"
    )
    sys.exit(1)

import torch
from neuralop.layers.embeddings import GridEmbeddingND,GridEmbedding2D
from neuralop.layers.padding import DomainPadding
from neuralop.layers.channel_mlp import ChannelMLP,LinearChannelMLP
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.normalization_layers import BatchNorm,InstanceNorm,AdaIN
from neuralop.layers.skip_connections import skip_connection
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.resample import resample
from neuralop.models.fno import FNO
from neuralop.utils import count_model_params


import jax 
# jax.config.update("jax_enable_x64", True)
# torch.set_default_dtype(torch.float64)
from fno import FNO as flax_FNO
from flax import nnx

import numpy as np
import jax.numpy as jnp
from embeddings import GridEmbeddingND as flax_GridEmbeddingND
from padding import DomainPadding as flax_DomainPadding
from channel_mlp import ChannelMLP as flax_ChannelMLP
from channel_mlp import LinearChannelMLP as flax_LinearChannelMLP
from spectral_convolution import SpectralConv as flax_SpectralConv
from normalization_layers import BatchNorm as flax_BatchNorm
from normalization_layers import InstanceNorm as flax_InstanceNorm
from normalization_layers import AdaIN as flax_AdaIN
from skip_connections import skip_connection as flax_skip_connection
from fno_block import FNOBlocks as flax_FNOBlocks
from resample import resample as flax_resample

import numpy as np

def compare_tensors(a, b, name_a="a", name_b="b"):
    """
    Systematically compare two NumPy tensors and print:
      - Shape match
      - Dtype match
      - Allclose within tolerance
      - Max & mean absolute error
      - Max & mean relative error
      - L2 & normalized L2 error
      - Cosine similarity
    """
    
    print(f"Comparing tensors: {name_a} vs {name_b}")
    print("-" * 50)

    # ---------------------------
    # Shape comparison
    # ---------------------------
    print("Shape:")
    print(f"  {name_a}: {a.shape}")
    print(f"  {name_b}: {b.shape}")
    shapes_match = (a.shape == b.shape)
    print("  Match:", shapes_match)
    print()

    if not shapes_match:
        print("Shapes do not match — stopping comparison.")
        return

    # ---------------------------
    # Dtype comparison
    # ---------------------------
    print("Dtype:")
    print(f"  {name_a}: {a.dtype}")
    print(f"  {name_b}: {b.dtype}")
    dtypes_match = (a.dtype == b.dtype)
    print("  Match:", dtypes_match)
    print()

    # ---------------------------
    # Numerical closeness
    # ---------------------------
    print("Numerical closeness:")
    allclose = np.allclose(a, b, rtol=1e-5, atol=1e-8)
    print("  allclose (rtol=1e-5, atol=1e-8):", allclose)

    # Absolute & relative differences
    diff = np.abs(a - b)
    print("  max abs diff:", diff.max())
    print("  mean abs diff:", diff.mean())

    rel = diff / (np.abs(b) + 1e-12)
    print("  max rel diff:", rel.max())
    print("  mean rel diff:", rel.mean())
    print()

    # ---------------------------
    # Norm-based metrics
    # ---------------------------
    print("Norm metrics:")
    l2_err = np.linalg.norm(a - b)
    l2_norm = np.linalg.norm(b)
    print("  L2 error:", l2_err)
    print("  Normalized L2 error:", l2_err / (l2_norm + 1e-12))
    print()

    # ---------------------------
    # Cosine similarity
    # ---------------------------
    flat_a = a.ravel().astype(np.float64)
    flat_b = b.ravel().astype(np.float64)
    cos = np.dot(flat_a, flat_b) / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b) + 1e-12)
    print("Cosine similarity:", cos)

    print("-" * 50)
    print("Comparison complete.")



def test_GridEmbeddingND():
    dim = 3  
    in_channels = 4
    bs = 1000
    flax_embed = flax_GridEmbeddingND(in_channels=in_channels,
                                        dim=dim,
                                        grid_boundaries=[[0, 1], [0, 1], [0, 1]])
    
    torch_embed = GridEmbeddingND(in_channels=in_channels,
                                  dim=dim,
                                  grid_boundaries=[[0, 1], [0, 1], [0, 1]])
    
    np.random.seed(0) 

    no_batch = False
    if no_batch:

        x_np_no_batch = np.random.randn(16, 16, 16, in_channels).astype(np.float32)
        torch_x_np_no_batch = np.moveaxis(x_np_no_batch,-1,0)

        flax_y_no_batch = np.array(flax_embed(jnp.array(x_np_no_batch),False))
        torch_y_no_batch = torch_embed(torch.from_numpy(torch_x_np_no_batch),False).detach().cpu().numpy()
        torch_y_no_batch = np.moveaxis(torch_y_no_batch,1,-1)

        compare_tensors(flax_y_no_batch,torch_y_no_batch)
    else:
        x_np = np.random.randn(bs, 16, 16, 16, in_channels).astype(np.float32)
        torch_x_np = np.moveaxis(x_np,-1,1)

        flax_y = np.array(flax_embed(jnp.array(x_np)))
        torch_y = torch_embed(torch.from_numpy(torch_x_np)).detach().cpu().numpy()
        torch_y = np.moveaxis(torch_y,1,-1)

        compare_tensors(torch_y,flax_y)


    flax_embed = flax_GridEmbeddingND(in_channels=in_channels,
                                      dim=2,
                                      grid_boundaries=[[0, 1], [0, 1]])
    
    torch_embed = GridEmbedding2D(in_channels=in_channels,
                                  grid_boundaries=[[0, 1], [0, 1]])


    no_batch = False
    if no_batch:

        x_np_no_batch = np.random.randn(16, 16, in_channels).astype(np.float32)
        torch_x_np_no_batch = np.moveaxis(x_np_no_batch,-1,0)

        flax_y_no_batch = np.array(flax_embed(jnp.array(x_np_no_batch),False))
        torch_y_no_batch = torch_embed(torch.from_numpy(torch_x_np_no_batch),False).detach().cpu().numpy()
        torch_y_no_batch = np.moveaxis(torch_y_no_batch,1,-1)

        compare_tensors(flax_y_no_batch,torch_y_no_batch)
    else:
        x_np = np.random.randn(bs, 16, 16, in_channels).astype(np.float32)
        torch_x_np = np.moveaxis(x_np,-1,1)

        flax_y = np.array(flax_embed(jnp.array(x_np)))
        torch_y = torch_embed(torch.from_numpy(torch_x_np)).detach().cpu().numpy()
        torch_y = np.moveaxis(torch_y,1,-1)

        compare_tensors(torch_y,flax_y)

def export_pytorch_model_params(module):
    param_dict = {}
    for name, param in module.named_parameters():
        if param.requires_grad:       # only modify trainable parameters
            param_dict[name] = param.data.numpy()

    import pickle
    with open("pytorch_params.pkl", "wb") as f:
        pickle.dump(param_dict, f)   

def seed_everything(seed: int = 0):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using CUDA

def test_FNO():

    n_dim = 3
    s = 16
    modes = 8
    width = 16
    batch_size = 125
    n_layers = 4

    size = (s,) * n_dim
    n_modes = (modes,) * n_dim

    # seed_everything(512)

    pytorch_fno_model = FNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=width,
        n_modes=n_modes,
        n_layers=n_layers,
    )

    pytorch_fno_model.eval()
    export_pytorch_model_params(pytorch_fno_model)
    pytorch_fno_model = pytorch_fno_model.to("cuda")

    total_params = sum(p.numel() for p in pytorch_fno_model.parameters() if p.requires_grad)
    print("Pytorch trainable parameters:", total_params)

    in_data = torch.randn(batch_size, 3, *size, dtype=torch.float32).to("cuda")
    print(f"in_data.shape:{in_data.shape}")
    pytorch_out = pytorch_fno_model(in_data).detach().cpu().numpy()
    pytorch_out = np.moveaxis(pytorch_out, 1, -1)
    print(f"Pytorch out.shape:{pytorch_out.shape}")

    rngs = nnx.Rngs(0)
    flax_fno_model = flax_FNO(
        in_channels=3,
        out_channels=1,
        hidden_channels=width,
        n_modes=n_modes,
        n_layers=n_layers,
        rngs=rngs
    )

    flax_fno_model.eval()

    import pickle
    with open("pytorch_params.pkl", "rb") as f:
        pytorch_params_dict = pickle.load(f) 

    graphdef, params = nnx.split(flax_fno_model, nnx.Param)
    flat_nnx_params = dict(nnx.to_flat_state(params))

    for key,nnx_var in flat_nnx_params.items():
      param_name = str(key[0])
      for key_seg in key[1:]:
        if key_seg != 'layers':
          if key_seg in ["kernel","scale"]:
            param_name += str("."+str("weight"))
          else:
            param_name += str("."+str(key_seg))

      pytorch_key = None
      for torch_param_name in pytorch_params_dict.keys():
          if param_name in torch_param_name:
              pytorch_key = torch_param_name
              break

      if pytorch_key is None:
         print(f"could not find param with name {param_name} in pytorch_params_dict, EXIT !")
         exit()

      pytorch_value = pytorch_params_dict[pytorch_key]

      py_torch_param_shape = pytorch_value.shape
      flax_param_shape = nnx_var.value.shape
      if py_torch_param_shape==flax_param_shape:
        nnx_var.value = pytorch_value
      elif pytorch_value.T.shape == flax_param_shape:
        nnx_var.value = pytorch_value.T
      elif flax_param_shape== np.squeeze(pytorch_params_dict[pytorch_key]).shape:
        nnx_var.value = np.squeeze(pytorch_params_dict[pytorch_key])
      elif flax_param_shape== np.squeeze(pytorch_params_dict[pytorch_key]).T.shape:
        nnx_var.value = np.squeeze(pytorch_params_dict[pytorch_key]).T
      elif flax_param_shape== np.moveaxis(pytorch_value, 1, -1).shape:
        nnx_var.value = np.moveaxis(pytorch_value, 1, -1)
      elif flax_param_shape== np.moveaxis(pytorch_value, 1, -1).T.shape:
        nnx_var.value = np.moveaxis(pytorch_value, 1, -1).T
      elif np.squeeze(nnx_var.value).shape == np.squeeze(pytorch_params_dict[pytorch_key]).shape:
        nnx_var.value = np.squeeze(pytorch_params_dict[pytorch_key]).reshape(nnx_var.value.shape)
      else:
        print(f"flax param:{param_name}/pytorch eqi:{pytorch_key}, has pytorch shape {py_torch_param_shape} and jax shape {flax_param_shape}, and are npt compatibel ! EXIT")
        exit()
    
    # now update
    nnx.update(flax_fno_model, params) 

    jax_in_data = jnp.array(in_data.cpu().numpy())
    jax_in_data_cl = jnp.moveaxis(jax_in_data, 1, -1)

    print(f"flax in_data.shape:{jax_in_data_cl.shape}")
    flax_out = np.array(flax_fno_model(jax_in_data_cl))
    print(f"flax out_data.shape:{flax_out.shape}")
    
    params = nnx.state(flax_fno_model, nnx.Param)
    total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print(f"Flax trainable parameters:{total_params}")

    compare_tensors(flax_out,pytorch_out)

    size = np.prod(flax_out.shape)

    indices = np.random.randint(0, size, size=10)

    print(f"pytorch fno out:{pytorch_out.ravel()[indices]}")
    print(f"flax fno out:{flax_out.ravel()[indices]}")
    
def test_Padding():

    np.random.seed(0)
    batch_size = 1000
    channels = 15
    grid_size = 16
    domain_padding = 0.2
    res_scale = 1
    dim = 3

    nchw_input_shape = (batch_size, channels) + (grid_size,) * dim
    x_np_nchw = np.random.randn(*nchw_input_shape).astype(np.float32)
    x_np_nhwc = np.moveaxis(x_np_nchw, 1, -1)


    pad_torch = DomainPadding(
        domain_padding=domain_padding,
        resolution_scaling_factor=res_scale,
    ).to("cuda")
    pad_jax = flax_DomainPadding(
        domain_padding=domain_padding,
        resolution_scaling_factor=res_scale,
    )

    y_torch = pad_torch.pad(torch.from_numpy(x_np_nchw).to("cuda")).detach().cpu().numpy()
    y_torch = np.moveaxis(y_torch, 1, -1)
    y_jax = np.array(pad_jax(jnp.array(x_np_nhwc)))
    compare_tensors(y_jax,y_torch)

    y_torch_unpadded = pad_torch.unpad(pad_torch.pad(torch.from_numpy(x_np_nchw).to("cuda"))).detach().cpu().numpy()          # NCHW
    y_torch_unpadded = np.moveaxis(y_torch_unpadded, 1, -1)
    y_jax_unpadded = np.array(pad_jax.unpad(y_jax)) 

    compare_tensors(y_jax_unpadded,y_torch_unpadded)

def test_ChannelMLP():

    def copy_channel_mlp_weights(torch_mlp, nnx_mlp):
        """
        Copy Conv1d weights/biases from TorchChannelMLP to NNXChannelMLP Linear layers.
        """
        assert len(torch_mlp.fcs) == len(nnx_mlp.fcs)
        for conv, lin in zip(torch_mlp.fcs, nnx_mlp.fcs):
            # conv.weight: (C_out, C_in, 1)
            W = conv.weight.detach().numpy().squeeze(-1)  # (C_out, C_in)
            b = conv.bias.detach().numpy()                # (C_out,)

            # lin.kernel: (in_features, out_features) = (C_in, C_out)
            lin.kernel.value = jnp.array(W.T)  # (C_in, C_out)
            lin.bias.value = jnp.array(b)      # (C_out,)    

    torch.manual_seed(0)
    # Hyperparams
    batch_size = 200
    in_channels = 3
    out_channels = 3
    hidden_channels = 8
    H, W = 5, 7
    n_layers = 3
    dropout = 0.1  # for clean comparison, keep dropout 0    
    
    torch_mlp = ChannelMLP(in_channels=in_channels,
                           out_channels=out_channels,
                           hidden_channels=hidden_channels,
                           n_layers=n_layers,
                           dropout=dropout)
    torch_mlp.eval()
    # Input: (B, C, H, W)
    x_torch = torch.randn(batch_size, in_channels, H, W)
    y_torch = torch_mlp(x_torch)  # (B, out_channels, H, W)  

    # Create NNX ChannelMLP (NHWC)
    rngs = nnx.Rngs(0)
    nnx_mlp = flax_ChannelMLP(in_channels=in_channels,
                              out_channels=out_channels,
                              hidden_channels=hidden_channels,
                              n_layers=n_layers,
                              dropout=dropout,
                              rngs=rngs)
    
    nnx_mlp.eval()
    
    # Copy weights Conv1d -> Linear
    copy_channel_mlp_weights(torch_mlp, nnx_mlp)  

    # Convert input to NHWC for JAX: (B, C, H, W) -> (B, H, W, C)
    x_jax = jnp.moveaxis(x_torch.detach().numpy(),1,-1)

    # Run NNX MLP
    y_jax = nnx_mlp(x_jax)  # (B, H, W, out_channels) 
    # Convert JAX output back to NCHW: (B, H, W, C) -> (B, C, H, W)
    y_jax_nchw = jnp.moveaxis(y_jax,-1,1)     

    y_torch_np = y_torch.detach().numpy()
    y_jax_np = np.array(y_jax_nchw)    

    # Compare
    compare_tensors(y_torch_np,y_jax_np)

def test_LinearChannelMLP():

    def copy_channel_mlp_weights(torch_mlp, nnx_mlp):
        """
        Copy Conv1d weights/biases from TorchChannelMLP to NNXChannelMLP Linear layers.
        """
        assert len(torch_mlp.fcs) == len(nnx_mlp.fcs)
        for conv, lin in zip(torch_mlp.fcs, nnx_mlp.fcs):
            W = conv.weight.detach().numpy()
            b = conv.bias.detach().numpy()

            lin.kernel.value = jnp.array(W.T)
            lin.bias.value = jnp.array(b)   

    torch.manual_seed(0)
    # Hyperparams
    batch_size = 200
    in_channels = 3
    out_channels = 6
    hidden_layers = [64, 128, 256, 10]
    dropout = 0.1   
    
    torch_mlp = LinearChannelMLP(layers=[in_channels] + hidden_layers + [out_channels],
                                 dropout=dropout)
    torch_mlp.eval()
    x_torch = torch.randn(batch_size, in_channels)
    y_torch = torch_mlp(x_torch) 

    # Create NNX ChannelMLP 
    rngs = nnx.Rngs(0)
    nnx_mlp = flax_LinearChannelMLP(layers=[in_channels] + hidden_layers + [out_channels],
                                    dropout=dropout,
                                    rngs=rngs)
    
    nnx_mlp.eval()
    
    # Copy weights Conv1d -> Linear
    copy_channel_mlp_weights(torch_mlp, nnx_mlp)  

    x_jax = jnp.array(x_torch.detach().numpy())

    # Run NNX MLP
    y_jax = nnx_mlp(x_jax)   

    y_torch_np = y_torch.detach().numpy()
    y_jax_np = np.array(y_jax)    

    # Compare
    compare_tensors(y_torch_np,y_jax_np)

def test_SpectralConv():

    dim=3
    modes = (10, 8, 6)
    B = 200
    Cin = 6
    Cout = 4    
    grid_size = 24

    torch_SpectralConv = SpectralConv(
        Cin,
        Cout,
        modes[:dim],
    )

    rngs = nnx.Rngs(0)
    jax_SpectralConv = flax_SpectralConv(rngs=rngs,
                                 in_channels=Cin,
                                 out_channels=Cout,
                                 n_modes=modes[:dim],)

    # -------------------------------
    # COPY WEIGHTS Torch → JAX
    # -------------------------------
    print("Copying weights...")
    torch_weight = torch_SpectralConv.weight
    if not torch.is_tensor(torch_weight):
        torch_weight = torch_weight.to_tensor()    

    jax_SpectralConv.weight.value = jnp.array(torch_weight.permute(0, 2, 3, 4, 1).detach().numpy())
    if torch_SpectralConv.bias is not None:
        jax_SpectralConv.bias.value = jnp.array(torch_SpectralConv.bias.permute(3, 1, 2, 0).detach().numpy()) 


    nchw_input_shape = (B, Cin) + (grid_size,) * dim
    x_np_nchw = np.random.randn(*nchw_input_shape).astype(np.float32)
    x_np_nhwc = np.moveaxis(x_np_nchw, 1, -1)

    y_jax = np.array(jax_SpectralConv(x_np_nhwc))
    y_torch = torch_SpectralConv(torch.from_numpy(x_np_nchw)).detach().cpu().numpy()
    y_torch = np.moveaxis(y_torch, 1, -1)

    compare_tensors(y_torch,y_jax)

    print(y_torch.flatten()[90:100])
    print(y_jax.flatten()[90:100])

def test_Normalizations(max_abs_tol=5e-4):
    
    validate_batch_norm = True
    if validate_batch_norm:
        key = jax.random.key(0)
        rngs = nnx.Rngs(params=0)

        for n_dim in [1, 2, 3, 4, 5]:
                print(f"\nTesting n_dim = {n_dim}")

                C = 8
                B = 200
                spatial = (4,) * n_dim
                size_cf = (B, C, *spatial)

                x_cf = jax.random.normal(key, size_cf)
                x_torch = torch.from_numpy(np.array(x_cf))

                torch_bn = BatchNorm(
                    n_dim=n_dim,
                    num_features=C,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    track_running_stats=False,
                )
                # torch_bn.train()
                y_torch = torch_bn(x_torch).detach().numpy()


                # Convert to channel-last
                perm_cf_to_cl = (0, *range(2, 2 + n_dim), 1)
                x_cl = np.transpose(np.array(x_cf), perm_cf_to_cl)

                nnx_bn = flax_BatchNorm(
                    n_dim=n_dim,
                    num_features=C,
                    eps=1e-5,
                    momentum=0.1,
                    affine=True,
                    rngs=rngs,
                )

                y_cl = np.array(nnx_bn(jnp.array(x_cl)))

                # Convert NNX back to channel-first
                perm_cl_to_cf = (0, 1 + n_dim, *range(1, 1 + n_dim))
                y_nnx_cf = np.transpose(y_cl, perm_cl_to_cf)

                diff = np.abs(y_torch - y_nnx_cf).max()
                print(f"max |Torch - NNX| = {diff:.6e}")

                assert diff < max_abs_tol, (
                    f"FAILED at n_dim={n_dim}: diff {diff} > {max_abs_tol}"
                )

    validate_instant_norm = True
    if validate_instant_norm:

        key = jax.random.key(0)
        rngs = nnx.Rngs(params=0)

        eps = 1e-5

        for n_dim in [1,2,3,4,5]:
            for affine in [False, True]:
                print(f"\nTesting n_dim={n_dim}, affine={affine}")

                N, C = 200, 1253
                spatial = (3,) * n_dim
                shape_cf = (N, C, *spatial)

                # Shared random input
                key, sk = jax.random.split(key)
                x_cf = jax.random.normal(sk, shape_cf).astype(jnp.float32)
                x_torch = torch.from_numpy(np.array(x_cf))

                eps = 1e-5
                kwargs = dict(eps=eps, use_input_stats=True)

                if affine:
                    key, k1, k2 = jax.random.split(key, 3)
                    w_np = jax.random.normal(k1, (C,), dtype=jnp.float32)
                    b_np = jax.random.normal(k2, (C,), dtype=jnp.float32)
                    kwargs.update(weight=torch.from_numpy(np.array(w_np)),
                                bias=torch.from_numpy(np.array(b_np)))
                else:
                    w_np = None
                    b_np = None
                    kwargs.update(weight=None, bias=None)

                # ---------------- PyTorch forward ----------------
                inst_pt = InstanceNorm(**kwargs)
                y_pt = inst_pt(x_torch).detach().numpy()

                # ---------------- NNX forward ----------------
                # Convert to channel-last
                perm_cf_to_cl = (0, *range(2, 2+n_dim), 1)
                x_cl = np.transpose(np.asarray(x_cf), perm_cf_to_cl)

                inst_nnx = flax_InstanceNorm(
                    num_features=C,
                    rngs=rngs,
                    eps=eps,
                    weight=w_np,
                    bias=b_np,
                )

                y_cl = np.array(inst_nnx(jnp.array(x_cl)))

                # Convert back CF for comparison
                perm_cl_to_cf = (0, 1+n_dim, *range(1, 1+n_dim))
                y_nnx_cf = np.transpose(y_cl, perm_cl_to_cf)


                diff = np.abs(y_pt - y_nnx_cf).max()
                print(f"max |Torch - NNX| = {diff:.6e}")
                assert diff < max_abs_tol, (
                    f"FAILED at n_dim={n_dim}, affine={affine}, diff={diff}"
                )         

    validate_AdaIN_norm = True
    if validate_AdaIN_norm:

        rng = np.random.default_rng(0)
        rngs = nnx.Rngs(params=0)

        # We use an identity MLP and embed_dim = 2 * C,
        # so the embedding itself is [weight, bias].
        for n_dim in [1, 2, 3, 4, 5]:
            print(f"\nTesting n_dim={n_dim}")

            N, C = 2256, 4
            spatial = (3,) * n_dim
            eps = 1e-5

            # Channel-first input for PyTorch
            x_cf_np = rng.standard_normal((N, C, *spatial), dtype=np.float64)

            # Shared embedding of length 2 * C
            embed_dim = 2 * C
            emb_np = rng.standard_normal((embed_dim,), dtype=np.float64)

            # ----- PyTorch AdaIN -----
            adain_pt = AdaIN(
                embed_dim=embed_dim,
                in_channels=C,
                mlp=None,
                eps=eps,
            )

            x_torch = torch.from_numpy(x_cf_np)
            emb_torch = torch.from_numpy(emb_np)

            adain_pt.set_embedding(emb_torch)
            y_pt = adain_pt(x_torch).detach().numpy()  # (N, C, *spatial)  

            x_cl_np = np.moveaxis(x_cf_np, 1, -1)
            x_jax = jnp.array(x_cl_np)
            emb_jax = jnp.array(emb_np)        

            adain_nnx = flax_AdaIN(
                embed_dim=embed_dim,
                in_channels=C,
                eps=eps,
                rngs=rngs,
            )       

            adain_nnx.mlp.fc1.kernel.value = jnp.array(adain_pt.mlp[0].weight.detach().numpy().T)
            adain_nnx.mlp.fc1.bias.value = jnp.array(adain_pt.mlp[0].bias.detach().numpy())        

            adain_nnx.mlp.fc2.kernel.value = jnp.array(adain_pt.mlp[2].weight.detach().numpy().T)
            adain_nnx.mlp.fc2.bias.value = jnp.array(adain_pt.mlp[2].bias.detach().numpy())
        
            adain_nnx.set_embedding(emb_jax)
            y_cl = adain_nnx(x_jax)              # (N, *spatial, C)
            y_cl_np = np.array(y_cl)     
            # Convert back to channel-first for comparison
            y_nnx_cf = np.moveaxis(y_cl_np, -1, 1)  # (N, C, *spatial)           

            # ----- Compare -----
            diff = np.abs(y_pt - y_nnx_cf).max()
            print(f"max |PyTorch - NNX| = {diff:.6e}")
            assert diff < max_abs_tol, (
                f"FAILED for n_dim={n_dim}: max diff {diff} > {max_abs_tol}"
            )      
    

def test_skip_connection_identity():
    rng = np.random.default_rng(0)
    rngs = nnx.Rngs(params=0)

    N, C, H, W = 2, 4, 5, 6

    x_cf_np = rng.standard_normal((N, C, H, W), dtype=np.float32)
    x_cl_np = np.moveaxis(x_cf_np, 1, -1)

    # Identity in PT
    id_pt = torch.nn.Identity()
    y_pt = id_pt(torch.from_numpy(x_cf_np)).detach().numpy()

    # Identity in NNX
    id_nnx = flax_skip_connection(
        in_features=C,
        out_features=C,
        n_dim=2,
        bias=False,
        skip_type="identity",
        rngs=rngs,
    )

    y_cl = id_nnx(jnp.array(x_cl_np))
    y_cl_np = np.array(y_cl)
    y_nnx_cf = np.moveaxis(y_cl_np, -1, 1)

    diff = np.max(np.abs(y_pt - y_nnx_cf))
    print("Identity max |Torch - NNX| =", diff)
    assert diff < 1e-7


def test_softgating(max_abs_tol: float = 1e-6):
    rng = np.random.default_rng(0)

    N, C, H, W, T = 2000, 116, 5, 6, 8
    n_dim = 3

    for bias_flag in [True,False]:
        print(f"\nTesting SoftGating with bias={bias_flag}")

        # ----- PyTorch (channel-first) -----
        sg_pt = skip_connection(
            in_features=C,
            out_features=None,
            n_dim=n_dim,
            bias=bias_flag,
            skip_type="soft-gating",
        )

        x_cf_np = rng.standard_normal((N, C, H, W, T), dtype=np.float32)
        x_torch = torch.from_numpy(x_cf_np)

        y_pt = sg_pt(x_torch).detach().numpy()  # (N, C, H, W)



        # ----- NNX (channel-last) -----

        sg_nnx = flax_skip_connection(
            in_features=C,
            out_features=None,
            n_dim=n_dim,
            bias=bias_flag,
            skip_type="soft-gating",
            rngs=rng
        )

        # Copy PyTorch params → NNX (weight: (1, C, 1, 1) → (C,))
        w_pt = sg_pt.weight.detach().numpy()       # (1, C, 1, 1)
        w_vec = w_pt.reshape(C)                    # per-channel
        sg_nnx.weight = nnx.Param(jnp.array(w_vec))

        if bias_flag:
            b_pt = sg_pt.bias.detach().numpy()     # (1, C, 1, 1)
            b_vec = b_pt.reshape(C)
            sg_nnx.bias = nnx.Param(jnp.array(b_vec))

        # Convert input to channel-last: (N, H, W, C)
        x_cl_np = np.moveaxis(x_cf_np, 1, -1)
        x_jax = jnp.array(x_cl_np)

        y_cl = sg_nnx(x_jax)               # (N, H, W, C)
        y_cl_np = np.array(y_cl)

        # Back to channel-first for comparison
        y_nnx_cf = np.moveaxis(y_cl_np, -1, 1)  # (N, C, H, W)

        diff = np.max(np.abs(y_pt - y_nnx_cf))
        print("max |Torch - NNX| =", diff)
        assert diff < max_abs_tol, (
            f"SoftGating diff {diff} exceeds tolerance {max_abs_tol}"
        )

def test_flattened1dconv(max_abs_tol: float = 1e-6):
    rng = np.random.default_rng(0)
    rngs = nnx.Rngs(params=0)

    # Example shape: 4D tensor (B, C, H, W)
    B, C_in, H, W, T = 2200, 3, 4, 5, 523
    C_out = 6
    kernel_size = 1  # important: keeps flattened length same
    bias_flag = True

    # ----- PyTorch module (channel-first) -----

    conv_pt = skip_connection(
        in_features=C_in,
        out_features=C_out,
        bias=bias_flag,
        skip_type="linear",
    )


    x_cf_np = rng.standard_normal((B, C_in, H, W, T), dtype=np.float64)
    x_torch = torch.from_numpy(x_cf_np)
    y_pt = conv_pt(x_torch).detach().numpy()   # (B, C_out, H, W)

    # ----- NNX module (channel-last) -----
    conv_nnx = flax_skip_connection(
        in_features=C_in,
        out_features=C_out,
        bias=bias_flag,
        skip_type="linear",
        rngs=rngs
    )

    # Copy Conv1d weights: PT (C_out, C_in, K) → NNX (K, C_in, C_out)
    w_pt = conv_pt.conv.weight.detach().numpy()   # (C_out, C_in, K)

    w_nnx = np.transpose(w_pt, (2, 1, 0))         # (K, C_in, C_out)

    conv_nnx.conv.kernel.value = jnp.array(w_nnx)

    if bias_flag:
        b_pt = conv_pt.conv.bias.detach().numpy()  # (C_out,)
        conv_nnx.conv.bias.value = jnp.array(b_pt)

    # Convert input to channel-last: (B, H, W, C_in)
    x_cl_np = np.moveaxis(x_cf_np, 1, -1)
    x_jax = jnp.array(x_cl_np)

    y_cl = conv_nnx(x_jax)                # (B, H, W, C_out)
    y_cl_np = np.array(y_cl)

    # Convert back to channel-first for comparison
    y_nnx_cf = np.moveaxis(y_cl_np, -1, 1)  # (B, C_out, H, W)

    compare_tensors(y_pt,y_nnx_cf)


def test_FNOBlock_resolution_scaling_factor():
    """Test FNOBlocks with upsampled or downsampled outputs"""
    max_n_modes = [8, 8, 8, 8]
    n_modes = [4, 4, 4, 4]

    size = [10] * 4
    channel_mlp_dropout = 0
    channel_mlp_expansion = 1.0
    channel_mlp_skip = "linear"
    for dim in [3]:


        # Downsample outputs
        block = FNOBlocks(
            4,
            4,
            n_modes[:dim],
            n_layers=3,
            stabilizer="tanh",
            preactivation = True
        )

        nnx_block = flax_FNOBlocks(
            4,
            4,
            n_modes[:dim],
            n_layers=3,
            stabilizer="tanh",
            preactivation = True,
            rngs=nnx.Rngs(0)
        )

        x = torch.randn(2, 4, *size[:dim])
        res = block(x)

        x_cl = jnp.moveaxis(x.detach().numpy(), 1, -1)
        res_cl = nnx_block(x_cl)
        print(res_cl.shape)
        print(res_cl.dtype)

def test_resample():
    a = torch.randn(10, 20, 40, 50)

    res_scale = [2, 3]
    axis = [-2, -1]

    b = resample(a, res_scale, axis)
    b_jax = flax_resample(a.detach().numpy(), res_scale, axis)
    assert b_jax.shape==b.shape

    a = torch.randn((10, 20, 40, 50, 60))

    res_scale = [0.5, 3, 4]
    axis = [-3, -2, -1]
    b = resample(a, res_scale, axis)

    b_jax = np.array(flax_resample(a.detach().numpy(), res_scale, axis))

    compare_tensors(b.detach().numpy(),b_jax)


# test_ChannelMLP()
# test_LinearChannelMLP()
# test_GridEmbeddingND()
# test_Padding()
# test_SpectralConv()
# test_Normalizations()
# test_skip_connection_identity()
# test_softgating()
# test_flattened1dconv()
# test_resample()
# test_FNOBlock_resolution_scaling_factor()
test_FNO()
