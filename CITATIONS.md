# Citations

External resources used by this repository, with BibTeX entries. The VAE and
CAE anomaly detectors (`guided_diffusion/autoencoder.py`) are in-repo PyTorch
implementations following the architectures and methods of the references
below.

## Diffusion anomaly detection (the original method this repo implements)

Wolleb et al. — the diffusion-model anomaly-detection method this codebase is
based on.

```bibtex
@inproceedings{wolleb2022diffusion,
  title     = {Diffusion Models for Medical Anomaly Detection},
  author    = {Wolleb, Julia and Bieder, Florentin and Sandk{\"u}hler, Robin and Cattin, Philippe C.},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022},
  pages     = {35--45},
  year      = {2022},
  publisher = {Springer},
  doi       = {10.1007/978-3-031-16452-1_4},
  note      = {arXiv:2203.04306}
}
```

## VAE

Kingma & Welling — the variational autoencoder. `VAEModel` implements the
standard convolutional VAE.

```bibtex
@inproceedings{kingma2014auto,
  title     = {Auto-Encoding Variational {B}ayes},
  author    = {Kingma, Diederik P. and Welling, Max},
  booktitle = {2nd International Conference on Learning Representations (ICLR)},
  year      = {2014},
  note      = {arXiv:1312.6114}
}
```

An & Cho — anomaly detection with VAEs via reconstruction-based scoring.
`VAEModel.reconstruct` uses the deterministic mean reconstruction and
`anomaly_map` the reconstruction residual, following this approach.

```bibtex
@techreport{an2015variational,
  title       = {Variational Autoencoder based Anomaly Detection using Reconstruction Probability},
  author      = {An, Jinwon and Cho, Sungzoon},
  institution = {Seoul National University Data Mining Center},
  series      = {Special Lecture on IE},
  volume      = {2},
  number      = {1},
  pages       = {1--18},
  year        = {2015}
}
```

AntixK/PyTorch-VAE — widely used PyTorch VAE reference implementation
(7.7k stars / 1.2k forks at time of writing; Apache-2.0). The `VAEModel`
architecture (stride-2 Conv/BatchNorm/LeakyReLU encoder, fully-connected
mu/logvar heads, mirrored ConvTranspose decoder, tanh output) follows its
`models/vanilla_vae.py`.

```bibtex
@misc{subramanian2020pytorchvae,
  title        = {{PyTorch-VAE}: A Collection of Variational Autoencoders ({VAE}) in {PyTorch}},
  author       = {Subramanian, A. K.},
  year         = {2020},
  howpublished = {\url{https://github.com/AntixK/PyTorch-VAE}},
  note         = {Apache License 2.0}
}
```

## CAE

Bergmann et al. — the canonical convolutional-autoencoder baseline for
unsupervised anomaly/defect segmentation via reconstruction residuals,
including the SSIM loss variant. `CAEModel` (and its optional `ssim_weight`
term) follows this work.

```bibtex
@inproceedings{bergmann2019improving,
  title     = {Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders},
  author    = {Bergmann, Paul and L{\"o}we, Sindy and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle = {Proceedings of the 14th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP) -- Volume 5: VISAPP},
  pages     = {372--380},
  year      = {2019},
  doi       = {10.5220/0007364503720380},
  note      = {arXiv:1807.02011}
}
```

Baur et al. — comparative study of AE/VAE architectures for unsupervised
anomaly segmentation in medical images; the dense-bottleneck option of
`CAEModel` (`spatial_latent=False`) follows the AE baselines surveyed here.

```bibtex
@article{baur2021autoencoders,
  title   = {Autoencoders for Unsupervised Anomaly Segmentation in Brain {MR} Images: A Comparative Study},
  author  = {Baur, Christoph and Denner, Stefan and Wiestler, Benedikt and Navab, Nassir and Albarqouni, Shadi},
  journal = {Medical Image Analysis},
  volume  = {69},
  pages   = {101952},
  year    = {2021},
  doi     = {10.1016/j.media.2020.101952},
  note    = {arXiv:2004.03271}
}
```

SSIM (used by the optional CAE loss term):

```bibtex
@article{wang2004image,
  title   = {Image Quality Assessment: From Error Visibility to Structural Similarity},
  author  = {Wang, Zhou and Bovik, Alan C. and Sheikh, Hamid R. and Simoncelli, Eero P.},
  journal = {IEEE Transactions on Image Processing},
  volume  = {13},
  number  = {4},
  pages   = {600--612},
  year    = {2004},
  doi     = {10.1109/TIP.2003.819861}
}
```
