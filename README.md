# Joint detection of AI-generated images and post-processing alterations

This project seeks to exploit the RRDataset (Li, Chunxiao et al. 2025) in order to train an double-headed model from a common backbone detecting jointly AI-generated images and the type of post-processing it went through (none/transfer on socials/redigitization).

Using ResNet18 as a backbone (mainly for computational limitations), this project proposes a comparison between the following setups:
- Unimodal baselines on each objective (1 head finetuned for 10 epochs)
- Joint detection model with two heads sharing the same weights. (10 epochs)
- Ablation study comparing different weightings of the shared CE loss between heads. (3 additional epochs on top of the balanced model)
- **Small personal innovation:** Joint detection model with adaptive weighting of the loss during training (13 epochs)

#### Main results (test set accuracy)

| Model | Real/Fake | Transform |
|---|---|---|
| Unimodal baseline (real/fake) | 90.68% | — |
| Unimodal baseline (transform) | — | 83.10% |
| Joint detection — balanced (α=β=1) | 90.61% | 81.80% |
| Joint detection — adaptive weighting | 90.68% | 77.56% |

One can run the `main.ipynb` notebook to reproduce the results of this experiment. Obtained results are further commented upon in the attached presentation slides.

### Instructions on reproducing results

1. Download RRDataset manually and unpack it in the `dataset/` folder. More instructions are provided in said folder.

2. If necessary, install all dependancies running this command:

```bash
pip install -r requirements.txt
```

3. Run the `main.ipynb` notebook to reproduce the experiment results.

**IMPORTANT NOTE:** It is not necessary to run section II. of the main notebook relative to model training. All pretrained models weights are provided in the `checkpoints/` folder and can be loaded directly from section III. Section II. was included for documentation purposes only and running it again could lead to slightly different results as our trainloader uses random crop and horizontal flip.

---
#### Bibliography

- Li, Chunxiao, et al. "Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection in Challenging Scenarios." *ICCV 2025*. [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/papers/Li_Bridging_the_Gap_Between_Ideal_and_Real-world_Evaluation_Benchmarking_AI-Generated_ICCV_2025_paper.pdf)

- Shao, Rui, Tianxing Wu, and Ziwei Liu. "Detecting and Grounding Multi-Modal Media Manipulation." *CVPR 2023*. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/papers/Shao_Detecting_and_Grounding_Multi-Modal_Media_Manipulation_CVPR_2023_paper.pdf)