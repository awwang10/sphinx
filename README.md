# Learning Belief Representations for Partially Observable Deep RL

## Introduction
This is the accompanying repository for the paper Learning Belief Representations for Partially Observable Deep RL.

By Andrew Wang*, Andrew C Li*, Toryn Q. Klassen, Rodrigo Toro Icarte, Sheila A. McIlraith


## Setup
```
git clone git@github.com:awwang10/sphinx.git
cd sphinx
pip install -r requirements.txt
```

## Running Experiments

Available environments: MiniGrid-Genie-8x8-v0, MiniGrid-NoisyTV-Genie-8x8-v0, MiniGrid-Lying-Genie-8x8-v0, MiniGrid-Modified-Cookie-9x9-v0, EscapeRoom-v0

### To collect data:
Genie Environment:
`python3 collect.py --env MiniGrid-Genie-8x8-v0 --episodes 3000`

Noisy TV Genie:
`python3 collect.py --env MiniGrid-NoisyTV-Genie-8x8-v0 --episodes 3000`

Lying Genie:
`python3 collect.py --env MiniGrid-Lying-Genie-8x8-v0 --episodes 3000`

Modified Cookie:
`python3 collect_modified_cookie.py --env MiniGrid-Modified-Cookie-9x9-v0 --episodes 1000`

Escape Room:
`python3 collect_escape.py --env EscapeRoom-v0 --episodes 500`

### To learn representations on collected data:
Genie Environment:
`python3 pretrain_representations.py --env MiniGrid-Genie-8x8-v0 --data-path collect_MiniGrid-Genie-8x8-v0.pt --epochs 1000 --batch-size 500 --beta 0.3 --dynamics-loss-s-coef 0.3 --dynamics-loss-o-coef 0.03 --reward-loss-coef 10.`

Noisy TV Genie:
`python3 pretrain_representations.py --env MiniGrid-NoisyTV-Genie-8x8-v0 --data-path collect_MiniGrid-NoisyTV-Genie-8x8-v0.pt --epochs 1000 --batch-size 500 --beta 0.3 --dynamics-loss-s-coef 0.3 --dynamics-loss-o-coef 0.03 --reward-loss-coef 10.`

Lying Genie:
`python3 pretrain_representations.py --env MiniGrid-Lying-Genie-8x8-v0 --data-path collect_MiniGrid-Lying-Genie-8x8-v0.pt --epochs 1000 --batch-size 500 --beta 0.3 --dynamics-loss-s-coef 0.3 --dynamics-loss-o-coef 0.03 --reward-loss-coef 10.`

Modified Cookie:
`python3 pretrain_representations.py --env MiniGrid-Modified-Cookie-9x9-v0 --data-path collect_MiniGrid-Modified-Cookie-9x9-v0.pt --epochs 100 --batch-size 500 --beta 0.03 --dynamics-loss-s-coef 0.1 --dynamics-loss-o-coef 0.1 --reward-loss-coef 300.`

Escape Room:
`python3 pretrain_representations.py --env EscapeRoom-v0 --data-path collect_EscapeRoom-v0.pt --epochs 300 --batch-size 500 --beta 0.03 --dynamics-loss-s-coef 0.1 --dynamics-loss-o-coef 0.003 --reward-loss-coef 100.`


### To pretrain the VAE on collected data:
Genie Environment:
`python3 -m pretrain_vae --algo belief_vae --env MiniGrid-Genie-8x8-v0 --save-interval 50 --epochs_g 3000 --lr-g 0.0003 --latent-dim-f 16 --data-path collect_MiniGrid-Genie-8x8-v0.pt`

Noisy TV Genie:
`python3 -m pretrain_vae --algo belief_vae --env MiniGrid-NoisyTV-Genie-8x8-v0 --save-interval 50 --epochs_g 5000 --lr-g 0.0003 --latent-dim-f 16 --data-path collect_MiniGrid-NoisyTV-Genie-8x8-v0.pt`

Lying Genie:
`python3 -m pretrain_vae --algo belief_vae --env MiniGrid-Lying-Genie-8x8-v0 --save-interval 50 --epochs_g 5000 --lr-g 0.0003 --latent-dim-f 16 --data-path collect_MiniGrid-Lying-Genie-8x8-v0.pt`

Modified Cookie:
`python3 -m pretrain_vae --algo belief_vae --env MiniGrid-Modified-Cookie-9x9-v0 --save-interval 50 --epochs_g 3000 --lr-g 0.0003 --latent-dim-f 16 --latent-dim-vae 64 --data-path collect_MiniGrid-Modified-Cookie-9x9-v0.pt`

Escape Room:
`python3 -m pretrain_vae --algo belief_vae --env EscapeRoom-v0 --save-interval 50 --batch-size-g 100 --epochs_g 5000 --lr-g 0.0003 --latent-dim-f 16 --data-path collect_EscapeRoom-v0.pt`

### To train the RL policy:
Genie Environment: `python3 -m train --algo belief_vae --env MiniGrid-Genie-8x8-v0 --save-interval 50 --frames 5000000 --procs 32 --recurrence 1 --frames_per_proc 256 --batch-size 2048 --epochs 24 --epochs_g 8 --lr-g 0.0003 --lr 0.0005 --entropy-coef 0.03 --latent-dim-vae 32 --latent-dim-f 16`

Noisy TV Genie: `python3 -m train --algo belief_vae --env MiniGrid-NoisyTV-Genie-8x8-v0 --save-interval 50 --frames 5000000 --procs 32 --recurrence 1 --frames_per_proc 256 --batch-size 2048 --epochs 24 --epochs_g 8 --lr-g 0.0003 --lr 0.0005 --entropy-coef 0.03 --latent-dim-vae 32 --latent-dim-f 16`

Lying Genie: `python3 -m train --algo belief_vae --env MiniGrid-Lying-Genie-8x8-v0 --save-interval 50 --frames 5000000 --procs 32 --recurrence 1 --frames_per_proc 256 --batch-size 2048 --epochs 24 --epochs_g 8 --lr-g 0.0003 --lr 0.0005 --entropy-coef 0.03 --latent-dim-vae 32 --latent-dim-f 16`

Modified Cookie: `python3 -m train --algo belief_vae --env MiniGrid-Modified-Cookie-9x9-v0 --save-interval 50 --frames 10000000 --procs 32 --recurrence 1 --frames_per_proc 512 --batch-size 4096 --batch-size-g 4096 --epochs 8 --epochs_g 8 --lr-g 0.001 --lr 0.001 --entropy-coef 0.003 --latent-dim-vae 64 --latent-dim-f 16 --discount 0.97`

Escape Room: `python3 -m train --algo belief_vae --env EscapeRoom-v0 --save-interval 50 --frames 5000000 --procs 32 --recurrence 1 --frames_per_proc 256 --batch-size 2048 --epochs 8 --batch-size-g 2048 --epochs_g 8 --lr-g 0.0003 --lr 0.0005 --entropy-coef 0.01 --latent-dim-vae 32 --latent-dim-f 16`